import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any
import os

from brute_force_search import BruteForceSearch
from lsh_search import LSHIndex
import sys
sys.path.append('code')
from src.inshop_evaluation import evaluate_inshop_retrieval
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /path/to/project/src
PROJECT_DIR = os.path.dirname(BASE_DIR)
class UnifiedEvaluator:
    """Unified evaluation system for comparing three algorithm performances"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_all_algorithms(self, vectors_path: str, labels_path: str = None,
                               metadata_path: str = None, k_values: List[int] = [1, 10, 50],
                               metric: str = 'cosine', test_size: int = 1000):
        """Evaluate all algorithms"""
        print(f"\n{'='*60}")
        print(f"UNIFIED EVALUATION - {os.path.basename(vectors_path)}")
        print(f"{'='*60}")
        
        # Load data
        vectors = np.load(vectors_path)
        print(f"Dataset: {len(vectors)} vectors, {vectors.shape[1]} dimensions")
        
        if test_size > len(vectors):
            test_size = len(vectors) // 10
            
        # Prepare queries
        query_indices = np.random.choice(len(vectors), test_size, replace=False)
        query_vectors = vectors[query_indices]
        
        results = {}
        
        # 1. Brute Force
        print(f"\n{'-'*20} BRUTE FORCE {'-'*20}")
        bf_searcher = BruteForceSearch(metric=metric)
        bf_build_time = bf_searcher.build_index(vectors)
        
        start_time = time.time()
        bf_distances, bf_indices = bf_searcher.search(query_vectors, max(k_values))
        bf_search_time = time.time() - start_time
        
        results['brute_force'] = {
            'build_time': bf_build_time,
            'search_time': bf_search_time,
            'avg_query_time_ms': (bf_search_time / len(query_vectors)) * 1000,
            'qps': len(query_vectors) / bf_search_time,
            'memory_mb': bf_searcher.get_stats()['memory_mb'],
            'distances': bf_distances,
            'indices': bf_indices
        }
        
        # 2. LSH with different parameters
        lsh_configs = [
            {'num_tables': 5, 'hash_size': 8},
            {'num_tables': 10, 'hash_size': 10},
            {'num_tables': 20, 'hash_size': 12},
        ]
        
        for i, config in enumerate(lsh_configs):
            print(f"\n{'-'*15} LSH Config {i+1} (T={config['num_tables']}, H={config['hash_size']}) {'-'*15}")
            
            hash_family = 'random_projection' if metric == 'cosine' else 'e2lsh'
            lsh = LSHIndex(hash_family=hash_family, **config)
            lsh_build_time = lsh.build_index(vectors)
            
            start_time = time.time()
            lsh_distances, lsh_indices = lsh.search(query_vectors, max(k_values), metric)
            lsh_search_time = time.time() - start_time
            
            lsh_stats = lsh.get_stats()
            results[f'lsh_config_{i+1}'] = {
                'build_time': lsh_build_time,
                'search_time': lsh_search_time,
                'avg_query_time_ms': (lsh_search_time / len(query_vectors)) * 1000,
                'qps': len(query_vectors) / lsh_search_time,
                'memory_mb': lsh_stats['total_memory_mb'],
                'distances': lsh_distances,
                'indices': lsh_indices,
                'config': config,
                'avg_bucket_size': lsh_stats['avg_bucket_size']
            }
        
        # 3. Calculate accuracy compared to Brute Force
        self._calculate_accuracy(results, bf_indices, k_values)
        
        # 4. Calculate Recall@K if labels available
        if labels_path and labels_path.endswith('.npy'):
            labels = np.load(labels_path)
            query_labels = labels[query_indices]
            self._calculate_recall(results, labels, query_labels, k_values)
        
        # 5. Generate reports
        self._generate_performance_report(results, k_values, vectors_path)
        self._generate_plots(results, k_values, vectors_path)
        
        return results
        
    def _calculate_accuracy(self, results: Dict, ground_truth_indices: np.ndarray, k_values: List[int]):
        """Calculate accuracy relative to brute force search"""
        print(f"\n{'-'*20} ACCURACY vs BRUTE FORCE {'-'*20}")
        
        for alg_name, alg_results in results.items():
            if alg_name == 'brute_force':
                continue
                
            alg_results['accuracy'] = {}
            
            for k in k_values:
                total_correct = 0
                total_queries = len(ground_truth_indices)
                
                for i in range(total_queries):
                    gt_top_k = set(ground_truth_indices[i][:k])
                    pred_top_k = set(alg_results['indices'][i][:k])
                    correct = len(gt_top_k.intersection(pred_top_k))
                    total_correct += correct
                
                accuracy = (total_correct / (total_queries * k)) * 100
                alg_results['accuracy'][k] = accuracy
                print(f"{alg_name} Accuracy@{k}: {accuracy:.2f}%")
                
    def _calculate_recall(self, results: Dict, labels: np.ndarray, 
                         query_labels: np.ndarray, k_values: List[int]):
        """Calculate Recall@K"""
        print(f"\n{'-'*20} RECALL@K (same class) {'-'*20}")
        
        for alg_name, alg_results in results.items():
            alg_results['recall'] = {}
            
            for k in k_values:
                correct = 0
                for i, query_label in enumerate(query_labels):
                    retrieved_labels = labels[alg_results['indices'][i][:k]]
                    if query_label in retrieved_labels:
                        correct += 1
                        
                recall = (correct / len(query_labels)) * 100
                alg_results['recall'][k] = recall
                print(f"{alg_name} Recall@{k}: {recall:.2f}%")
                
    def _generate_performance_report(self, results: Dict, k_values: List[int], dataset_name: str):
        """Generate performance report"""
        report = []
        
        for alg_name, alg_results in results.items():
            row = {
                'Algorithm': alg_name,
                'Build Time (s)': f"{alg_results['build_time']:.4f}",
                'Avg Query Time (ms)': f"{alg_results['avg_query_time_ms']:.4f}",
                'QPS': f"{alg_results['qps']:.2f}",
                'Memory (MB)': f"{alg_results['memory_mb']:.2f}"
            }
            
            # Add accuracy
            if 'accuracy' in alg_results:
                for k in k_values:
                    row[f'Accuracy@{k}'] = f"{alg_results['accuracy'][k]:.2f}%"
                    
            # Add recall
            if 'recall' in alg_results:
                for k in k_values:
                    row[f'Recall@{k}'] = f"{alg_results['recall'][k]:.2f}%"
                    
            # Add LSH specific parameters
            if 'config' in alg_results:
                row['Tables'] = alg_results['config']['num_tables']
                row['Hash Size'] = alg_results['config']['hash_size']
                row['Avg Bucket Size'] = f"{alg_results['avg_bucket_size']:.2f}"
                
            report.append(row)
        
        # Save as CSV
        df = pd.DataFrame(report)
        report_path = os.path.join(self.output_dir, f"performance_report_{os.path.basename(dataset_name)}_en.csv")
        df.to_csv(report_path, index=False)
        print(f"\nPerformance report saved to: {report_path}")
        
        # Print doc
        print(f"\n{'-'*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'-'*60}")
        print(df.to_string(index=False))
        
    def _generate_plots(self, results: Dict, k_values: List[int], dataset_name: str):
        """Generate visualization plots"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Algorithm Comparison - {os.path.basename(dataset_name)}', fontsize=16)
        
        # Prepare data
        algorithms = list(results.keys())
        
        # 1. Build Time
        build_times = [results[alg]['build_time'] for alg in algorithms]
        axes[0, 0].bar(algorithms, build_times)
        axes[0, 0].set_title('Build Time (seconds)')
        axes[0, 0].set_ylabel('Time (s)')
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Query Time
        query_times = [results[alg]['avg_query_time_ms'] for alg in algorithms]
        axes[0, 1].bar(algorithms, query_times)
        axes[0, 1].set_title('Average Query Time (ms)')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].set_yscale('log')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Memory Usage
        memory_usage = [results[alg]['memory_mb'] for alg in algorithms]
        axes[0, 2].bar(algorithms, memory_usage)
        axes[0, 2].set_title('Memory Usage (MB)')
        axes[0, 2].set_ylabel('Memory (MB)')
        plt.setp(axes[0, 2].xaxis.get_majorticklabels(), rotation=45)
        
        # 4. QPS
        qps_values = [results[alg]['qps'] for alg in algorithms]
        axes[1, 0].bar(algorithms, qps_values)
        axes[1, 0].set_title('Queries Per Second (QPS)')
        axes[1, 0].set_ylabel('QPS')
        axes[1, 0].set_yscale('log')
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 5. Accuracy@K (if available)
        if 'accuracy' in list(results.values())[1]:  # Skip brute force
            for k in k_values:
                accuracies = [results[alg]['accuracy'][k] if 'accuracy' in results[alg] else 100 
                             for alg in algorithms]
                axes[1, 1].plot(algorithms, accuracies, marker='o', label=f'k={k}')
            axes[1, 1].set_title('Accuracy@K vs Brute Force (%)')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].legend()
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 6. Recall@K (if available)
        if 'recall' in list(results.values())[0]:
            for k in k_values:
                recalls = [results[alg]['recall'][k] for alg in algorithms]
                axes[1, 2].plot(algorithms, recalls, marker='s', label=f'k={k}')
            axes[1, 2].set_title('Recall@K (%)')
            axes[1, 2].set_ylabel('Recall (%)')
            axes[1, 2].legend()
            plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"comparison_plots_{os.path.basename(dataset_name)}_en.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plots saved to: {plot_path}")


def main():
    """Main function - run complete evaluation"""
    evaluator = UnifiedEvaluator(output_dir="../../doc/report/LSHBFreports/results")
    
    # Evaluation configuration
    k_values = [1, 10, 50]
    test_size = 1000
    
    # 1. Evaluate Fashion-MNIST dataset
    fmnist_vectors_path = os.path.join(PROJECT_DIR, "data/fmnist_resnet50_vectors.npy")
    fmnist_labels_path = os.path.join(PROJECT_DIR, "data/fmnist_resnet50_labels.npy")
    
    if os.path.exists(fmnist_vectors_path):
        print("Evaluating Fashion-MNIST dataset...")
        fmnist_results = evaluator.evaluate_all_algorithms(
            vectors_path=fmnist_vectors_path,
            labels_path=fmnist_labels_path,
            k_values=k_values,
            metric='cosine',
            test_size=test_size
        )
    else:
        print(f"Fashion-MNIST vectors not found at {fmnist_vectors_path}")
    
    # 2. Evaluate DeepFashion dataset
    deepfashion_vectors_path = os.path.join(PROJECT_DIR, "data/inshop_clip_vectors_gallery.npy")
    deepfashion_metadata_path = os.path.join(PROJECT_DIR, "data/inshop_clip_ids_gallery.json")
    
    if os.path.exists(deepfashion_vectors_path):
        print("\nEvaluating DeepFashion dataset...")
        deepfashion_results = evaluator.evaluate_all_algorithms(
            vectors_path=deepfashion_vectors_path,
            metadata_path=deepfashion_metadata_path,
            k_values=k_values,
            metric='cosine',
            test_size=test_size
        )
    else:
        print(f"DeepFashion vectors not found at {deepfashion_vectors_path}")
        print("Please run inshop_embedder.py first to generate feature vectors.")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"Results saved in: results/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()


