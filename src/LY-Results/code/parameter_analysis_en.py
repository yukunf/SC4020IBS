import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from src.brute_force_search import BruteForceSearch
from src.lsh_search import LSHIndex
import os

def analyze_lsh_parameters():
    """Analyze the impact of LSH parameters on performance"""
    print("=== LSH Parameter Analysis ===")
    
    # Load data
    vectors = np.load('data/fmnist_resnet50_vectors.npy')
    labels = np.load('data/fmnist_resnet50_labels.npy')
    
    # Select smaller test set for faster analysis
    test_size = 500
    query_indices = np.random.choice(len(vectors), test_size, replace=False)
    query_vectors = vectors[query_indices]
    query_labels = labels[query_indices]
    
    # Parameter grid
    num_tables_range = [5, 10, 15, 20, 25, 30]
    hash_size_range = [8, 10, 12, 14, 16]
    
    results = []
    
    print("Starting parameter grid search...")
    for num_tables in num_tables_range:
        for hash_size in hash_size_range:
            print(f"Testing configuration: Tables={num_tables}, Hash_size={hash_size}")
            
            # Build LSH index
            lsh = LSHIndex(hash_family='random_projection', 
                          num_tables=num_tables, 
                          hash_size=hash_size)
            
            # Measure build time
            start_time = time.time()
            lsh.build_index(vectors)
            build_time = time.time() - start_time
            
            # Measure search time and accuracy
            start_time = time.time()
            distances, indices = lsh.search(query_vectors, 50, 'cosine')
            search_time = time.time() - start_time
            
            avg_query_time = (search_time / len(query_vectors)) * 1000  # ms
            
            # Calculate recall
            correct = 0
            for i, query_label in enumerate(query_labels):
                retrieved_labels = labels[indices[i][:10]]  # Top-10
                if query_label in retrieved_labels:
                    correct += 1
            recall_at_10 = (correct / len(query_labels)) * 100
            
            # Get statistics
            stats = lsh.get_stats()
            
            results.append({
                'num_tables': num_tables,
                'hash_size': hash_size,
                'build_time': build_time,
                'avg_query_time_ms': avg_query_time,
                'recall_at_10': recall_at_10,
                'total_memory_mb': stats['total_memory_mb'],
                'avg_bucket_size': stats['avg_bucket_size'],
                'total_buckets': stats['total_buckets']
            })
    
    return pd.DataFrame(results)

def analyze_data_size_scalability():
    """Analyze performance under different data scales"""
    print("\n=== Data Size Scalability Analysis ===")
    
    # Load full data
    full_vectors = np.load('data/fmnist_resnet50_vectors.npy')
    full_labels = np.load('data/fmnist_resnet50_labels.npy')
    
    # Different data sizes
    data_sizes = [1000, 5000, 10000, 20000, 50000, 70000]
    test_query_size = 200
    
    results = []
    
    for data_size in data_sizes:
        print(f"Testing data size: {data_size}")
        
        # Select data subset
        indices = np.random.choice(len(full_vectors), data_size, replace=False)
        vectors = full_vectors[indices]
        labels = full_labels[indices]
        
        # Select queries
        query_indices = np.random.choice(len(vectors), min(test_query_size, len(vectors)//10), replace=False)
        query_vectors = vectors[query_indices]
        query_labels = labels[query_indices]
        
        # 1. Brute Force
        bf = BruteForceSearch('cosine')
        start_time = time.time()
        bf.build_index(vectors)
        bf_build_time = time.time() - start_time
        
        start_time = time.time()
        bf_distances, bf_indices = bf.search(query_vectors, 10)
        bf_search_time = time.time() - start_time
        bf_avg_query_time = (bf_search_time / len(query_vectors)) * 1000
        
        # 2. LSH
        lsh = LSHIndex(hash_family='random_projection', num_tables=10, hash_size=12)
        start_time = time.time()
        lsh.build_index(vectors)
        lsh_build_time = time.time() - start_time
        
        start_time = time.time()
        lsh_distances, lsh_indices = lsh.search(query_vectors, 10, 'cosine')
        lsh_search_time = time.time() - start_time
        lsh_avg_query_time = (lsh_search_time / len(query_vectors)) * 1000
        
        # Calculate LSH recall
        correct = 0
        for i, query_label in enumerate(query_labels):
            retrieved_labels = labels[lsh_indices[i][:10]]
            if query_label in retrieved_labels:
                correct += 1
        lsh_recall = (correct / len(query_labels)) * 100
        
        results.append({
            'data_size': data_size,
            'bf_build_time': bf_build_time,
            'bf_avg_query_time_ms': bf_avg_query_time,
            'lsh_build_time': lsh_build_time,
            'lsh_avg_query_time_ms': lsh_avg_query_time,
            'lsh_recall_at_10': lsh_recall,
            'speedup_factor': bf_avg_query_time / lsh_avg_query_time if lsh_avg_query_time > 0 else 0
        })
    
    return pd.DataFrame(results)

def create_parameter_analysis_plots(param_df, scale_df):
    """Create parameter analysis visualization plots"""
    print("\n=== Generating Analysis Plots ===")
    
    # Create output directory
    os.makedirs('results/analysis', exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. LSH parameter heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LSH Parameter Analysis', fontsize=16)
    
    # Build time heatmap
    pivot_build_time = param_df.pivot(index='num_tables', columns='hash_size', values='build_time')
    sns.heatmap(pivot_build_time, annot=True, fmt='.2f', ax=axes[0,0], cmap='YlOrRd')
    axes[0,0].set_title('Build Time (seconds)')
    axes[0,0].set_xlabel('Hash Size')
    axes[0,0].set_ylabel('Number of Tables')
    
    # Query time heatmap
    pivot_query_time = param_df.pivot(index='num_tables', columns='hash_size', values='avg_query_time_ms')
    sns.heatmap(pivot_query_time, annot=True, fmt='.1f', ax=axes[0,1], cmap='YlOrRd')
    axes[0,1].set_title('Average Query Time (ms)')
    axes[0,1].set_xlabel('Hash Size')
    axes[0,1].set_ylabel('Number of Tables')
    
    # Recall heatmap
    pivot_recall = param_df.pivot(index='num_tables', columns='hash_size', values='recall_at_10')
    sns.heatmap(pivot_recall, annot=True, fmt='.1f', ax=axes[0,2], cmap='YlGnBu')
    axes[0,2].set_title('Recall@10 (%)')
    axes[0,2].set_xlabel('Hash Size')
    axes[0,2].set_ylabel('Number of Tables')
    
    # Memory usage heatmap
    pivot_memory = param_df.pivot(index='num_tables', columns='hash_size', values='total_memory_mb')
    sns.heatmap(pivot_memory, annot=True, fmt='.1f', ax=axes[1,0], cmap='YlOrRd')
    axes[1,0].set_title('Memory Usage (MB)')
    axes[1,0].set_xlabel('Hash Size')
    axes[1,0].set_ylabel('Number of Tables')
    
    # Average bucket size heatmap
    pivot_bucket = param_df.pivot(index='num_tables', columns='hash_size', values='avg_bucket_size')
    sns.heatmap(pivot_bucket, annot=True, fmt='.1f', ax=axes[1,1], cmap='YlGnBu')
    axes[1,1].set_title('Average Bucket Size')
    axes[1,1].set_xlabel('Hash Size')
    axes[1,1].set_ylabel('Number of Tables')
    
    # Parameter tradeoff scatter plot
    scatter = axes[1,2].scatter(param_df['avg_query_time_ms'], param_df['recall_at_10'], 
                               c=param_df['num_tables'], s=param_df['hash_size']*10, 
                               alpha=0.7, cmap='viridis')
    axes[1,2].set_xlabel('Average Query Time (ms)')
    axes[1,2].set_ylabel('Recall@10 (%)')
    axes[1,2].set_title('Parameter Tradeoff Analysis\n(Color=Tables, Size=Hash Size)')
    plt.colorbar(scatter, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig('results/analysis/lsh_parameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Scalability analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Size Scalability Analysis', fontsize=16)
    
    # Build time comparison
    axes[0,0].plot(scale_df['data_size'], scale_df['bf_build_time'], 'o-', label='Brute Force', linewidth=2)
    axes[0,0].plot(scale_df['data_size'], scale_df['lsh_build_time'], 's-', label='LSH', linewidth=2)
    axes[0,0].set_xlabel('Data Size')
    axes[0,0].set_ylabel('Build Time (seconds)')
    axes[0,0].set_title('Index Build Time')
    axes[0,0].legend()
    axes[0,0].set_yscale('log')
    
    # Query time comparison
    axes[0,1].plot(scale_df['data_size'], scale_df['bf_avg_query_time_ms'], 'o-', label='Brute Force', linewidth=2)
    axes[0,1].plot(scale_df['data_size'], scale_df['lsh_avg_query_time_ms'], 's-', label='LSH', linewidth=2)
    axes[0,1].set_xlabel('Data Size')
    axes[0,1].set_ylabel('Average Query Time (ms)')
    axes[0,1].set_title('Query Time')
    axes[0,1].legend()
    axes[0,1].set_yscale('log')
    
    # Speedup ratio
    axes[1,0].plot(scale_df['data_size'], scale_df['speedup_factor'], 'g^-', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('Data Size')
    axes[1,0].set_ylabel('Speedup Ratio (BF/LSH)')
    axes[1,0].set_title('LSH Speedup vs Brute Force')
    axes[1,0].grid(True, alpha=0.3)
    
    # LSH recall
    axes[1,1].plot(scale_df['data_size'], scale_df['lsh_recall_at_10'], 'ro-', linewidth=2)
    axes[1,1].set_xlabel('Data Size')
    axes[1,1].set_ylabel('Recall@10 (%)')
    axes[1,1].set_title('LSH Recall Rate')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim([90, 105])
    
    plt.tight_layout()
    plt.savefig('results/analysis/scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_analysis_report(param_df, scale_df):
    """Generate detailed analysis report"""
    print("\n=== Generating Analysis Report ===")
    
    report = []
    
    report.append("# Similarity Search Algorithm Parameter Analysis Report\n")
    report.append(f"Generated: {pd.Timestamp.now()}\n")
    
    report.append("## 1. LSH Parameter Analysis Summary\n")
    
    # Find optimal configurations
    best_config = param_df.loc[param_df['recall_at_10'].idxmax()]
    fastest_config = param_df.loc[param_df['avg_query_time_ms'].idxmin()]
    
    report.append("### 1.1 Best Recall Configuration")
    report.append(f"- Number of Tables: {best_config['num_tables']}")
    report.append(f"- Hash Size: {best_config['hash_size']}")
    report.append(f"- Recall@10: {best_config['recall_at_10']:.2f}%")
    report.append(f"- Query Time: {best_config['avg_query_time_ms']:.2f}ms")
    report.append(f"- Build Time: {best_config['build_time']:.2f}s")
    report.append(f"- Memory Usage: {best_config['total_memory_mb']:.2f}MB\n")
    
    report.append("### 1.2 Fastest Query Configuration")
    report.append(f"- Number of Tables: {fastest_config['num_tables']}")
    report.append(f"- Hash Size: {fastest_config['hash_size']}")
    report.append(f"- Recall@10: {fastest_config['recall_at_10']:.2f}%")
    report.append(f"- Query Time: {fastest_config['avg_query_time_ms']:.2f}ms")
    report.append(f"- Build Time: {fastest_config['build_time']:.2f}s")
    report.append(f"- Memory Usage: {fastest_config['total_memory_mb']:.2f}MB\n")
    
    report.append("### 1.3 Parameter Impact Analysis")
    report.append("**Impact of Number of Tables:**")
    # 直接在原始数据框上计算相关性，而不是对聚合后的均值计算
    report.append(f"- Build time correlation with table count: {param_df['num_tables'].corr(param_df['build_time']):.3f}")
    report.append(f"- Query time correlation with table count: {param_df['num_tables'].corr(param_df['avg_query_time_ms']):.3f}")
    report.append(f"- Recall correlation with table count: {param_df['num_tables'].corr(param_df['recall_at_10']):.3f}\n")
    
    report.append("**Impact of Hash Size:**")
    # 直接在原始数据框上计算相关性，而不是对聚合后的均值计算
    report.append(f"- Build time correlation with hash size: {param_df['hash_size'].corr(param_df['build_time']):.3f}")
    report.append(f"- Query time correlation with hash size: {param_df['hash_size'].corr(param_df['avg_query_time_ms']):.3f}")
    report.append(f"- Recall correlation with hash size: {param_df['hash_size'].corr(param_df['recall_at_10']):.3f}\n")
    
    report.append("## 2. Scalability Analysis Summary\n")
    
    # Results at maximum data size
    max_scale = scale_df.loc[scale_df['data_size'].idxmax()]
    
    report.append("### 2.1 Large Scale Data Performance")
    report.append(f"- Maximum test size: {max_scale['data_size']:,} vectors")
    report.append(f"- Brute force query time: {max_scale['bf_avg_query_time_ms']:.2f}ms")
    report.append(f"- LSH query time: {max_scale['lsh_avg_query_time_ms']:.2f}ms")
    report.append(f"- Speedup ratio: {max_scale['speedup_factor']:.2f}x")
    report.append(f"- LSH recall: {max_scale['lsh_recall_at_10']:.2f}%\n")
    
    report.append("### 2.2 Scalability Conclusions")
    # Calculate time complexity
    bf_time_growth = np.polyfit(np.log(scale_df['data_size']), np.log(scale_df['bf_avg_query_time_ms']), 1)[0]
    lsh_time_growth = np.polyfit(np.log(scale_df['data_size']), np.log(scale_df['lsh_avg_query_time_ms']), 1)[0]
    
    report.append(f"- Brute force time complexity estimate: O(n^{bf_time_growth:.2f})")
    report.append(f"- LSH time complexity estimate: O(n^{lsh_time_growth:.2f})")
    report.append(f"- LSH performs better on large-scale data\n")
    
    report.append("## 3. Practical Recommendations\n")
    report.append("### 3.1 Algorithm Selection Guidelines")
    report.append("- **Small data (<10K)**: Use brute force, simple and efficient")
    report.append("- **Medium data (10K-50K)**: Use LSH, balance performance and accuracy")
    report.append("- **Large data (>50K)**: Must use LSH or FAISS, brute force infeasible\n")
    
    report.append("### 3.2 LSH Parameter Tuning Guidelines")
    report.append("- **High accuracy requirements**: Use more hash tables (20-30) and longer hashes (12-16)")
    report.append("- **Fast query requirements**: Use fewer hash tables (5-10) and shorter hashes (8-10)")
    report.append("- **Memory constraints**: Reduce number of hash tables, accept some accuracy loss")
    report.append("- **Build time sensitive**: Avoid too many hash tables, prioritize query time optimization\n")
    
    # Save report
    with open('results/analysis/analysis_report_en.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("Analysis report saved to: results/analysis/analysis_report_en.md")

def main():
    """Main function"""
    print("Starting parameter analysis and performance evaluation...")
    
    # Create results directory
    os.makedirs('results/analysis', exist_ok=True)
    
    # 1. LSH parameter analysis
    param_df = analyze_lsh_parameters()
    param_df.to_csv('results/analysis/lsh_parameter_analysis_en.csv', index=False)
    print("LSH parameter analysis completed, results saved")
    
    # 2. Scalability analysis
    scale_df = analyze_data_size_scalability()
    scale_df.to_csv('results/analysis/scalability_analysis_en.csv', index=False)
    print("Scalability analysis completed, results saved")
    
    # 3. Generate visualization plots
    create_parameter_analysis_plots(param_df, scale_df)
    
    # 4. Generate analysis report
    generate_analysis_report(param_df, scale_df)
    
    print("\n=== Parameter Analysis Complete ===")
    print("All results saved to: results/analysis/")
    print("- LSH parameter analysis: lsh_parameter_analysis_en.csv")
    print("- Scalability analysis: scalability_analysis_en.csv")
    print("- Parameter analysis plots: lsh_parameter_analysis.png")
    print("- Scalability plots: scalability_analysis.png")
    print("- Detailed report: analysis_report_en.md")

if __name__ == '__main__':
    main()


