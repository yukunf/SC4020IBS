import numpy as np
import time
import json
import pandas as pd
from typing import Tuple, List, Dict, Any
from tqdm import tqdm

class BruteForceSearch:
    """暴力搜索算法实现
    
    支持欧氏距离和余弦相似度两种度量方式
    """
    
    def __init__(self, metric='cosine'):
        """
        Args:
            metric (str): 距离度量方式，'cosine' 或 'euclidean'
        """
        self.metric = metric
        self.gallery_vectors = None
        self.gallery_metadata = None
        self.is_built = False
        
    def build_index(self, vectors: np.ndarray, metadata: List[Dict] = None):
        """构建搜索索引（对于暴力搜索就是存储向量）
        
        Args:
            vectors: shape (N, D) 的特征向量
            metadata: 可选的元数据信息
        """
        print(f"Building brute force index with {len(vectors)} vectors...")
        start_time = time.time()
        
        self.gallery_vectors = vectors.astype(np.float32)
        self.gallery_metadata = metadata
        
        # 对于余弦相似度，预先归一化向量
        if self.metric == 'cosine':
            norms = np.linalg.norm(self.gallery_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除零
            self.gallery_vectors = self.gallery_vectors / norms
            
        self.is_built = True
        build_time = time.time() - start_time
        print(f"Index built in {build_time:.4f} seconds")
        return build_time
        
    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """执行搜索
        
        Args:
            query_vectors: shape (Q, D) 的查询向量
            k: 返回的近邻数量
            
        Returns:
            distances: shape (Q, k) 的距离
            indices: shape (Q, k) 的索引
        """
        if not self.is_built:
            raise ValueError("Index not built. Call build_index() first.")
            
        print(f"Searching {len(query_vectors)} queries with k={k}...")
        start_time = time.time()
        
        query_vectors = query_vectors.astype(np.float32)
        
        # 归一化查询向量（如果使用余弦相似度）
        if self.metric == 'cosine':
            norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query_vectors = query_vectors / norms
            
        # 计算所有查询与所有gallery向量的距离
        if self.metric == 'cosine':
            # 余弦相似度 = 点积（因为向量已归一化）
            # 距离 = 1 - 相似度
            similarities = query_vectors @ self.gallery_vectors.T
            distances = 1 - similarities
        else:  # euclidean
            # 使用广播计算欧氏距离
            distances = np.sqrt(((query_vectors[:, np.newaxis, :] - 
                                self.gallery_vectors[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # 获取top-k索引
        indices = np.argpartition(distances, k, axis=1)[:, :k]
        
        # 对top-k结果进行排序
        for i in range(len(query_vectors)):
            sorted_idx = np.argsort(distances[i, indices[i]])
            indices[i] = indices[i][sorted_idx]
            
        # 获取对应的距离值
        result_distances = np.array([distances[i, indices[i]] for i in range(len(query_vectors))])
        
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.4f} seconds")
        
        return result_distances, indices
        
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if not self.is_built:
            return {"status": "not_built"}
            
        return {
            "algorithm": "BruteForce",
            "metric": self.metric,
            "gallery_size": len(self.gallery_vectors),
            "vector_dim": self.gallery_vectors.shape[1],
            "memory_mb": self.gallery_vectors.nbytes / (1024 * 1024)
        }


def evaluate_brute_force(vectors_path: str, labels_path: str = None, 
                        metadata_path: str = None, k_values: List[int] = [1, 10, 50],
                        metric: str = 'cosine', test_size: int = 1000):
    """评估暴力搜索算法
    
    Args:
        vectors_path: 向量文件路径
        labels_path: 标签文件路径（用于Fashion-MNIST）
        metadata_path: 元数据文件路径（用于DeepFashion）
        k_values: 评估的k值列表
        metric: 距离度量方式
        test_size: 测试查询数量
    """
    print(f"\n=== Brute Force Search Evaluation ===")
    print(f"Metric: {metric}")
    print(f"K values: {k_values}")
    
    # 1. 加载数据
    print("Loading data...")
    vectors = np.load(vectors_path)
    print(f"Loaded {len(vectors)} vectors of dimension {vectors.shape[1]}")
    
    # 2. 准备查询和gallery
    if test_size > len(vectors):
        test_size = len(vectors) // 10
    
    # 随机选择查询
    query_indices = np.random.choice(len(vectors), test_size, replace=False)
    query_vectors = vectors[query_indices]
    
    # 3. 构建索引
    searcher = BruteForceSearch(metric=metric)
    build_time = searcher.build_index(vectors)
    
    # 4. 执行搜索
    max_k = max(k_values)
    distances, indices = searcher.search(query_vectors, max_k)
    
    # 5. 计算评估指标
    print("\n=== Results ===")
    print(f"Index build time: {build_time:.4f}s")
    
    # 计算平均查询时间
    start_time = time.time()
    _, _ = searcher.search(query_vectors[:100], max_k)  # 小规模测试时延
    avg_query_time = (time.time() - start_time) / 100 * 1000  # ms
    qps = 1000 / avg_query_time if avg_query_time > 0 else float('inf')
    
    print(f"Average query time: {avg_query_time:.4f}ms")
    print(f"QPS: {qps:.2f}")
    
    # 内存占用
    stats = searcher.get_stats()
    print(f"Memory usage: {stats['memory_mb']:.2f}MB")
    
    # 如果有标签数据，计算召回率
    if labels_path and labels_path.endswith('.npy'):
        labels = np.load(labels_path)
        query_labels = labels[query_indices]
        
        print("\n=== Recall@K (same class) ===")
        for k in k_values:
            correct = 0
            for i, query_label in enumerate(query_labels):
                retrieved_labels = labels[indices[i][:k]]
                if query_label in retrieved_labels:
                    correct += 1
            recall = correct / len(query_labels) * 100
            print(f"Recall@{k}: {recall:.2f}%")
    
    return {
        'build_time': build_time,
        'avg_query_time_ms': avg_query_time,
        'qps': qps,
        'memory_mb': stats['memory_mb'],
        'stats': stats
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Brute Force Search Evaluation")
    parser.add_argument('--vectors', required=True, help="Path to vectors .npy file")
    parser.add_argument('--labels', help="Path to labels .npy file (for FMNIST)")
    parser.add_argument('--metadata', help="Path to metadata .json file (for DeepFashion)")
    parser.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--k', type=int, nargs='+', default=[1, 10, 50])
    parser.add_argument('--test_size', type=int, default=1000)
    
    args = parser.parse_args()
    
    evaluate_brute_force(
        vectors_path=args.vectors,
        labels_path=args.labels,
        metadata_path=args.metadata,
        k_values=args.k,
        metric=args.metric,
        test_size=args.test_size
    )

