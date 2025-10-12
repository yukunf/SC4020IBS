import numpy as np
import time
import json
import random
from typing import Tuple, List, Dict, Any, Set
from collections import defaultdict
from tqdm import tqdm

class LSHIndex:
    """局部敏感哈希索引实现
    
    支持多种哈希函数族：
    - Random Projection (适用于余弦相似度)
    - MinHash (适用于Jaccard相似度，这里用于二值化向量)
    - E2LSH (适用于欧氏距离)
    """
    
    def __init__(self, hash_family='random_projection', num_tables=10, 
                 hash_size=10, bucket_size=None):
        """
        Args:
            hash_family: 哈希函数族类型
            num_tables: 哈希表数量
            hash_size: 每个哈希函数的位数
            bucket_size: 桶大小限制（可选）
        """
        self.hash_family = hash_family
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.bucket_size = bucket_size
        
        self.hash_functions = []
        self.hash_tables = []
        self.vectors = None
        self.metadata = None
        self.is_built = False
        
    def _generate_random_projection_functions(self, dim: int):
        """生成随机投影哈希函数"""
        functions = []
        for _ in range(self.num_tables):
            # 每个表有hash_size个随机向量
            random_vectors = np.random.randn(self.hash_size, dim)
            random_vectors = random_vectors / np.linalg.norm(random_vectors, axis=1, keepdims=True)
            functions.append(random_vectors)
        return functions
        
    def _generate_e2lsh_functions(self, dim: int, r: float = 1.0):
        """生成E2LSH哈希函数（适用于欧氏距离）"""
        functions = []
        for _ in range(self.num_tables):
            # 每个表的哈希函数参数
            a = np.random.randn(self.hash_size, dim)  # 随机向量
            b = np.random.uniform(0, r, self.hash_size)  # 随机偏移
            functions.append((a, b, r))
        return functions
        
    def _hash_vector_random_projection(self, vector: np.ndarray, table_idx: int) -> str:
        """使用随机投影哈希向量"""
        projections = self.hash_functions[table_idx] @ vector
        binary_hash = (projections > 0).astype(int)
        return ''.join(map(str, binary_hash))
        
    def _hash_vector_e2lsh(self, vector: np.ndarray, table_idx: int) -> str:
        """使用E2LSH哈希向量"""
        a, b, r = self.hash_functions[table_idx]
        projections = (a @ vector + b) / r
        hash_values = np.floor(projections).astype(int)
        return ','.join(map(str, hash_values))
        
    def build_index(self, vectors: np.ndarray, metadata: List[Dict] = None):
        """构建LSH索引"""
        print(f"Building LSH index with {self.num_tables} tables, {self.hash_size} bits each...")
        start_time = time.time()
        
        self.vectors = vectors.astype(np.float32)
        self.metadata = metadata
        dim = vectors.shape[1]
        
        # 生成哈希函数
        if self.hash_family == 'random_projection':
            self.hash_functions = self._generate_random_projection_functions(dim)
        elif self.hash_family == 'e2lsh':
            self.hash_functions = self._generate_e2lsh_functions(dim)
        else:
            raise ValueError(f"Unsupported hash family: {self.hash_family}")
            
        # 初始化哈希表
        self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]
        
        # 对每个向量进行哈希并插入到表中
        for idx, vector in enumerate(tqdm(vectors, desc="Hashing vectors")):
            for table_idx in range(self.num_tables):
                if self.hash_family == 'random_projection':
                    hash_key = self._hash_vector_random_projection(vector, table_idx)
                elif self.hash_family == 'e2lsh':
                    hash_key = self._hash_vector_e2lsh(vector, table_idx)
                    
                # 限制桶大小（可选）
                if self.bucket_size is None or len(self.hash_tables[table_idx][hash_key]) < self.bucket_size:
                    self.hash_tables[table_idx][hash_key].append(idx)
        
        self.is_built = True
        build_time = time.time() - start_time
        
        # 打印统计信息
        total_buckets = sum(len(table) for table in self.hash_tables)
        avg_bucket_size = np.mean([len(bucket) for table in self.hash_tables for bucket in table.values()])
        
        print(f"Index built in {build_time:.4f} seconds")
        print(f"Total buckets: {total_buckets}")
        print(f"Average bucket size: {avg_bucket_size:.2f}")
        
        return build_time
        
    def _get_candidates(self, query_vector: np.ndarray) -> Set[int]:
        """获取候选向量索引"""
        candidates = set()
        
        for table_idx in range(self.num_tables):
            if self.hash_family == 'random_projection':
                hash_key = self._hash_vector_random_projection(query_vector, table_idx)
            elif self.hash_family == 'e2lsh':
                hash_key = self._hash_vector_e2lsh(query_vector, table_idx)
                
            candidates.update(self.hash_tables[table_idx].get(hash_key, []))
            
        return candidates
        
    def search(self, query_vectors: np.ndarray, k: int = 10, 
               metric: str = 'cosine') -> Tuple[np.ndarray, np.ndarray]:
        """执行搜索"""
        if not self.is_built:
            raise ValueError("Index not built. Call build_index() first.")
            
        print(f"Searching {len(query_vectors)} queries with k={k}...")
        start_time = time.time()
        
        all_distances = []
        all_indices = []
        candidate_counts = []
        
        for query_vector in tqdm(query_vectors, desc="Searching"):
            # 1. 获取候选集
            candidates = self._get_candidates(query_vector)
            candidate_counts.append(len(candidates))
            
            if len(candidates) == 0:
                # 如果没有候选，随机选择一些向量
                candidates = set(np.random.choice(len(self.vectors), min(k*2, len(self.vectors)), replace=False))
            
            # 2. 计算候选向量的精确距离
            candidate_vectors = self.vectors[list(candidates)]
            
            if metric == 'cosine':
                # 归一化
                query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
                candidate_norms = candidate_vectors / (np.linalg.norm(candidate_vectors, axis=1, keepdims=True) + 1e-8)
                similarities = candidate_norms @ query_norm
                distances = 1 - similarities
            else:  # euclidean
                distances = np.linalg.norm(candidate_vectors - query_vector, axis=1)
            
            # 3. 选择top-k
            if len(distances) >= k:
                top_k_idx = np.argpartition(distances, k)[:k]
                top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]
            else:
                top_k_idx = np.argsort(distances)
                
            candidate_list = list(candidates)
            result_indices = [candidate_list[i] for i in top_k_idx]
            result_distances = distances[top_k_idx]
            
            # 4. 如果结果不足k个，用随机向量补充
            while len(result_indices) < k:
                random_idx = np.random.randint(0, len(self.vectors))
                if random_idx not in result_indices:
                    result_indices.append(random_idx)
                    if metric == 'cosine':
                        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
                        vec_norm = self.vectors[random_idx] / (np.linalg.norm(self.vectors[random_idx]) + 1e-8)
                        sim = np.dot(query_norm, vec_norm)
                        dist = 1 - sim
                    else:
                        dist = np.linalg.norm(self.vectors[random_idx] - query_vector)
                    result_distances = np.append(result_distances, dist)
            
            all_distances.append(result_distances[:k])
            all_indices.append(result_indices[:k])
        
        search_time = time.time() - start_time
        avg_candidates = np.mean(candidate_counts)
        print(f"Search completed in {search_time:.4f} seconds")
        print(f"Average candidates per query: {avg_candidates:.1f}")
        
        return np.array(all_distances), np.array(all_indices)
        
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if not self.is_built:
            return {"status": "not_built"}
            
        total_buckets = sum(len(table) for table in self.hash_tables)
        bucket_sizes = [len(bucket) for table in self.hash_tables for bucket in table.values()]
        
        # 估算内存占用
        index_memory = 0
        for table in self.hash_tables:
            for bucket in table.values():
                index_memory += len(bucket) * 4  # 假设每个索引4字节
                
        vector_memory = self.vectors.nbytes
        
        return {
            "algorithm": "LSH",
            "hash_family": self.hash_family,
            "num_tables": self.num_tables,
            "hash_size": self.hash_size,
            "total_buckets": total_buckets,
            "avg_bucket_size": np.mean(bucket_sizes) if bucket_sizes else 0,
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "index_memory_mb": index_memory / (1024 * 1024),
            "vector_memory_mb": vector_memory / (1024 * 1024),
            "total_memory_mb": (index_memory + vector_memory) / (1024 * 1024)
        }


def evaluate_lsh(vectors_path: str, labels_path: str = None, 
                metadata_path: str = None, k_values: List[int] = [1, 10, 50],
                metric: str = 'cosine', test_size: int = 1000,
                num_tables: int = 10, hash_size: int = 10):
    """评估LSH算法"""
    print(f"\n=== LSH Search Evaluation ===")
    print(f"Metric: {metric}")
    print(f"Hash tables: {num_tables}, Hash size: {hash_size}")
    print(f"K values: {k_values}")
    
    # 1. 加载数据
    print("Loading data...")
    vectors = np.load(vectors_path)
    print(f"Loaded {len(vectors)} vectors of dimension {vectors.shape[1]}")
    
    # 2. 准备查询和gallery
    if test_size > len(vectors):
        test_size = len(vectors) // 10
    
    query_indices = np.random.choice(len(vectors), test_size, replace=False)
    query_vectors = vectors[query_indices]
    
    # 3. 选择哈希函数族
    if metric == 'cosine':
        hash_family = 'random_projection'
    else:
        hash_family = 'e2lsh'
    
    # 4. 构建索引
    lsh = LSHIndex(hash_family=hash_family, num_tables=num_tables, hash_size=hash_size)
    build_time = lsh.build_index(vectors)
    
    # 5. 执行搜索
    max_k = max(k_values)
    distances, indices = lsh.search(query_vectors, max_k, metric)
    
    # 6. 计算评估指标
    print("\n=== Results ===")
    print(f"Index build time: {build_time:.4f}s")
    
    # 计算平均查询时间
    start_time = time.time()
    _, _ = lsh.search(query_vectors[:100], max_k, metric)
    avg_query_time = (time.time() - start_time) / 100 * 1000
    qps = 1000 / avg_query_time if avg_query_time > 0 else float('inf')
    
    print(f"Average query time: {avg_query_time:.4f}ms")
    print(f"QPS: {qps:.2f}")
    
    # 内存占用
    stats = lsh.get_stats()
    print(f"Total memory usage: {stats['total_memory_mb']:.2f}MB")
    print(f"Index memory: {stats['index_memory_mb']:.2f}MB")
    print(f"Average bucket size: {stats['avg_bucket_size']:.2f}")
    
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
        'total_memory_mb': stats['total_memory_mb'],
        'stats': stats
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="LSH Search Evaluation")
    parser.add_argument('--vectors', required=True, help="Path to vectors .npy file")
    parser.add_argument('--labels', help="Path to labels .npy file (for FMNIST)")
    parser.add_argument('--metadata', help="Path to metadata .json file (for DeepFashion)")
    parser.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--k', type=int, nargs='+', default=[1, 10, 50])
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--num_tables', type=int, default=10, help="Number of hash tables")
    parser.add_argument('--hash_size', type=int, default=10, help="Hash size (bits)")
    
    args = parser.parse_args()
    
    evaluate_lsh(
        vectors_path=args.vectors,
        labels_path=args.labels,
        metadata_path=args.metadata,
        k_values=args.k,
        metric=args.metric,
        test_size=args.test_size,
        num_tables=args.num_tables,
        hash_size=args.hash_size
    )

