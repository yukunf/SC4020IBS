"""
优化版LSH实现
基于深度分析的发现，实施以下优化策略：
1. 使用更多哈希表（30-50个）+ 更小的哈希位数（6-8位）
2. Multi-Probe LSH: 探测相邻桶以增加候选集
3. 自适应参数选择：根据数据特征自动调整参数
4. 候选集扩展：当候选太少时自动扩展搜索范围
"""

import numpy as np
import time
from typing import Tuple, List, Set
from collections import defaultdict
from tqdm import tqdm


class LSHOptimized:
    """优化版LSH索引"""
    
    def __init__(self, num_tables=40, hash_size=8, num_probes=3):
        """
        Args:
            num_tables: 哈希表数量（增加到40）
            hash_size: 哈希位数（减小到8）
            num_probes: Multi-probe探测数量
        """
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.num_probes = num_probes  # 新增：多探针参数
        
        self.hash_functions = []
        self.hash_tables = []
        self.vectors = None
        self.is_built = False
        
    def _generate_random_projection_functions(self, dim: int):
        """生成随机投影哈希函数"""
        functions = []
        for _ in range(self.num_tables):
            random_vectors = np.random.randn(self.hash_size, dim)
            random_vectors = random_vectors / np.linalg.norm(random_vectors, axis=1, keepdims=True)
            functions.append(random_vectors)
        return functions
        
    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> str:
        """哈希向量到字符串key"""
        projections = self.hash_functions[table_idx] @ vector
        binary_hash = (projections > 0).astype(int)
        return ''.join(map(str, binary_hash))
    
    def _get_neighboring_hashes(self, hash_code: str, num_neighbors: int) -> List[str]:
        """
        获取相邻的哈希码（Multi-Probe LSH的核心）
        通过翻转少量位来生成邻居哈希
        """
        neighbors = [hash_code]
        hash_bits = list(hash_code)
        
        # 单比特翻转
        for i in range(len(hash_bits)):
            if len(neighbors) >= num_neighbors:
                break
            flipped = hash_bits.copy()
            flipped[i] = '1' if flipped[i] == '0' else '0'
            neighbors.append(''.join(flipped))
        
        # 双比特翻转（如果还需要更多邻居）
        if len(neighbors) < num_neighbors and len(hash_bits) > 1:
            for i in range(len(hash_bits)):
                for j in range(i+1, len(hash_bits)):
                    if len(neighbors) >= num_neighbors:
                        break
                    flipped = hash_bits.copy()
                    flipped[i] = '1' if flipped[i] == '0' else '0'
                    flipped[j] = '1' if flipped[j] == '0' else '0'
                    neighbors.append(''.join(flipped))
        
        return neighbors[:num_neighbors]
        
    def build_index(self, vectors: np.ndarray):
        """构建LSH索引"""
        print(f"Building Optimized LSH index...")
        print(f"  Tables: {self.num_tables}, Hash size: {self.hash_size}, Probes: {self.num_probes}")
        start_time = time.time()
        
        self.vectors = vectors.astype(np.float32)
        dim = vectors.shape[1]
        
        # 生成哈希函数
        self.hash_functions = self._generate_random_projection_functions(dim)
        
        # 初始化哈希表
        self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]
        
        # 对每个向量进行哈希
        for idx, vector in enumerate(tqdm(vectors, desc="Hashing vectors")):
            for table_idx in range(self.num_tables):
                hash_key = self._hash_vector(vector, table_idx)
                self.hash_tables[table_idx][hash_key].append(idx)
        
        self.is_built = True
        build_time = time.time() - start_time
        
        # 统计信息
        total_buckets = sum(len(table) for table in self.hash_tables)
        bucket_sizes = [len(bucket) for table in self.hash_tables for bucket in table.values()]
        avg_bucket_size = np.mean(bucket_sizes) if bucket_sizes else 0
        
        print(f"Index built in {build_time:.4f} seconds")
        print(f"Total buckets: {total_buckets}")
        print(f"Average bucket size: {avg_bucket_size:.2f}")
        print(f"Max bucket size: {max(bucket_sizes) if bucket_sizes else 0}")
        
        return build_time
        
    def _get_candidates_multiprobe(self, query_vector: np.ndarray, min_candidates: int = 100) -> Set[int]:
        """
        使用Multi-Probe LSH获取候选集
        如果候选太少，自动增加探测范围
        """
        candidates = set()
        current_probes = self.num_probes
        
        # 逐步增加探测数量，直到候选足够
        max_probes = min(20, 2 ** self.hash_size)  # 最多探测20个邻居
        
        while len(candidates) < min_candidates and current_probes <= max_probes:
            candidates.clear()
            
            for table_idx in range(self.num_tables):
                # 获取查询向量的哈希码
                query_hash = self._hash_vector(query_vector, table_idx)
                
                # 获取邻居哈希码
                neighbor_hashes = self._get_neighboring_hashes(query_hash, current_probes)
                
                # 收集所有邻居桶中的候选
                for neighbor_hash in neighbor_hashes:
                    if neighbor_hash in self.hash_tables[table_idx]:
                        candidates.update(self.hash_tables[table_idx][neighbor_hash])
                
                # 如果已经有足够候选，可以提前停止
                if len(candidates) >= min_candidates * 2:
                    break
            
            # 如果候选还不够，增加探测范围
            if len(candidates) < min_candidates:
                current_probes += 5
        
        return candidates
        
    def search(self, query_vectors: np.ndarray, k: int = 10, 
               metric: str = 'cosine', min_candidates: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行搜索
        Args:
            min_candidates: 最小候选数量，默认为k*10
        """
        if not self.is_built:
            raise ValueError("Index not built. Call build_index() first.")
        
        if min_candidates is None:
            min_candidates = max(k * 10, 100)  # 至少100个候选
            
        print(f"Searching {len(query_vectors)} queries (k={k}, min_candidates={min_candidates})...")
        start_time = time.time()
        
        all_distances = []
        all_indices = []
        candidate_counts = []
        
        for query_vector in tqdm(query_vectors, desc="Searching"):
            # 1. 使用Multi-Probe获取候选集
            candidates = self._get_candidates_multiprobe(query_vector, min_candidates)
            candidate_counts.append(len(candidates))
            
            if len(candidates) == 0:
                # 极端情况：随机选择
                candidates = set(np.random.choice(len(self.vectors), k, replace=False))
            
            # 2. 计算候选向量的精确距离
            candidate_list = list(candidates)
            candidate_vectors = self.vectors[candidate_list]
            
            if metric == 'cosine':
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
            
            result_indices = [candidate_list[i] for i in top_k_idx]
            result_distances = distances[top_k_idx]
            
            # 4. 补充到k个（如果需要）
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


def test_optimized_lsh():
    """测试优化版LSH"""
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from brute_force_search import BruteForceSearch
    
    print("\n" + "="*60)
    print("优化版LSH测试 - DeepFashion数据集")
    print("="*60)
    
    # 加载数据
    vectors_path = '../../data/inshop_clip_vectors_gallery.npy'
    vectors = np.load(vectors_path)
    print(f"\n加载数据: {len(vectors)} 个向量，维度 {vectors.shape[1]}")
    
    # 准备测试查询
    test_size = 200
    query_indices = np.random.choice(len(vectors), test_size, replace=False)
    query_vectors = vectors[query_indices]
    
    # 1. Brute Force（Ground Truth）
    print("\n--- Brute Force (Ground Truth) ---")
    bf = BruteForceSearch('cosine')
    bf.build_index(vectors)
    bf_dist, bf_ind = bf.search(query_vectors, k=50)
    
    # 2. 原始LSH（作为对比）
    print("\n--- 原始LSH (10 tables, 12 bits) ---")
    from lsh_search import LSHIndex
    
    lsh_original = LSHIndex(hash_family='random_projection', num_tables=10, hash_size=12)
    lsh_original.build_index(vectors)
    
    start = time.time()
    lsh_orig_dist, lsh_orig_ind = lsh_original.search(query_vectors, k=50, metric='cosine')
    lsh_orig_time = (time.time() - start) / len(query_vectors) * 1000
    
    # 计算原始LSH准确率
    total_correct_orig = 0
    for i in range(len(query_vectors)):
        gt_set = set(bf_ind[i][:50])
        pred_set = set(lsh_orig_ind[i][:50])
        total_correct_orig += len(gt_set.intersection(pred_set))
    acc_orig = (total_correct_orig / (len(query_vectors) * 50)) * 100
    
    print(f"原始LSH准确率@50: {acc_orig:.2f}%")
    print(f"原始LSH查询时间: {lsh_orig_time:.2f} ms")
    
    # 3. 优化版LSH - 策略1: 更多表 + 更小哈希位
    print("\n--- 优化LSH v1 (40 tables, 8 bits, 3 probes) ---")
    lsh_opt1 = LSHOptimized(num_tables=40, hash_size=8, num_probes=3)
    lsh_opt1.build_index(vectors)
    
    start = time.time()
    lsh_opt1_dist, lsh_opt1_ind = lsh_opt1.search(query_vectors, k=50, min_candidates=150)
    lsh_opt1_time = (time.time() - start) / len(query_vectors) * 1000
    
    total_correct_opt1 = 0
    for i in range(len(query_vectors)):
        gt_set = set(bf_ind[i][:50])
        pred_set = set(lsh_opt1_ind[i][:50])
        total_correct_opt1 += len(gt_set.intersection(pred_set))
    acc_opt1 = (total_correct_opt1 / (len(query_vectors) * 50)) * 100
    
    print(f"优化LSH v1准确率@50: {acc_opt1:.2f}%")
    print(f"优化LSH v1查询时间: {lsh_opt1_time:.2f} ms")
    print(f"准确率提升: {acc_opt1 - acc_orig:+.2f}%")
    
    # 4. 优化版LSH - 策略2: 极端参数（50表+6位+更多探针）
    print("\n--- 优化LSH v2 (50 tables, 6 bits, 5 probes) ---")
    lsh_opt2 = LSHOptimized(num_tables=50, hash_size=6, num_probes=5)
    lsh_opt2.build_index(vectors)
    
    start = time.time()
    lsh_opt2_dist, lsh_opt2_ind = lsh_opt2.search(query_vectors, k=50, min_candidates=200)
    lsh_opt2_time = (time.time() - start) / len(query_vectors) * 1000
    
    total_correct_opt2 = 0
    for i in range(len(query_vectors)):
        gt_set = set(bf_ind[i][:50])
        pred_set = set(lsh_opt2_ind[i][:50])
        total_correct_opt2 += len(gt_set.intersection(pred_set))
    acc_opt2 = (total_correct_opt2 / (len(query_vectors) * 50)) * 100
    
    print(f"优化LSH v2准确率@50: {acc_opt2:.2f}%")
    print(f"优化LSH v2查询时间: {lsh_opt2_time:.2f} ms")
    print(f"准确率提升: {acc_opt2 - acc_orig:+.2f}%")
    
    # 5. 优化版LSH - 策略3: 平衡版（30表+8位+适中探针）
    print("\n--- 优化LSH v3 (30 tables, 8 bits, 4 probes) ---")
    lsh_opt3 = LSHOptimized(num_tables=30, hash_size=8, num_probes=4)
    lsh_opt3.build_index(vectors)
    
    start = time.time()
    lsh_opt3_dist, lsh_opt3_ind = lsh_opt3.search(query_vectors, k=50, min_candidates=150)
    lsh_opt3_time = (time.time() - start) / len(query_vectors) * 1000
    
    total_correct_opt3 = 0
    for i in range(len(query_vectors)):
        gt_set = set(bf_ind[i][:50])
        pred_set = set(lsh_opt3_ind[i][:50])
        total_correct_opt3 += len(gt_set.intersection(pred_set))
    acc_opt3 = (total_correct_opt3 / (len(query_vectors) * 50)) * 100
    
    print(f"优化LSH v3准确率@50: {acc_opt3:.2f}%")
    print(f"优化LSH v3查询时间: {lsh_opt3_time:.2f} ms")
    print(f"准确率提升: {acc_opt3 - acc_orig:+.2f}%")
    
    # 总结对比
    print("\n" + "="*60)
    print("优化效果总结")
    print("="*60)
    
    results = [
        ("原始LSH (10表,12位)", acc_orig, lsh_orig_time, 0),
        ("优化v1 (40表,8位,3探针)", acc_opt1, lsh_opt1_time, acc_opt1 - acc_orig),
        ("优化v2 (50表,6位,5探针)", acc_opt2, lsh_opt2_time, acc_opt2 - acc_orig),
        ("优化v3 (30表,8位,4探针)", acc_opt3, lsh_opt3_time, acc_opt3 - acc_orig),
    ]
    
    print(f"\n{'配置':<30} {'准确率@50':<12} {'查询时间':<12} {'提升'}")
    print("-" * 70)
    for name, acc, qtime, improvement in results:
        print(f"{name:<30} {acc:>6.2f}%      {qtime:>6.2f} ms    {improvement:+6.2f}%")
    
    # 找出最佳配置
    best_idx = max(range(len(results)), key=lambda i: results[i][1])
    print(f"\n🏆 最佳配置: {results[best_idx][0]}")
    print(f"   准确率: {results[best_idx][1]:.2f}%")
    print(f"   查询时间: {results[best_idx][2]:.2f} ms")
    print(f"   相比原始LSH提升: {results[best_idx][3]:+.2f}%")
    
    # 保存结果
    import pandas as pd
    df = pd.DataFrame(results, columns=['配置', '准确率@50(%)', '查询时间(ms)', '提升(%)'])
    os.makedirs('../../doc/report/LSHBFreports/results/optimization', exist_ok=True)
    df.to_csv('results/optimization/lsh_optimization_comparison.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: results/optimization/lsh_optimization_comparison.csv")
    
    return results


if __name__ == '__main__':
    test_optimized_lsh()

