# src/evaluator.py (卓越版)
import numpy as np
import faiss
import time
import os
import gc
import argparse

def build_index(index_type, vectors):
    """根据指定的类型构建Faiss索引"""
    dim = vectors.shape[1]
    if index_type == "IndexFlatL2":
        print(f"Building {index_type}...")
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        return index
    # --- 未来扩展点 ---
    # elif index_type == "IndexIVFFlat":
    #     # 这里将是我们实现IVF索引的地方
    #     print("Building IndexIVFFlat... (Not implemented yet)")
    #     return None 
    else:
        raise NotImplementedError(f"Index type '{index_type}' is not supported.")

def main(args):
    """Main function to run Faiss evaluation."""
    # --- 1. Load Data ---
    print("Loading feature vectors and labels...")
    if not (os.path.exists(args.vectors_path) and os.path.exists(args.labels_path)):
        print(f"Error: Data files not found. Please check paths.")
        return

    feature_vectors = np.load(args.vectors_path).astype('float32')
    labels = np.load(args.labels_path)
    
    # --- 2. Build Faiss Index ---
    index = build_index(args.index_type, feature_vectors)
    if index is None:
        return
        
    # --- 3. Run Evaluation ---
    num_queries = args.num_queries
    k = args.k
    
    print(f"\nRunning evaluation on '{args.index_type}' with {num_queries} queries (k={k})...")
    
    query_indices = np.random.choice(index.ntotal, num_queries, replace=False)
    query_vectors = feature_vectors[query_indices]
    query_labels = labels[query_indices]

    start_time = time.time()
    distances, neighbor_indices = index.search(query_vectors, k)
    end_time = time.time()

    # --- 4. Calculate Precision@K & Report ---
    total_time = end_time - start_time
    avg_time_per_query = (total_time / num_queries) * 1000 # ms
    qps = 1 / (avg_time_per_query / 1000) if avg_time_per_query > 0 else float('inf')

    # [FIXED] Correctly calculate Average Precision@K
    total_correct_items = 0
    # 我们评估的是返回的 k 个结果
    total_retrieved_items = num_queries * k 

    for i in range(num_queries):
        query_label = query_labels[i]
        
        # 暴力搜索返回的第一个结果是查询本身，我们需要排除
        # 但对于ANN，返回的第一个结果不一定是查询本身，所以需要更严谨的处理
        retrieved_indices = neighbor_indices[i]
        # 找到查询本身在返回结果中的位置 (如果有)
        self_match_mask = retrieved_indices == query_indices[i]
        # 排除查询本身
        retrieved_indices = retrieved_indices[~self_match_mask]
        
        retrieved_labels = labels[retrieved_indices]
        
        correct_count = np.sum(retrieved_labels == query_label)
        total_correct_items += correct_count
            
    avg_precision_at_k = (total_correct_items / (num_queries * (k-1))) * 100 # 除以 k-1 个评估对象

    print("\n--- Evaluation Report ---")
    print(f"Index Type: {args.index_type}")
    print(f"Vectors Path: {args.vectors_path}")
    print(f"Total search time: {total_time:.4f} seconds")
    print(f"Avg. time/query: {avg_time_per_query:.4f} ms")
    print(f"Queries Per Second (QPS): {qps:.2f}")
    print(f"Average Precision@{k-1}: {avg_precision_at_k:.2f}%")
    print("-------------------------")

    del feature_vectors, labels, index
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Faiss index performance.")
    parser.add_argument("--vectors_path", type=str, required=True, help="Path to the .npy file containing feature vectors.")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to the .npy file containing labels.")
    parser.add_argument("--index_type", type=str, default="IndexFlatL2", help="Type of Faiss index to evaluate.")
    parser.add_argument("--num_queries", type=int, default=1000, help="Number of queries to run.")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors to retrieve.")
    
    args = parser.parse_args()
    main(args)