import numpy as np
import faiss
import time
import argparse
import pandas as pd
from collections import defaultdict

def evaluate_inshop_retrieval(k_values):
    """使用 In-shop 数据集的特征文件评估检索性能。"""

    # --- 1. 路径配置 ---
    # 我们只使用 test/gallery 数据集，并从中划分 query 和 gallery
    VECTORS_PATH = 'data/inshop_clip_vectors_gallery.npy'
    IDS_PATH = 'data/inshop_clip_ids_gallery.json'

    # --- 2. 加载数据 ---
    print("Loading data files...")
    try:
        vectors = np.load(VECTORS_PATH).astype('float32')
        df_items = pd.read_json(IDS_PATH, lines=True)
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        return

    # --- 3. 划分 Query 和 Gallery ---
    print("Splitting test set into query and gallery...")
    
    query_indices = []
    query_item_ids = []
    gallery_indices = list(range(len(df_items)))
    gallery_item_ids = df_items['item_id'].tolist()

    # 对于每个 item_id，选择第一张图片作为 query
    for item_id, group in df_items.groupby('item_id'):
        query_idx = group.index[0]
        query_indices.append(query_idx)
        query_item_ids.append(item_id)

    query_vectors = vectors[query_indices]
    gallery_vectors = vectors # Gallery 是整个 test 集

    print(f"Found {len(query_vectors)} queries and {len(gallery_vectors)} gallery items.")

    # --- 4. 构建真实的 Ground Truth ---
    print("Building ground truth...")
    ground_truth = defaultdict(list)
    for idx, item_id in enumerate(gallery_item_ids):
        ground_truth[item_id].append(idx)

    # --- 5. 构建Faiss索引 ---
    print("Building Faiss index for gallery vectors...")
    dim = gallery_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(gallery_vectors)

    # --- 6. 执行搜索 ---
    max_k = max(k_values)
    # We search for k+1 because the first result will be the query itself
    print(f"Running retrieval for {len(query_vectors)} queries (k={max_k})...")
    start_time = time.time()
    # Search for k+1 neighbors, as the first one is the query itself
    distances, neighbor_indices = index.search(query_vectors, max_k + 1)
    end_time = time.time()

    # --- 7. 计算Recall@k ---
    print("Calculating Recall@k...")
    recall_at_k = {}
    
    num_queries_with_gt = 0
    # First, count how many queries have at least one ground truth image
    for i in range(len(query_item_ids)):
        query_item_id = query_item_ids[i]
        query_idx = query_indices[i]
        gt_indices = {idx for idx in ground_truth[query_item_id] if idx != query_idx}
        if gt_indices:
            num_queries_with_gt += 1

    for k in k_values:
        correct_matches = 0
        for i in range(len(query_item_ids)):
            query_item_id = query_item_ids[i]
            query_idx = query_indices[i]
            
            # Ground truth are all images with the same item_id, excluding the query itself.
            gt_indices = {idx for idx in ground_truth[query_item_id] if idx != query_idx}

            # Skip queries that have no other images of the same item in the gallery
            if not gt_indices:
                continue

            # Get top k results, skipping the first one (the query itself)
            top_k_indices = neighbor_indices[i, 1:k+1]
            
            # Check if any of the ground truth indices are in the top k
            if any(retrieved_idx in gt_indices for retrieved_idx in top_k_indices):
                correct_matches += 1
        
        if num_queries_with_gt > 0:
            recall = (correct_matches / num_queries_with_gt) * 100
            recall_at_k[k] = recall
        else:
            recall_at_k[k] = 0.0

    # --- 8. 打印报告 ---
    total_time = end_time - start_time
    avg_time_per_query = (total_time / len(query_vectors)) * 1000 if len(query_vectors) > 0 else 0
    qps = 1 / (avg_time_per_query / 1000) if avg_time_per_query > 0 else float('inf')

    print("\n--- In-shop Retrieval Evaluation Report ---")
    print(f"Total queries evaluated: {num_queries_with_gt}")
    print(f"Total gallery size: {len(gallery_vectors)}")
    print(f"Total search time: {total_time:.4f} seconds")
    print(f"Avg. time/query: {avg_time_per_query:.4f} ms")
    print(f"Queries Per Second (QPS): {qps:.2f}")
    print("-------------------------------------")
    for k, recall in recall_at_k.items():
        print(f"Recall@{k}: {recall:.2f}%")
    print("-------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Evaluate In-shop Retrieval Performance.")
    parser.add_argument("--k", type=int, nargs='+', default=[1, 10, 50], help="Values of k for Recall@k calculation.")
    args = parser.parse_args()
    evaluate_inshop_retrieval(args.k)

if __name__ == '__main__':
    main()