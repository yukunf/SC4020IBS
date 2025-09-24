# src/validator.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from tqdm import tqdm
import os
import argparse

def cosine_similarity(vec_a, vec_b):
    """Calculates cosine similarity between two vectors."""
    # Add a small epsilon for numerical stability
    epsilon = 1e-8
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return np.dot(vec_a, vec_b) / ((norm_a * norm_b) + epsilon)

def run_validation(args):
    """
    Loads feature vectors and labels, then generates and saves validation charts.
    """
    print("--- Starting Validation Process ---")
    
    # --- 1. Load Data ---
    print(f"Loading data from: {args.vectors_path}")
    if not os.path.exists(args.vectors_path) or not os.path.exists(args.labels_path):
        print("Error: Data files not found. Please run the embedder script first.")
        return
        
    all_vectors = np.load(args.vectors_path).astype('float32')
    all_labels = np.load(args.labels_path)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print("Data loaded successfully.")

    # --- 2. Calculate Global Similarity Matrix ---
    print("\nCalculating global similarity matrix...")
    samples_per_class = args.samples_per_class
    similarity_matrix = np.zeros((10, 10))
    
    sampled_vectors_by_class = []
    for i in range(10):
        indices = np.where(all_labels == i)[0]
        # Ensure we don't sample more than available
        num_to_sample = min(samples_per_class, len(indices))
        sampled_indices = np.random.choice(indices, num_to_sample, replace=False)
        sampled_vectors_by_class.append(all_vectors[sampled_indices])

    pbar = tqdm(list(combinations_with_replacement(range(10), 2)), desc="Calculating similarity matrix")
    for i, j in pbar:
        vectors_i = sampled_vectors_by_class[i]
        vectors_j = sampled_vectors_by_class[j]
        
        if i == j: # Intra-class similarity
            # Approximate by calculating mean similarity to the class centroid
            centroid_i = np.mean(vectors_i, axis=0)
            sims = [cosine_similarity(vec, centroid_i) for vec in vectors_i]
            avg_sim = np.mean(sims)
        else: # Inter-class similarity
            # Approximate by calculating the similarity between class centroids
            centroid_i = np.mean(vectors_i, axis=0)
            centroid_j = np.mean(vectors_j, axis=0)
            avg_sim = cosine_similarity(centroid_i, centroid_j)
            
        similarity_matrix[i, j] = similarity_matrix[j, i] = avg_sim

    print("Similarity matrix calculation complete.")

    # --- 3. Visualize and Save Heatmap ---
    df_heatmap = pd.DataFrame(similarity_matrix, index=class_names, columns=class_names)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_heatmap, 
                annot=True,     
                fmt=".3f",      
                cmap="viridis", 
                linewidths=.5)
    plt.title('Global Semantic Similarity Matrix Heatmap', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    os.makedirs(args.output_dir, exist_ok=True)
    # Generate a dynamic filename based on the input vector file
    vector_filename = os.path.basename(args.vectors_path).replace('_vectors.npy', '')
    heatmap_path = os.path.join(args.output_dir, f"similarity_heatmap_{vector_filename}.png")
    
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()

    print(f"\nValidation heatmap saved to: {heatmap_path}")
    print("\n--- Similarity Matrix (rounded to 3 decimal places) ---")
    print(df_heatmap.round(3))
    print("\n--- Validation Process Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate embedding quality by generating a similarity heatmap.")
    parser.add_argument("--vectors_path", type=str, required=True, help="Path to the .npy file with feature vectors.")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to the .npy file with corresponding labels.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the output heatmap image.")
    parser.add_argument("--samples_per_class", type=int, default=200, help="Number of samples per class to use for calculation.")
    
    args = parser.parse_args()
    run_validation(args)
    # --- 诊断代码 ---
print("--- Running Vector Normality Diagnosis ---")
vectors_to_check = np.load('./data/fmnist_clip-vit-base-patch32_vectors.npy')

# 计算每个向量的L2范数 (长度)
norms = np.linalg.norm(vectors_to_check, axis=1)

# 打印统计摘要
print("Statistics of vector L2 norms:")
print(f"  Mean: {np.mean(norms):.4f}")
print(f"  Std Dev: {np.std(norms):.4f}")
print(f"  Min: {np.min(norms):.4f}")
print(f"  Max: {np.max(norms):.4f}")

# 理想情况下，如果向量被归一化了，所有这些值都应该非常接近 1.0
if np.allclose(np.mean(norms), 1.0, atol=1e-5) and np.std(norms) < 1e-5:
    print("\n[Diagnosis Result]: Vectors appear to be correctly L2-normalized.")
else:
    print("\n[Diagnosis Result]: WARNING! Vectors are NOT L2-normalized. This is likely the cause of the issue.")