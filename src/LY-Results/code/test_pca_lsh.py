"""
测试PCA降维 + LSH优化
探索降维对LSH性能的影响
"""

import numpy as np
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))
from lsh_search import LSHIndex
from lsh_optimized import LSHOptimized
from brute_force_search import BruteForceSearch


def test_pca_lsh():
    """测试不同降维方案的LSH性能"""
    
    print("\n" + "="*70)
    print("PCA降维 + LSH优化测试")
    print("="*70)
    
    # 加载原始数据
    vectors_path = '../../data/data_LY/inshop_clip_vectors_gallery.npy'
    vectors = np.load(vectors_path)
    print(f"\n原始数据: {len(vectors)} 个向量，维度 {vectors.shape[1]}")
    
    # 准备测试查询
    test_size = 200
    query_indices = np.random.choice(len(vectors), test_size, replace=False)
    query_vectors = vectors[query_indices]
    
    # ========================================================================
    # 阶段1: 测试不同降维程度
    # ========================================================================
    print("\n" + "="*70)
    print("阶段1: 测试不同降维维度")
    print("="*70)
    
    dimensions = [2048, 1024, 512, 256, 128]  # 从原始到极端降维
    results_dim = []
    
    # Brute Force ground truth (只计算一次)
    print("\n--- Brute Force (Ground Truth) ---")
    bf = BruteForceSearch('cosine')
    bf.build_index(vectors)
    bf_dist, bf_ind = bf.search(query_vectors, k=50)
    
    for dim in dimensions:
        print(f"\n{'='*50}")
        print(f"测试维度: {dim}")
        print(f"{'='*50}")
        
        if dim == 2048:
            # 原始维度
            vectors_reduced = vectors
            query_reduced = query_vectors
            pca_time = 0
            variance_ratio = 1.0
        else:
            # 降维
            print(f"执行PCA降维: {vectors.shape[1]} → {dim}...")
            pca_start = time.time()
            pca = PCA(n_components=dim, random_state=42)
            vectors_reduced = pca.fit_transform(vectors)
            query_reduced = pca.transform(query_vectors)
            pca_time = time.time() - pca_start
            variance_ratio = pca.explained_variance_ratio_.sum()
            
            print(f"  降维时间: {pca_time:.2f}s")
            print(f"  保留方差: {variance_ratio*100:.2f}%")
        
        # 测试优化LSH v2配置（最佳配置）
        print(f"\n测试LSH (50表, 6位, 5探针)...")
        lsh = LSHOptimized(num_tables=50, hash_size=6, num_probes=5)
        
        build_start = time.time()
        lsh.build_index(vectors_reduced)
        build_time = time.time() - build_start
        
        search_start = time.time()
        lsh_dist, lsh_ind = lsh.search(query_reduced, k=50, min_candidates=200)
        search_time = time.time() - search_start
        avg_query_time = (search_time / len(query_vectors)) * 1000
        
        # 计算准确率（与原始空间的ground truth比较）
        total_correct = 0
        for i in range(len(query_vectors)):
            gt_set = set(bf_ind[i][:50])
            pred_set = set(lsh_ind[i][:50])
            total_correct += len(gt_set.intersection(pred_set))
        accuracy = (total_correct / (len(query_vectors) * 50)) * 100
        
        print(f"  构建时间: {build_time:.2f}s")
        print(f"  查询时间: {avg_query_time:.2f} ms")
        print(f"  准确率@50: {accuracy:.2f}%")
        
        results_dim.append({
            'dimension': dim,
            'pca_time': pca_time,
            'variance_retained': variance_ratio * 100,
            'build_time': build_time,
            'query_time_ms': avg_query_time,
            'accuracy_at_50': accuracy
        })
    
    # ========================================================================
    # 阶段2: 512维降维 + 参数优化
    # ========================================================================
    print("\n" + "="*70)
    print("阶段2: 512维降维 + 多种LSH配置测试")
    print("="*70)
    
    # PCA降维到512
    print("\n执行PCA降维: 2048 → 512...")
    pca = PCA(n_components=512, random_state=42)
    vectors_512 = pca.fit_transform(vectors)
    query_512 = pca.transform(query_vectors)
    print(f"保留方差: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    # 测试多种LSH配置
    configs = [
        {'name': '原始LSH (10表,12位)', 'num_tables': 10, 'hash_size': 12, 'probes': 0, 'use_optimized': False},
        {'name': '优化v1 (40表,8位,3探针)', 'num_tables': 40, 'hash_size': 8, 'probes': 3, 'use_optimized': True},
        {'name': '优化v2 (50表,6位,5探针)', 'num_tables': 50, 'hash_size': 6, 'probes': 5, 'use_optimized': True},
        {'name': '优化v3 (30表,8位,4探针)', 'num_tables': 30, 'hash_size': 8, 'probes': 4, 'use_optimized': True},
        {'name': '512维专用 (20表,10位,3探针)', 'num_tables': 20, 'hash_size': 10, 'probes': 3, 'use_optimized': True},
    ]
    
    results_512 = []
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        
        if config['use_optimized']:
            lsh = LSHOptimized(num_tables=config['num_tables'], 
                              hash_size=config['hash_size'],
                              num_probes=config['probes'])
        else:
            lsh = LSHIndex(hash_family='random_projection',
                          num_tables=config['num_tables'],
                          hash_size=config['hash_size'])
        
        lsh.build_index(vectors_512)
        
        start = time.time()
        if config['use_optimized']:
            lsh_dist, lsh_ind = lsh.search(query_512, k=50, min_candidates=200)
        else:
            lsh_dist, lsh_ind = lsh.search(query_512, k=50, metric='cosine')
        query_time = (time.time() - start) / len(query_512) * 1000
        
        # 计算准确率
        total_correct = 0
        for i in range(len(query_512)):
            gt_set = set(bf_ind[i][:50])
            pred_set = set(lsh_ind[i][:50])
            total_correct += len(gt_set.intersection(pred_set))
        accuracy = (total_correct / (len(query_512) * 50)) * 100
        
        print(f"  准确率@50: {accuracy:.2f}%")
        print(f"  查询时间: {query_time:.2f} ms")
        
        results_512.append({
            'config': config['name'],
            'accuracy': accuracy,
            'query_time': query_time
        })
    
    # ========================================================================
    # 对比：2048维 vs 512维
    # ========================================================================
    print("\n" + "="*70)
    print("最终对比: 2048维 vs 512维 (最佳配置)")
    print("="*70)
    
    # 2048维最佳结果（从之前的测试）
    print("\n2048维 + 优化LSH v2:")
    print(f"  准确率@50: 14.32%")
    print(f"  查询时间: 16.70 ms")
    
    # 512维最佳结果
    best_512 = max(results_512, key=lambda x: x['accuracy'])
    print(f"\n512维 + {best_512['config']}:")
    print(f"  准确率@50: {best_512['accuracy']:.2f}%")
    print(f"  查询时间: {best_512['query_time']:.2f} ms")
    
    improvement = best_512['accuracy'] - 14.32
    print(f"\n降维效果:")
    print(f"  准确率变化: {improvement:+.2f}%")
    print(f"  查询时间变化: {best_512['query_time'] - 16.70:+.2f} ms")
    
    # ========================================================================
    # 生成可视化报告
    # ========================================================================
    print("\n生成可视化报告...")
    
    df_dim = pd.DataFrame(results_dim)
    df_512 = pd.DataFrame(results_512)
    
    # 保存结果
    os.makedirs('results/optimization', exist_ok=True)
    df_dim.to_csv('results/optimization/pca_dimension_analysis.csv', index=False)
    df_512.to_csv('results/optimization/pca_512_config_analysis.csv', index=False)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 维度 vs 准确率
    axes[0, 0].plot(df_dim['dimension'], df_dim['accuracy_at_50'], 'o-', linewidth=2, markersize=10, color='#4ECDC4')
    axes[0, 0].set_xlabel('Dimension', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy@50 (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Dimension vs Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].invert_xaxis()
    axes[0, 0].grid(True, alpha=0.3)
    for i, row in df_dim.iterrows():
        axes[0, 0].annotate(f"{row['accuracy_at_50']:.1f}%", 
                           (row['dimension'], row['accuracy_at_50']),
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # 2. 维度 vs 查询时间
    axes[0, 1].plot(df_dim['dimension'], df_dim['query_time_ms'], 'o-', linewidth=2, markersize=10, color='#FF6B6B')
    axes[0, 1].set_xlabel('Dimension', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Query Time (ms)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Dimension vs Query Time', fontsize=12, fontweight='bold')
    axes[0, 1].invert_xaxis()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 维度 vs 方差保留
    ax_twin = axes[0, 2].twinx()
    line1 = axes[0, 2].plot(df_dim['dimension'], df_dim['accuracy_at_50'], 'o-', 
                            linewidth=2, markersize=10, color='#4ECDC4', label='Accuracy')
    line2 = ax_twin.plot(df_dim['dimension'][1:], df_dim['variance_retained'][1:], 's-', 
                         linewidth=2, markersize=10, color='#FFA07A', label='Variance Retained')
    axes[0, 2].set_xlabel('Dimension', fontsize=11, fontweight='bold')
    axes[0, 2].set_ylabel('Accuracy@50 (%)', fontsize=11, fontweight='bold', color='#4ECDC4')
    ax_twin.set_ylabel('Variance Retained (%)', fontsize=11, fontweight='bold', color='#FFA07A')
    axes[0, 2].set_title('Accuracy vs Variance Retained', fontsize=12, fontweight='bold')
    axes[0, 2].invert_xaxis()
    axes[0, 2].grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[0, 2].legend(lines, labels, loc='best')
    
    # 4. 512维不同配置准确率对比
    axes[1, 0].barh(range(len(df_512)), df_512['accuracy'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4'])
    axes[1, 0].set_yticks(range(len(df_512)))
    axes[1, 0].set_yticklabels([c.replace(' ', '\n') for c in df_512['config']], fontsize=8)
    axes[1, 0].set_xlabel('Accuracy@50 (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('512D Configs: Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    for i, row in df_512.iterrows():
        axes[1, 0].text(row['accuracy'], i, f" {row['accuracy']:.2f}%", 
                       va='center', fontsize=9, fontweight='bold')
    
    # 5. 512维不同配置查询时间对比
    axes[1, 1].barh(range(len(df_512)), df_512['query_time'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4'])
    axes[1, 1].set_yticks(range(len(df_512)))
    axes[1, 1].set_yticklabels([c.replace(' ', '\n') for c in df_512['config']], fontsize=8)
    axes[1, 1].set_xlabel('Query Time (ms)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('512D Configs: Query Time Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    # 6. 总结对比表
    axes[1, 2].axis('off')
    
    summary_text = f"""
PCA降维效果总结

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
最佳维度: 512
保留方差: {pca.explained_variance_ratio_.sum()*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2048维 (原始):
  准确率: 14.32%
  查询时间: 16.70 ms

512维 (降维):
  准确率: {best_512['accuracy']:.2f}%
  查询时间: {best_512['query_time']:.2f} ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
降维效果:
  准确率: {improvement:+.2f}%
  查询时间: {best_512['query_time']-16.70:+.2f} ms
  维度压缩: 75% ↓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
推荐方案:
{'✅ 推荐降维!' if improvement > 0 else '❌ 不推荐降维'}
理由: {'准确率提升且速度更快' if improvement > 0 and best_512['query_time'] < 16.70 
       else '准确率提升但速度稍慢' if improvement > 0 
       else '准确率下降'}
    """
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=11, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('PCA Dimensionality Reduction + LSH Optimization Analysis\nDeepFashion Dataset', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plt.savefig('results/optimization/pca_lsh_analysis.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 可视化已保存: results/optimization/pca_lsh_analysis.png")
    
    plt.show()
    
    # ========================================================================
    # 最终推荐
    # ========================================================================
    print("\n" + "="*70)
    print("🎯 最终推荐方案")
    print("="*70)
    
    if improvement > 0:
        print(f"\n✅ 推荐使用PCA降维!")
        print(f"\n最佳方案: PCA(512维) + {best_512['config']}")
        print(f"  • 准确率: {best_512['accuracy']:.2f}% (提升 {improvement:+.2f}%)")
        print(f"  • 查询时间: {best_512['query_time']:.2f} ms")
        print(f"  • 维度压缩: 2048 → 512 (75%压缩)")
        print(f"  • 方差保留: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        
        print(f"\n优势:")
        print(f"  1. 准确率更高")
        if best_512['query_time'] < 16.70:
            print(f"  2. 查询速度更快")
        print(f"  3. 内存占用更小")
        print(f"  4. 索引构建更快")
    else:
        print(f"\n❌ 不推荐降维")
        print(f"\n原因: 准确率下降了 {abs(improvement):.2f}%")
        print(f"\n建议继续使用: 2048维 + 优化LSH v2")
    
    return results_dim, results_512, improvement


if __name__ == '__main__':
    results_dim, results_512, improvement = test_pca_lsh()

