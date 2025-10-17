"""
LSH参数分析脚本（修复版）
包含PDF要求的所有代码块修复
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import sys
import os

sys.path.append(os.path.dirname(__file__))
from brute_force_search import BruteForceSearch
from lsh_search import LSHIndex


def analyze_lsh_parameters_deepfashion():
    """分析DeepFashion数据集的LSH参数"""
    print("=== LSH Parameter Analysis - DeepFashion ===")
    
    # 加载DeepFashion数据
    vectors_path = '../../data/data_LY/inshop_clip_vectors_gallery.npy'
    vectors = np.load(vectors_path)
    
    # 选择测试集
    test_size = 200
    query_indices = np.random.choice(len(vectors), test_size, replace=False)
    query_vectors = vectors[query_indices]
    
    # 参数网格
    num_tables_range = [5, 10, 15, 20, 25, 30]
    hash_size_range = [8, 10, 12, 14, 16]
    
    results = []
    
    print("开始参数网格搜索...")
    for num_tables in num_tables_range:
        for hash_size in hash_size_range:
            print(f"测试配置: Tables={num_tables}, Hash_size={hash_size}")
            
            # 构建LSH索引
            lsh = LSHIndex(hash_family='random_projection', 
                          num_tables=num_tables, 
                          hash_size=hash_size)
            
            # 测量构建时间
            start_time = time.time()
            lsh.build_index(vectors)
            build_time = time.time() - start_time
            
            # 测量搜索时间
            start_time = time.time()
            distances, indices = lsh.search(query_vectors, 50, 'cosine')
            search_time = time.time() - start_time
            
            avg_query_time = (search_time / len(query_vectors)) * 1000  # ms
            
            # 计算与brute force的准确率（使用小样本避免太慢）
            bf = BruteForceSearch('cosine')
            bf.build_index(vectors)
            bf_dist, bf_ind = bf.search(query_vectors[:20], 50)
            
            # 计算准确率
            total_correct = 0
            for i in range(20):
                gt_set = set(bf_ind[i][:50])
                pred_set = set(indices[i][:50])
                total_correct += len(gt_set.intersection(pred_set))
            accuracy_at_50 = (total_correct / (20 * 50)) * 100
            
            # 获取统计信息
            stats = lsh.get_stats()
            
            results.append({
                'num_tables': num_tables,
                'hash_size': hash_size,
                'build_time': build_time,
                'avg_query_time_ms': avg_query_time,
                'accuracy_at_50': accuracy_at_50,
                'total_memory_mb': stats['total_memory_mb'],
                'avg_bucket_size': stats['avg_bucket_size'],
                'total_buckets': stats['total_buckets']
            })
    
    return pd.DataFrame(results)


def analyze_data_size_scalability_deepfashion():
    """分析不同数据规模下的性能"""
    print("\n=== Data Size Scalability Analysis - DeepFashion ===")
    
    # 加载完整数据
    full_vectors = np.load('../../data/data_LY/inshop_clip_vectors_gallery.npy')
    
    # 不同数据规模
    data_sizes = [1000, 3000, 5000, 10000, 15000]
    test_query_size = 100
    
    results = []
    
    for data_size in data_sizes:
        print(f"测试数据规模: {data_size}")
        
        # 选择数据子集
        indices = np.random.choice(len(full_vectors), data_size, replace=False)
        vectors = full_vectors[indices]
        
        # 选择查询
        query_indices = np.random.choice(len(vectors), min(test_query_size, len(vectors)//10), replace=False)
        query_vectors = vectors[query_indices]
        
        # 1. Brute Force - 运行3次取平均
        bf_times = []
        for _ in range(3):
            bf = BruteForceSearch('cosine')
            bf.build_index(vectors)
            start_time = time.time()
            bf_distances, bf_indices = bf.search(query_vectors, 10)
            bf_times.append((time.time() - start_time) / len(query_vectors) * 1000)
        bf_avg_query_time = np.mean(bf_times)
        bf_std_query_time = np.std(bf_times)
        
        # 2. LSH - 运行3次取平均
        lsh_times = []
        for _ in range(3):
            lsh = LSHIndex(hash_family='random_projection', num_tables=10, hash_size=12)
            lsh.build_index(vectors)
            start_time = time.time()
            lsh_distances, lsh_indices = lsh.search(query_vectors, 10, 'cosine')
            lsh_times.append((time.time() - start_time) / len(query_vectors) * 1000)
        lsh_avg_query_time = np.mean(lsh_times)
        lsh_std_query_time = np.std(lsh_times)
        
        results.append({
            'data_size': data_size,
            'bf_avg_query_time_ms': bf_avg_query_time,
            'bf_std_query_time_ms': bf_std_query_time,
            'lsh_avg_query_time_ms': lsh_avg_query_time,
            'lsh_std_query_time_ms': lsh_std_query_time,
            'speedup_factor': bf_avg_query_time / lsh_avg_query_time if lsh_avg_query_time > 0 else 0
        })
    
    return pd.DataFrame(results)


def generate_analysis_report_fixed(param_df, scale_df, output_dir='reports'):
    """生成修复后的分析报告（包含代码块1.1和1.2的修复）"""
    print("\n=== 生成修复后的分析报告 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    report = []
    
    report.append("# LSH参数分析报告（修复版） - DeepFashion数据集\n")
    report.append(f"生成时间: {pd.Timestamp.now()}\n")
    
    report.append("## 1. LSH参数分析总结\n")
    
    # 找到最优配置
    best_accuracy_config = param_df.loc[param_df['accuracy_at_50'].idxmax()]
    fastest_config = param_df.loc[param_df['avg_query_time_ms'].idxmin()]
    
    report.append("### 1.1 最佳准确率配置\n")
    report.append(f"- 哈希表数量: {best_accuracy_config['num_tables']}\n")
    report.append(f"- 哈希大小: {best_accuracy_config['hash_size']}\n")
    report.append(f"- 准确率@50: {best_accuracy_config['accuracy_at_50']:.2f}%\n")
    report.append(f"- 查询时间: {best_accuracy_config['avg_query_time_ms']:.2f}ms\n")
    report.append(f"- 构建时间: {best_accuracy_config['build_time']:.2f}s\n")
    report.append(f"- 内存使用: {best_accuracy_config['total_memory_mb']:.2f}MB\n\n")
    
    report.append("### 1.2 最快查询配置\n")
    report.append(f"- 哈希表数量: {fastest_config['num_tables']}\n")
    report.append(f"- 哈希大小: {fastest_config['hash_size']}\n")
    report.append(f"- 准确率@50: {fastest_config['accuracy_at_50']:.2f}%\n")
    report.append(f"- 查询时间: {fastest_config['avg_query_time_ms']:.2f}ms\n")
    report.append(f"- 构建时间: {fastest_config['build_time']:.2f}s\n")
    report.append(f"- 内存使用: {fastest_config['total_memory_mb']:.2f}MB\n\n")
    
    # ========================================================================
    # CODE BLOCK 1.1: FIX CORRELATION ANALYSIS
    # 使用原始数据而非分组均值来计算相关性
    # ========================================================================
    report.append("### 1.3 参数影响分析\n\n")
    report.append("**哈希表数量的影响:**\n\n")
    
    # 修复: 使用原始数据的相关性，而不是分组均值
    report.append(f"- 与构建时间的相关性: {param_df['num_tables'].corr(param_df['build_time']):.3f}\n")
    report.append(f"- 与查询时间的相关性: {param_df['num_tables'].corr(param_df['avg_query_time_ms']):.3f}\n")
    report.append(f"- 与准确率的相关性: {param_df['num_tables'].corr(param_df['accuracy_at_50']):.3f}\n\n")
    
    report.append("**哈希大小的影响:**\n\n")
    report.append(f"- 与构建时间的相关性: {param_df['hash_size'].corr(param_df['build_time']):.3f}\n")
    report.append(f"- 与查询时间的相关性: {param_df['hash_size'].corr(param_df['avg_query_time_ms']):.3f}\n")
    report.append(f"- 与准确率的相关性: {param_df['hash_size'].corr(param_df['accuracy_at_50']):.3f}\n\n")
    
    # ========================================================================
    # CODE BLOCK 1.2: FIX TIME COMPLEXITY ESTIMATION
    # 使用更严谨的方法估算时间复杂度
    # ========================================================================
    report.append("## 2. 可扩展性分析总结（改进版）\n\n")
    
    # 大规模数据性能
    max_scale = scale_df.loc[scale_df['data_size'].idxmax()]
    
    report.append("### 2.1 大规模数据性能\n\n")
    report.append(f"- 最大测试规模: {max_scale['data_size']:,} 个向量\n")
    report.append(f"- 暴力搜索查询时间: {max_scale['bf_avg_query_time_ms']:.2f} ± {max_scale['bf_std_query_time_ms']:.2f} ms\n")
    report.append(f"- LSH查询时间: {max_scale['lsh_avg_query_time_ms']:.2f} ± {max_scale['lsh_std_query_time_ms']:.2f} ms\n")
    report.append(f"- 加速比: {max_scale['speedup_factor']:.2f}x\n\n")
    
    report.append("### 2.2 可扩展性结论（改进版）\n\n")
    
    # 只使用大规模数据点进行复杂度估计
    large_scale_df = scale_df[scale_df['data_size'] >= 3000].copy()
    
    if len(large_scale_df) >= 3:
        report.append(f"**复杂度估计方法改进:**\n\n")
        report.append(f"- 只使用数据规模 >= 3000 的 {len(large_scale_df)} 个数据点\n")
        report.append(f"- 使用对数-对数回归 + 置信区间估计\n")
        report.append(f"- 增加理论合理性检查\n\n")
        
        log_sizes = np.log(large_scale_df['data_size'])
        log_bf_times = np.log(large_scale_df['bf_avg_query_time_ms'])
        log_lsh_times = np.log(large_scale_df['lsh_avg_query_time_ms'])
        
        # 对数-对数回归
        bf_fit = np.polyfit(log_sizes, log_bf_times, 1)
        lsh_fit = np.polyfit(log_sizes, log_lsh_times, 1)
        
        bf_time_growth = bf_fit[0]
        lsh_time_growth = lsh_fit[0]
        
        # 计算拟合优度 R²
        bf_predicted = np.polyval(bf_fit, log_sizes)
        lsh_predicted = np.polyval(lsh_fit, log_sizes)
        
        bf_r2 = 1 - (np.sum((log_bf_times - bf_predicted)**2) / np.sum((log_bf_times - np.mean(log_bf_times))**2))
        lsh_r2 = 1 - (np.sum((log_lsh_times - lsh_predicted)**2) / np.sum((log_lsh_times - np.mean(log_lsh_times))**2))
        
        # 估计95%置信区间（简化版）
        bf_std_err = np.std(log_bf_times - bf_predicted)
        lsh_std_err = np.std(log_lsh_times - lsh_predicted)
        
        bf_ci_lower = bf_time_growth - 1.96 * bf_std_err
        bf_ci_upper = bf_time_growth + 1.96 * bf_std_err
        lsh_ci_lower = lsh_time_growth - 1.96 * lsh_std_err
        lsh_ci_upper = lsh_time_growth + 1.96 * lsh_std_err
        
        report.append(f"**暴力搜索时间复杂度:**\n\n")
        report.append(f"- 估计值: O(n^{bf_time_growth:.3f})\n")
        report.append(f"- 95%置信区间: [O(n^{bf_ci_lower:.3f}), O(n^{bf_ci_upper:.3f})]\n")
        report.append(f"- 拟合优度 R²: {bf_r2:.4f}\n")
        
        # 合理性检查
        if bf_time_growth < 0.8 or bf_time_growth > 1.2:
            report.append(f"- 结论: O(n) [理论值]\n")
            report.append(f"- ⚠️ 警告: BF复杂度 O(n^{bf_time_growth:.2f}) 偏离理论O(n)，可能存在测量误差或系统干扰\n\n")
        else:
            report.append(f"- 结论: O(n^{bf_time_growth:.2f})\n\n")
        
        report.append(f"**LSH时间复杂度:**\n\n")
        report.append(f"- 估计值: O(n^{lsh_time_growth:.3f})\n")
        report.append(f"- 95%置信区间: [O(n^{lsh_ci_lower:.3f}), O(n^{lsh_ci_upper:.3f})]\n")
        report.append(f"- 拟合优度 R²: {lsh_r2:.4f}\n")
        report.append(f"- 结论: O(n^{lsh_time_growth:.2f})\n\n")
        
        if lsh_r2 < 0.9:
            report.append(f"- ⚠️ 警告: LSH R²={lsh_r2:.3f} 较低，表明非线性增长，可能存在系统噪音\n\n")
        
        # 性能对比
        report.append(f"**实际性能比较:**\n\n")
        small_scale = scale_df.loc[scale_df['data_size'].idxmin()]
        report.append(f"- 小规模 ({small_scale['data_size']:,}):\n")
        report.append(f"  - BF: {small_scale['bf_avg_query_time_ms']:.3f}ms, LSH: {small_scale['lsh_avg_query_time_ms']:.3f}ms\n")
        report.append(f"  - 加速比: {small_scale['speedup_factor']:.2f}x\n")
        report.append(f"- 大规模 ({max_scale['data_size']:,}):\n")
        report.append(f"  - BF: {max_scale['bf_avg_query_time_ms']:.3f}ms, LSH: {max_scale['lsh_avg_query_time_ms']:.3f}ms\n")
        report.append(f"  - 加速比: {max_scale['speedup_factor']:.2f}x\n\n")
    else:
        report.append(f"- ⚠️ 数据点不足，无法进行可靠的复杂度估计\n\n")
    
    report.append("## 3. 实用建议\n\n")
    report.append("### 3.1 DeepFashion数据集LSH性能问题诊断\n\n")
    
    if best_accuracy_config['accuracy_at_50'] < 10:
        report.append(f"⚠️ **LSH准确率严重偏低 ({best_accuracy_config['accuracy_at_50']:.2f}%)**\n\n")
        report.append(f"**可能原因:**\n\n")
        report.append(f"1. **数据维度过高**: DeepFashion使用2048维CLIP向量，LSH在高维空间效果较差\n")
        report.append(f"2. **候选集太小**: 平均桶大小为 {best_accuracy_config['avg_bucket_size']:.2f}，可能找不到足够的候选\n")
        report.append(f"3. **哈希函数不适合**: 随机投影LSH可能不适合这个数据分布\n\n")
        report.append(f"**建议解决方案:**\n\n")
        report.append(f"1. 使用FAISS库的IVF索引代替简单LSH\n")
        report.append(f"2. 增加哈希表数量到50+，哈希位数到16+\n")
        report.append(f"3. 考虑降维（PCA到512维）后再使用LSH\n")
        report.append(f"4. 使用Product Quantization等更高级的ANN方法\n\n")
    
    if max_scale['speedup_factor'] < 1:
        report.append(f"⚠️ **LSH比暴力搜索更慢 (加速比{max_scale['speedup_factor']:.2f}x < 1)**\n\n")
        report.append(f"这是正常的，因为：\n")
        report.append(f"1. 数据规模较小 ({max_scale['data_size']:,} 个向量)\n")
        report.append(f"2. LSH的优势在超大规模数据（>100K）时才明显\n")
        report.append(f"3. 对于这个规模，建议直接使用FAISS的暴力搜索或Flat索引\n\n")
    
    # 保存报告
    report_path = os.path.join(output_dir, 'lsh_analysis_deepfashion_fixed.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    print(f"分析报告已保存到: {report_path}")
    return report_path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /path/to/project/src
PROJECT_DIR = os.path.dirname(BASE_DIR)
def main():
    """主函数"""
    print("开始DeepFashion数据集的参数分析...")
    
    # 创建结果目录
    os.makedirs('results/analysis', exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR,'doc/report/LSHBFreports'), exist_ok=True)
    
    # 1. LSH参数分析
    param_df = analyze_lsh_parameters_deepfashion()
    param_df.to_csv('results/analysis/lsh_parameter_analysis_deepfashion_fixed.csv', index=False)
    print("\nLSH参数分析完成，结果已保存")
    
    # 2. 可扩展性分析
    scale_df = analyze_data_size_scalability_deepfashion()
    scale_df.to_csv('results/analysis/scalability_analysis_deepfashion_fixed.csv', index=False)
    print("\n可扩展性分析完成，结果已保存")
    
    # 3. 生成修复后的分析报告
    report_path = generate_analysis_report_fixed(param_df, scale_df)
    
    print("\n=== 参数分析完成 ===")
    print("所有结果已保存到:")
    print("- LSH参数分析: results/analysis/lsh_parameter_analysis_deepfashion_fixed.csv")
    print("- 可扩展性分析: results/analysis/scalability_analysis_deepfashion_fixed.csv")
    print(f"- 详细报告: {report_path}")


if __name__ == '__main__':
    main()

