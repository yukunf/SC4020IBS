import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from src.brute_force_search import BruteForceSearch
from src.lsh_search import LSHIndex
import os
from scipy import stats

def analyze_lsh_parameters():
    """分析LSH参数对性能的影响"""
    print("=== LSH 参数分析 ===")
    
    # 加载数据
    vectors = np.load('data/fmnist_resnet50_vectors.npy')
    labels = np.load('data/fmnist_resnet50_labels.npy')
    
    # 选择较小的测试集以加快分析
    test_size = 500
    query_indices = np.random.choice(len(vectors), test_size, replace=False)
    query_vectors = vectors[query_indices]
    query_labels = labels[query_indices]
    
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
            
            # 测量搜索时间和准确率
            start_time = time.time()
            distances, indices = lsh.search(query_vectors, 50, 'cosine')
            search_time = time.time() - start_time
            
            avg_query_time = (search_time / len(query_vectors)) * 1000  # ms
            
            # 计算召回率
            correct = 0
            for i, query_label in enumerate(query_labels):
                retrieved_labels = labels[indices[i][:10]]  # Top-10
                if query_label in retrieved_labels:
                    correct += 1
            recall_at_10 = (correct / len(query_labels)) * 100
            
            # 获取统计信息
            stats_info = lsh.get_stats()
            
            results.append({
                'num_tables': num_tables,
                'hash_size': hash_size,
                'build_time': build_time,
                'avg_query_time_ms': avg_query_time,
                'recall_at_10': recall_at_10,
                'total_memory_mb': stats_info['total_memory_mb'],
                'avg_bucket_size': stats_info['avg_bucket_size'],
                'total_buckets': stats_info['total_buckets']
            })
    
    return pd.DataFrame(results)

def analyze_data_size_scalability():
    """分析不同数据规模下的性能（改进版）"""
    print("\n=== 数据规模可扩展性分析（改进版）===")
    
    # 加载完整数据
    full_vectors = np.load('data/fmnist_resnet50_vectors.npy')
    full_labels = np.load('data/fmnist_resnet50_labels.npy')
    
    # 改进的数据规模范围 - 增加更大规模和更多中间点
    data_sizes = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000]
    test_query_size = 200
    
    # 多次运行取平均值以减少噪声
    num_runs = 3
    
    results = []
    
    for data_size in data_sizes:
        print(f"测试数据规模: {data_size} (进行 {num_runs} 次运行)")
        
        # 多次运行的结果
        run_results = {
            'bf_build_times': [],
            'bf_query_times': [],
            'lsh_build_times': [],
            'lsh_query_times': [],
            'lsh_recalls': []
        }
        
        for run in range(num_runs):
            # 选择数据子集（每次运行使用不同的随机子集）
            indices = np.random.choice(len(full_vectors), data_size, replace=False)
            vectors = full_vectors[indices]
            labels = full_labels[indices]
            
            # 选择查询样本
            query_indices = np.random.choice(len(vectors), 
                                            min(test_query_size, len(vectors)//10), 
                                            replace=False)
            query_vectors = vectors[query_indices]
            query_labels = labels[query_indices]
            
            # 1. 暴力搜索
            bf = BruteForceSearch('cosine')
            
            # 预热运行（避免首次调用的开销）
            if run == 0 and data_size == data_sizes[0]:
                bf.build_index(vectors[:100])
                bf.search(query_vectors[:10], 10)
            
            start_time = time.perf_counter()  # 使用更精确的计时器
            bf.build_index(vectors)
            bf_build_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            bf_distances, bf_indices = bf.search(query_vectors, 10)
            bf_search_time = time.perf_counter() - start_time
            bf_avg_query_time = (bf_search_time / len(query_vectors)) * 1000
            
            # 2. LSH
            lsh = LSHIndex(hash_family='random_projection', num_tables=10, hash_size=12)
            
            # 预热运行
            if run == 0 and data_size == data_sizes[0]:
                lsh.build_index(vectors[:100])
                lsh.search(query_vectors[:10], 10, 'cosine')
            
            start_time = time.perf_counter()
            lsh.build_index(vectors)
            lsh_build_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            lsh_distances, lsh_indices = lsh.search(query_vectors, 10, 'cosine')
            lsh_search_time = time.perf_counter() - start_time
            lsh_avg_query_time = (lsh_search_time / len(query_vectors)) * 1000
            
            # 计算LSH召回率
            correct = 0
            for i, query_label in enumerate(query_labels):
                retrieved_labels = labels[lsh_indices[i][:10]]
                if query_label in retrieved_labels:
                    correct += 1
            lsh_recall = (correct / len(query_labels)) * 100
            
            # 记录本次运行结果
            run_results['bf_build_times'].append(bf_build_time)
            run_results['bf_query_times'].append(bf_avg_query_time)
            run_results['lsh_build_times'].append(lsh_build_time)
            run_results['lsh_query_times'].append(lsh_avg_query_time)
            run_results['lsh_recalls'].append(lsh_recall)
        
        # 计算平均值和标准差
        bf_avg_query_time = np.mean(run_results['bf_query_times'])
        lsh_avg_query_time = np.mean(run_results['lsh_query_times'])
        
        results.append({
            'data_size': data_size,
            'bf_build_time': np.mean(run_results['bf_build_times']),
            'bf_build_time_std': np.std(run_results['bf_build_times']),
            'bf_avg_query_time_ms': bf_avg_query_time,
            'bf_query_time_std': np.std(run_results['bf_query_times']),
            'lsh_build_time': np.mean(run_results['lsh_build_times']),
            'lsh_build_time_std': np.std(run_results['lsh_build_times']),
            'lsh_avg_query_time_ms': lsh_avg_query_time,
            'lsh_query_time_std': np.std(run_results['lsh_query_times']),
            'lsh_recall_at_10': np.mean(run_results['lsh_recalls']),
            'lsh_recall_std': np.std(run_results['lsh_recalls']),
            'speedup_factor': bf_avg_query_time / lsh_avg_query_time if lsh_avg_query_time > 0 else 0
        })
        
        print(f"  BF: {bf_avg_query_time:.3f}ms, LSH: {lsh_avg_query_time:.3f}ms, "
              f"加速比: {results[-1]['speedup_factor']:.2f}x")
    
    return pd.DataFrame(results)

def estimate_complexity_robust(scale_df, min_data_size=10000):
    """
    稳健的复杂度估计，只使用大规模数据点
    
    参数:
        scale_df: 可扩展性分析结果
        min_data_size: 最小数据规模阈值（默认10000）
    
    返回:
        包含复杂度分析结果的字典
    """
    # 只使用大规模数据点
    large_scale_df = scale_df[scale_df['data_size'] >= min_data_size].copy()
    
    results = {
        'all_points_used': len(scale_df),
        'large_scale_points_used': len(large_scale_df),
        'min_size_threshold': min_data_size,
        'warnings': []
    }
    
    if len(large_scale_df) < 3:
        results['warnings'].append(f"警告：只有 {len(large_scale_df)} 个大规模数据点（>= {min_data_size}），可能不足以进行可靠的复杂度估计")
        # 降低阈值
        min_data_size = scale_df['data_size'].median()
        large_scale_df = scale_df[scale_df['data_size'] >= min_data_size].copy()
        results['min_size_threshold'] = min_data_size
        results['large_scale_points_used'] = len(large_scale_df)
    
    # 对数-对数回归
    log_sizes = np.log(large_scale_df['data_size'].values)
    log_bf_times = np.log(large_scale_df['bf_avg_query_time_ms'].values)
    log_lsh_times = np.log(large_scale_df['lsh_avg_query_time_ms'].values)
    
    # 暴力搜索的复杂度估计
    bf_slope, bf_intercept, bf_r_value, bf_p_value, bf_stderr = stats.linregress(log_sizes, log_bf_times)
    results['bf_complexity_exponent'] = bf_slope
    results['bf_r_squared'] = bf_r_value ** 2
    results['bf_p_value'] = bf_p_value
    results['bf_stderr'] = bf_stderr
    
    # LSH的复杂度估计
    lsh_slope, lsh_intercept, lsh_r_value, lsh_p_value, lsh_stderr = stats.linregress(log_sizes, log_lsh_times)
    results['lsh_complexity_exponent'] = lsh_slope
    results['lsh_r_squared'] = lsh_r_value ** 2
    results['lsh_p_value'] = lsh_p_value
    results['lsh_stderr'] = lsh_stderr
    
    # 合理性检查
    # 暴力搜索应该接近 O(n)，即斜率接近 1.0
    if bf_slope < 0.85 or bf_slope > 1.15:
        results['warnings'].append(
            f"警告：暴力搜索复杂度 O(n^{bf_slope:.2f}) 偏离理论值 O(n)，"
            f"可能存在测量误差或系统干扰"
        )
        results['bf_likely_actual'] = 'O(n) [理论值]'
    else:
        results['bf_likely_actual'] = f'O(n^{bf_slope:.2f})'
    
    # LSH应该是亚线性的，即斜率 < 1.0
    if lsh_slope >= 1.0:
        results['warnings'].append(
            f"警告：LSH复杂度 O(n^{lsh_slope:.2f}) 不是亚线性，"
            f"可能说明：1) LSH配置不当；2) 数据规模还不够大；3) 实现存在问题"
        )
        results['lsh_likely_actual'] = f'O(n^{lsh_slope:.2f}) [异常]'
    else:
        results['lsh_likely_actual'] = f'O(n^{lsh_slope:.2f})'
    
    # R²值检查（拟合优度）
    if bf_r_value ** 2 < 0.90:
        results['warnings'].append(
            f"警告：暴力搜索拟合优度 R²={bf_r_value**2:.3f} 较低，"
            f"说明时间增长不够线性，可能存在系统噪声"
        )
    
    if lsh_r_value ** 2 < 0.90:
        results['warnings'].append(
            f"警告：LSH拟合优度 R²={lsh_r_value**2:.3f} 较低，"
            f"说明时间增长不够线性，可能存在系统噪声"
        )
    
    # 计算置信区间（95%）
    from scipy.stats import t as t_dist
    n = len(large_scale_df)
    t_val = t_dist.ppf(0.975, n - 2)  # 95% 置信度
    
    results['bf_ci_lower'] = bf_slope - t_val * bf_stderr
    results['bf_ci_upper'] = bf_slope + t_val * bf_stderr
    results['lsh_ci_lower'] = lsh_slope - t_val * lsh_stderr
    results['lsh_ci_upper'] = lsh_slope + t_val * lsh_stderr
    
    return results

def create_parameter_analysis_plots(param_df, scale_df):
    """创建参数分析可视化图表"""
    print("\n=== 生成分析图表 ===")
    
    # 创建输出目录
    os.makedirs('results/analysis', exist_ok=True)
    
    # 设置样式
    plt.style.use('seaborn-v0_8')
    # 设置中文字体（macOS使用STHeiti，Windows使用SimHei）
    import platform
    if platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'Heiti TC']
    else:  # Windows/Linux
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 1. LSH参数热力图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LSH参数分析', fontsize=16)
    
    # 构建时间热力图
    pivot_build_time = param_df.pivot(index='num_tables', columns='hash_size', values='build_time')
    sns.heatmap(pivot_build_time, annot=True, fmt='.2f', ax=axes[0,0], cmap='YlOrRd')
    axes[0,0].set_title('构建时间 (秒)')
    axes[0,0].set_xlabel('哈希大小')
    axes[0,0].set_ylabel('哈希表数量')
    
    # 查询时间热力图
    pivot_query_time = param_df.pivot(index='num_tables', columns='hash_size', values='avg_query_time_ms')
    sns.heatmap(pivot_query_time, annot=True, fmt='.1f', ax=axes[0,1], cmap='YlOrRd')
    axes[0,1].set_title('平均查询时间 (ms)')
    axes[0,1].set_xlabel('哈希大小')
    axes[0,1].set_ylabel('哈希表数量')
    
    # 召回率热力图
    pivot_recall = param_df.pivot(index='num_tables', columns='hash_size', values='recall_at_10')
    sns.heatmap(pivot_recall, annot=True, fmt='.1f', ax=axes[0,2], cmap='YlGnBu')
    axes[0,2].set_title('召回率@10 (%)')
    axes[0,2].set_xlabel('哈希大小')
    axes[0,2].set_ylabel('哈希表数量')
    
    # 内存使用热力图
    pivot_memory = param_df.pivot(index='num_tables', columns='hash_size', values='total_memory_mb')
    sns.heatmap(pivot_memory, annot=True, fmt='.1f', ax=axes[1,0], cmap='YlOrRd')
    axes[1,0].set_title('内存使用 (MB)')
    axes[1,0].set_xlabel('哈希大小')
    axes[1,0].set_ylabel('哈希表数量')
    
    # 平均桶大小热力图
    pivot_bucket = param_df.pivot(index='num_tables', columns='hash_size', values='avg_bucket_size')
    sns.heatmap(pivot_bucket, annot=True, fmt='.1f', ax=axes[1,1], cmap='YlGnBu')
    axes[1,1].set_title('平均桶大小')
    axes[1,1].set_xlabel('哈希大小')
    axes[1,1].set_ylabel('哈希表数量')
    
    # 参数权衡散点图
    scatter = axes[1,2].scatter(param_df['avg_query_time_ms'], param_df['recall_at_10'], 
                               c=param_df['num_tables'], s=param_df['hash_size']*10, 
                               alpha=0.7, cmap='viridis')
    axes[1,2].set_xlabel('平均查询时间 (ms)')
    axes[1,2].set_ylabel('召回率@10 (%)')
    axes[1,2].set_title('参数权衡分析\n(颜色=表数量, 大小=哈希大小)')
    plt.colorbar(scatter, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig('results/analysis/lsh_parameter_analysis_improved.png', dpi=300, bbox_inches='tight')
    print("参数分析图表已保存")
    
    # 2. 改进的可扩展性分析图表
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('数据规模可扩展性分析（改进版）', fontsize=16)
    
    # 2.1 查询时间对比（线性坐标）
    axes[0,0].errorbar(scale_df['data_size'], scale_df['bf_avg_query_time_ms'], 
                       yerr=scale_df['bf_query_time_std'], 
                       fmt='o-', label='暴力搜索', linewidth=2, capsize=5)
    axes[0,0].errorbar(scale_df['data_size'], scale_df['lsh_avg_query_time_ms'],
                       yerr=scale_df['lsh_query_time_std'],
                       fmt='s-', label='LSH', linewidth=2, capsize=5)
    axes[0,0].set_xlabel('数据规模')
    axes[0,0].set_ylabel('平均查询时间 (ms)')
    axes[0,0].set_title('查询时间对比（线性坐标）')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2.2 查询时间对比（对数-对数坐标）
    axes[0,1].errorbar(scale_df['data_size'], scale_df['bf_avg_query_time_ms'],
                       yerr=scale_df['bf_query_time_std'],
                       fmt='o-', label='暴力搜索', linewidth=2, capsize=5)
    axes[0,1].errorbar(scale_df['data_size'], scale_df['lsh_avg_query_time_ms'],
                       yerr=scale_df['lsh_query_time_std'],
                       fmt='s-', label='LSH', linewidth=2, capsize=5)
    axes[0,1].set_xlabel('数据规模 (对数)')
    axes[0,1].set_ylabel('平均查询时间 (ms, 对数)')
    axes[0,1].set_title('查询时间对比（对数-对数坐标）')
    axes[0,1].set_xscale('log')
    axes[0,1].set_yscale('log')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3, which='both')
    
    # 添加拟合线（只用大规模数据）
    complexity_results = estimate_complexity_robust(scale_df)
    large_scale_df = scale_df[scale_df['data_size'] >= complexity_results['min_size_threshold']]
    
    if len(large_scale_df) >= 3:
        x_fit = np.logspace(np.log10(large_scale_df['data_size'].min()), 
                           np.log10(large_scale_df['data_size'].max()), 100)
        
        # 暴力搜索拟合线
        bf_exp = complexity_results['bf_complexity_exponent']
        bf_fit_y = (large_scale_df['bf_avg_query_time_ms'].iloc[0] / 
                   (large_scale_df['data_size'].iloc[0] ** bf_exp)) * (x_fit ** bf_exp)
        axes[0,1].plot(x_fit, bf_fit_y, 'r--', alpha=0.5, 
                      label=f'BF拟合: O(n^{bf_exp:.2f})')
        
        # LSH拟合线
        lsh_exp = complexity_results['lsh_complexity_exponent']
        lsh_fit_y = (large_scale_df['lsh_avg_query_time_ms'].iloc[0] / 
                    (large_scale_df['data_size'].iloc[0] ** lsh_exp)) * (x_fit ** lsh_exp)
        axes[0,1].plot(x_fit, lsh_fit_y, 'g--', alpha=0.5,
                      label=f'LSH拟合: O(n^{lsh_exp:.2f})')
        
        axes[0,1].legend()
    
    # 2.3 加速比
    axes[0,2].plot(scale_df['data_size'], scale_df['speedup_factor'], 
                   'g^-', linewidth=2, markersize=8)
    axes[0,2].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='无加速')
    axes[0,2].set_xlabel('数据规模')
    axes[0,2].set_ylabel('加速比 (BF/LSH)')
    axes[0,2].set_title('LSH相对暴力搜索的加速比')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 2.4 构建时间对比
    axes[1,0].errorbar(scale_df['data_size'], scale_df['bf_build_time'],
                       yerr=scale_df['bf_build_time_std'],
                       fmt='o-', label='暴力搜索', linewidth=2, capsize=5)
    axes[1,0].errorbar(scale_df['data_size'], scale_df['lsh_build_time'],
                       yerr=scale_df['lsh_build_time_std'],
                       fmt='s-', label='LSH', linewidth=2, capsize=5)
    axes[1,0].set_xlabel('数据规模')
    axes[1,0].set_ylabel('索引构建时间 (秒)')
    axes[1,0].set_title('索引构建时间对比')
    axes[1,0].legend()
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)
    
    # 2.5 LSH召回率
    axes[1,1].errorbar(scale_df['data_size'], scale_df['lsh_recall_at_10'],
                       yerr=scale_df['lsh_recall_std'],
                       fmt='ro-', linewidth=2, capsize=5)
    axes[1,1].set_xlabel('数据规模')
    axes[1,1].set_ylabel('召回率@10 (%)')
    axes[1,1].set_title('LSH召回率')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim([max(90, scale_df['lsh_recall_at_10'].min() - 5), 105])
    
    # 2.6 复杂度分析摘要
    axes[1,2].axis('off')
    summary_text = "复杂度分析摘要\n" + "="*40 + "\n\n"
    summary_text += f"分析数据点：\n"
    summary_text += f"  总数据点: {complexity_results['all_points_used']}\n"
    summary_text += f"  用于拟合: {complexity_results['large_scale_points_used']} (>= {complexity_results['min_size_threshold']})\n\n"
    
    summary_text += f"暴力搜索复杂度：\n"
    summary_text += f"  估计值: O(n^{complexity_results['bf_complexity_exponent']:.3f})\n"
    summary_text += f"  95%置信区间: [{complexity_results['bf_ci_lower']:.3f}, {complexity_results['bf_ci_upper']:.3f}]\n"
    summary_text += f"  R²: {complexity_results['bf_r_squared']:.4f}\n"
    summary_text += f"  结论: {complexity_results['bf_likely_actual']}\n\n"
    
    summary_text += f"LSH复杂度：\n"
    summary_text += f"  估计值: O(n^{complexity_results['lsh_complexity_exponent']:.3f})\n"
    summary_text += f"  95%置信区间: [{complexity_results['lsh_ci_lower']:.3f}, {complexity_results['lsh_ci_upper']:.3f}]\n"
    summary_text += f"  R²: {complexity_results['lsh_r_squared']:.4f}\n"
    summary_text += f"  结论: {complexity_results['lsh_likely_actual']}\n\n"
    
    if complexity_results['warnings']:
        summary_text += "⚠️ 警告：\n"
        for i, warning in enumerate(complexity_results['warnings'][:3], 1):  # 只显示前3个警告
            # 截断过长的警告
            warning_short = warning if len(warning) < 60 else warning[:57] + "..."
            summary_text += f"{i}. {warning_short}\n"
    
    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('results/analysis/scalability_analysis_improved.png', dpi=300, bbox_inches='tight')
    print("可扩展性分析图表已保存")

def generate_analysis_report(param_df, scale_df):
    """生成详细分析报告（改进版）"""
    print("\n=== 生成分析报告 ===")
    
    report = []
    
    report.append("# 相似度搜索算法参数分析报告（改进版）\n")
    report.append(f"生成时间: {pd.Timestamp.now()}\n")
    report.append(f"改进内容: 使用大规模数据进行复杂度估计，增加置信区间和合理性检查\n")
    
    report.append("## 1. LSH参数分析总结\n")
    
    # 找到最优配置
    best_config = param_df.loc[param_df['recall_at_10'].idxmax()]
    fastest_config = param_df.loc[param_df['avg_query_time_ms'].idxmin()]
    
    report.append("### 1.1 最佳召回率配置")
    report.append(f"- 哈希表数量: {best_config['num_tables']}")
    report.append(f"- 哈希大小: {best_config['hash_size']}")
    report.append(f"- 召回率@10: {best_config['recall_at_10']:.2f}%")
    report.append(f"- 查询时间: {best_config['avg_query_time_ms']:.2f}ms")
    report.append(f"- 构建时间: {best_config['build_time']:.2f}s")
    report.append(f"- 内存使用: {best_config['total_memory_mb']:.2f}MB\n")
    
    report.append("### 1.2 最快查询配置")
    report.append(f"- 哈希表数量: {fastest_config['num_tables']}")
    report.append(f"- 哈希大小: {fastest_config['hash_size']}")
    report.append(f"- 召回率@10: {fastest_config['recall_at_10']:.2f}%")
    report.append(f"- 查询时间: {fastest_config['avg_query_time_ms']:.2f}ms")
    report.append(f"- 构建时间: {fastest_config['build_time']:.2f}s")
    report.append(f"- 内存使用: {fastest_config['total_memory_mb']:.2f}MB\n")
    
    report.append("### 1.3 参数影响分析")
    report.append("**哈希表数量的影响:**")
    report.append(f"- 与构建时间的相关性: {param_df['num_tables'].corr(param_df['build_time']):.3f}")
    report.append(f"- 与查询时间的相关性: {param_df['num_tables'].corr(param_df['avg_query_time_ms']):.3f}")
    report.append(f"- 与召回率的相关性: {param_df['num_tables'].corr(param_df['recall_at_10']):.3f}\n")
    
    report.append("**哈希大小的影响:**")
    report.append(f"- 与构建时间的相关性: {param_df['hash_size'].corr(param_df['build_time']):.3f}")
    report.append(f"- 与查询时间的相关性: {param_df['hash_size'].corr(param_df['avg_query_time_ms']):.3f}")
    report.append(f"- 与召回率的相关性: {param_df['hash_size'].corr(param_df['recall_at_10']):.3f}\n")
    
    report.append("## 2. 可扩展性分析总结（改进版）\n")
    
    # 最大规模的结果
    max_scale = scale_df.loc[scale_df['data_size'].idxmax()]
    
    report.append("### 2.1 大规模数据性能")
    report.append(f"- 最大测试规模: {max_scale['data_size']:,} 个向量")
    report.append(f"- 暴力搜索查询时间: {max_scale['bf_avg_query_time_ms']:.2f} ± {max_scale['bf_query_time_std']:.2f} ms")
    report.append(f"- LSH查询时间: {max_scale['lsh_avg_query_time_ms']:.2f} ± {max_scale['lsh_query_time_std']:.2f} ms")
    report.append(f"- 加速比: {max_scale['speedup_factor']:.2f}x")
    report.append(f"- LSH召回率: {max_scale['lsh_recall_at_10']:.2f} ± {max_scale['lsh_recall_std']:.2f}%\n")
    
    report.append("### 2.2 可扩展性结论（改进版）\n")
    
    # 使用改进的复杂度估计
    complexity_results = estimate_complexity_robust(scale_df, min_data_size=10000)
    
    report.append(f"**复杂度估计方法改进:**")
    report.append(f"- 只使用数据规模 >= {complexity_results['min_size_threshold']} 的 {complexity_results['large_scale_points_used']} 个数据点")
    report.append(f"- 使用对数-对数回归 + 置信区间估计")
    report.append(f"- 增加理论合理性检查\n")
    
    report.append(f"**暴力搜索时间复杂度:**")
    report.append(f"- 估计值: O(n^{complexity_results['bf_complexity_exponent']:.3f})")
    report.append(f"- 95%置信区间: [O(n^{complexity_results['bf_ci_lower']:.3f}), O(n^{complexity_results['bf_ci_upper']:.3f})]")
    report.append(f"- 拟合优度 R²: {complexity_results['bf_r_squared']:.4f}")
    report.append(f"- 结论: {complexity_results['bf_likely_actual']}\n")
    
    report.append(f"**LSH时间复杂度:**")
    report.append(f"- 估计值: O(n^{complexity_results['lsh_complexity_exponent']:.3f})")
    report.append(f"- 95%置信区间: [O(n^{complexity_results['lsh_ci_lower']:.3f}), O(n^{complexity_results['lsh_ci_upper']:.3f})]")
    report.append(f"- 拟合优度 R²: {complexity_results['lsh_r_squared']:.4f}")
    report.append(f"- 结论: {complexity_results['lsh_likely_actual']}\n")
    
    if complexity_results['warnings']:
        report.append(f"**⚠️ 警告和建议:**")
        for warning in complexity_results['warnings']:
            report.append(f"- {warning}")
        report.append("")
    
    # 实际性能比较
    report.append("**实际性能比较:**")
    min_scale = scale_df.loc[scale_df['data_size'].idxmin()]
    report.append(f"- 小规模 ({min_scale['data_size']:,}):")
    report.append(f"  - BF: {min_scale['bf_avg_query_time_ms']:.3f}ms, LSH: {min_scale['lsh_avg_query_time_ms']:.3f}ms")
    report.append(f"  - 加速比: {min_scale['speedup_factor']:.2f}x")
    report.append(f"- 大规模 ({max_scale['data_size']:,}):")
    report.append(f"  - BF: {max_scale['bf_avg_query_time_ms']:.3f}ms, LSH: {max_scale['lsh_avg_query_time_ms']:.3f}ms")
    report.append(f"  - 加速比: {max_scale['speedup_factor']:.2f}x\n")
    
    report.append("## 3. 实用建议\n")
    
    report.append("### 3.1 算法选择指南")
    if max_scale['speedup_factor'] > 1.0:
        # LSH 确实更快
        crossover_point = scale_df[scale_df['speedup_factor'] >= 1.0]['data_size'].min()
        report.append(f"- **小数据 (<{crossover_point:,})**: 使用暴力搜索，简单高效")
        report.append(f"- **中等数据 ({crossover_point:,}-50K)**: 使用LSH，平衡性能和准确性")
        report.append(f"- **大数据 (>50K)**: 必须使用LSH或FAISS，暴力搜索不可行\n")
    else:
        # LSH 还不够快（当前实现或配置有问题）
        report.append(f"- ⚠️ **当前LSH配置在所有测试规模下都比暴力搜索慢**")
        report.append(f"- 建议: 1) 优化LSH参数; 2) 使用更大的数据规模测试; 3) 考虑使用FAISS等优化库\n")
    
    report.append("### 3.2 LSH参数调优指南")
    report.append("- **高准确率需求**: 使用更多哈希表(20-30)和更长哈希(12-16)")
    report.append("- **快速查询需求**: 使用较少哈希表(5-10)和较短哈希(8-10)")
    report.append("- **内存受限**: 减少哈希表数量，接受部分准确率损失")
    report.append("- **构建时间敏感**: 避免过多哈希表，优先优化查询时间\n")
    
    report.append("### 3.3 进一步优化建议")
    report.append("- 考虑使用FAISS库，提供高度优化的ANN实现")
    report.append("- 对于超大规模数据(>1M)，考虑分布式索引")
    report.append("- 实施查询结果缓存以加速重复查询")
    report.append("- 根据具体应用场景调整召回率-速度权衡\n")
    
    # 保存报告
    with open('results/analysis/analysis_report_improved.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("分析报告已保存到: results/analysis/analysis_report_improved.md")

def main():
    """主函数"""
    print("开始参数分析和性能评估（改进版）...")
    print("改进点:")
    print("1. 多次运行取平均，减少随机误差")
    print("2. 使用大规模数据点(>=10K)进行复杂度估计")
    print("3. 添加置信区间和统计检验")
    print("4. 增加理论合理性检查")
    print("5. 提供详细的警告和诊断信息\n")
    
    # 创建结果目录
    os.makedirs('results/analysis', exist_ok=True)
    
    # 1. LSH参数分析
    param_df = analyze_lsh_parameters()
    param_df.to_csv('results/analysis/lsh_parameter_analysis_improved.csv', index=False)
    print("\nLSH参数分析完成，结果已保存")
    
    # 2. 可扩展性分析
    scale_df = analyze_data_size_scalability()
    scale_df.to_csv('results/analysis/scalability_analysis_improved.csv', index=False)
    print("\n可扩展性分析完成，结果已保存")
    
    # 3. 生成可视化图表
    create_parameter_analysis_plots(param_df, scale_df)
    
    # 4. 生成分析报告
    generate_analysis_report(param_df, scale_df)
    
    # 5. 打印复杂度分析结果
    print("\n" + "="*60)
    print("复杂度分析结果")
    print("="*60)
    complexity_results = estimate_complexity_robust(scale_df)
    
    print(f"\n使用数据点: {complexity_results['large_scale_points_used']}/{complexity_results['all_points_used']} ")
    print(f"(只使用规模 >= {complexity_results['min_size_threshold']} 的数据点)\n")
    
    print(f"暴力搜索:")
    print(f"  估计复杂度: O(n^{complexity_results['bf_complexity_exponent']:.3f})")
    print(f"  95%置信区间: [O(n^{complexity_results['bf_ci_lower']:.3f}), O(n^{complexity_results['bf_ci_upper']:.3f})]")
    print(f"  拟合优度 R²: {complexity_results['bf_r_squared']:.4f}")
    print(f"  结论: {complexity_results['bf_likely_actual']}\n")
    
    print(f"LSH:")
    print(f"  估计复杂度: O(n^{complexity_results['lsh_complexity_exponent']:.3f})")
    print(f"  95%置信区间: [O(n^{complexity_results['lsh_ci_lower']:.3f}), O(n^{complexity_results['lsh_ci_upper']:.3f})]")
    print(f"  拟合优度 R²: {complexity_results['lsh_r_squared']:.4f}")
    print(f"  结论: {complexity_results['lsh_likely_actual']}\n")
    
    if complexity_results['warnings']:
        print("⚠️ 警告:")
        for i, warning in enumerate(complexity_results['warnings'], 1):
            print(f"{i}. {warning}")
        print()
    
    print("\n=== 参数分析完成 ===")
    print("所有结果已保存到: results/analysis/")
    print("- LSH参数分析: lsh_parameter_analysis_improved.csv")
    print("- 可扩展性分析: scalability_analysis_improved.csv")
    print("- 参数分析图表: lsh_parameter_analysis_improved.png")
    print("- 可扩展性图表: scalability_analysis_improved.png")
    print("- 详细报告: analysis_report_improved.md")

if __name__ == '__main__':
    main()

