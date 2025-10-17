# LSH与Brute Force相似度搜索实验

## 项目概述

本项目实现了两种相似度搜索算法（Brute Force和LSH），并在两个图像数据集上进行了完整的性能对比实验。

## 文件结构

```
LY-Results/
├── README.md                           # 【本文件】项目说明
│
├── code/                               # 核心代码
│   ├── brute_force_search.py          # Brute Force算法实现
│   ├── lsh_search.py                  # LSH算法实现
│   ├── unified_evaluation_en.py       # 统一评测框架
│   └── parameter_analysis_en.py       # 参数分析工具
│
├── results/                            # 实验结果
│   ├── performance_report_fmnist_resnet50_vectors.npy_en.csv        # Fashion-MNIST性能报告
│   ├── comparison_plots_fmnist_resnet50_vectors.npy_en.png          # Fashion-MNIST对比图
│   ├── performance_report_inshop_clip_vectors_gallery.npy_en.csv    # DeepFashion性能报告
│   └── comparison_plots_inshop_clip_vectors_gallery.npy_en.png      # DeepFashion对比图
│
├── reports/                            # 实验报告
│   ├── 双数据集完整实验报告.md        # 完整分析报告（Markdown）
│   └── 双数据集完整实验报告.pdf       # 完整分析报告（PDF）
│
├── docs/                               # 使用文档
│   ├── 完整实验使用指南.md            # 使用指南（Markdown）
│   └── 完整实验使用指南.pdf           # 使用指南（PDF）
│
└── analysis/                           # 参数分析结果
    ├── analysis_report_en.md           # 参数分析报告
    ├── lsh_parameter_analysis_en.csv   # 参数实验数据
    ├── lsh_parameter_analysis.png      # 参数热力图
    ├── scalability_analysis_en.csv     # 可扩展性数据
    └── scalability_analysis.png        # 可扩展性趋势图
```

## 快速开始

### 环境要求

```bash
pip install numpy pandas matplotlib seaborn tqdm
```

### 运行实验

假设你已经有数据文件在 `../data/` 目录下：

```bash
# 运行完整评测（两个数据集）
python code/unified_evaluation_en.py

# 运行参数分析
python code/parameter_analysis_en.py
```

### 查看结果

- **性能报告**：`results/performance_report_*_en.csv`
- **可视化图表**：`results/comparison_plots_*_en.png`
- **完整分析**：`reports/双数据集完整实验报告.pdf`
- **使用指南**：`docs/完整实验使用指南.pdf`

## 实验结果摘要

### 数据集1：Fashion-MNIST (70,000样本)

| 算法 | 查询时间(ms) | QPS | Recall@50 | Accuracy@50 |
|------|--------------|-----|-----------|-------------|
| Brute Force | 0.95 | 1,048 | 100.00% | - |
| LSH配置1 (5表,8位) | 39.09 | 26 | 100.00% | 86.75% |
| LSH配置2 (10表,10位) | 40.64 | 25 | 100.00% | 97.24% |
| LSH配置3 (20表,12位) | 36.07 | 28 | 100.00% | 98.64% |

### 数据集2：DeepFashion (15,000样本)

| 算法 | 查询时间(ms) | QPS | Accuracy@50 |
|------|--------------|-----|-------------|
| Brute Force | 0.22 | 4,485 | - |
| LSH配置1 (5表,8位) | 2.60 | 385 | 6.64% |
| LSH配置2 (10表,10位) | 0.97 | 1,031 | 5.00% |
| LSH配置3 (20表,12位) | 0.52 | 1,918 | 4.02% |

## 关键发现

1. **数据规模影响显著**：Brute Force在小数据集(15K)上表现优异，在大数据集(70K)上仍然可用
2. **LSH需要合适场景**：在Fashion-MNIST上LSH反而比Brute Force慢，因为候选集过大
3. **参数调优重要**：不同的num_tables和hash_size配置对性能影响很大
4. **数据分布关键**：DeepFashion数据集上LSH表现不佳，可能因为数据规模较小且分布特殊

## 算法说明

### Brute Force Search

**原理**：遍历所有向量，计算与查询向量的距离，返回Top-K

**优势**：
- 100%准确率
- 实现简单
- 索引构建快

**劣势**：
- 查询时间随数据规模线性增长
- 不适合大规模数据

### LSH (Locality Sensitive Hashing)

**原理**：使用哈希函数将相似向量映射到同一个桶中，只搜索桶内候选

**优势**：
- 查询速度快（理论上）
- 支持大规模数据
- 可调节精度/速度权衡

**劣势**：
- 需要参数调优
- 准确率可能<100%
- 索引构建时间较长

## 代码使用示例

### Brute Force

```python
from brute_force_search import BruteForceSearch

# 初始化
bf = BruteForceSearch(metric='cosine')
bf.build_index(vectors)

# 搜索
distances, indices = bf.search(query_vectors, k=50)
```

### LSH

```python
from lsh_search import LSHIndex

# 初始化
lsh = LSHIndex(
    hash_family='random_projection',
    num_tables=10,
    hash_size=10
)
lsh.build_index(vectors)

# 搜索
distances, indices = lsh.search(query_vectors, k=50, metric='cosine')
```

## 课程要求达成

✅ **实现至少两种方法**：Brute Force + LSH (3种配置)  
✅ **在至少两个数据集上评测**：Fashion-MNIST + DeepFashion  
✅ **进行实证比较**：详细的性能分析、参数讨论、成功/失败案例  

## 参考资料

- 完整分析报告：`reports/双数据集完整实验报告.pdf`
- 使用指南：`docs/完整实验使用指南.pdf`
- 参数分析：`analysis/analysis_report_en.md`

## 作者

SC4020 数据库系统实现 - 任务B

**负责内容**：LSH与Brute Force算法实现与对比

**完成时间**：2025年10月

---

如有问题，请参考 `docs/完整实验使用指南.pdf` 中的常见问题部分。

