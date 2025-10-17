# 文件清理建议 - LY-Results目录

基于最终方案：**PCA(512维) + 优化LSH v2**

## 📋 清理分类

### ✅ 必须保留的核心文件

#### code/ 目录
- ✅ `brute_force_search.py` - Ground truth算法
- ✅ `lsh_search.py` - 原始LSH实现
- ✅ `lsh_optimized.py` - 优化LSH实现（Multi-Probe）
- ✅ `test_pca_lsh.py` - PCA降维测试（最终方案验证）
- ✅ `parameter_analysis_fixed.py` - 参数分析（最新版）
- ✅ `unified_evaluation_en.py` - 统一评估框架
- ✅ `visualize_pca_lsh_en.py` - 英文可视化生成

#### results/optimization/ 目录
- ✅ `pca_lsh_analysis_en.png` - **主要图表**（全英文，无乱码）
- ✅ `pca_dimension_analysis.csv` - 降维实验数据
- ✅ `pca_512_config_analysis.csv` - 512维配置对比数据
- ✅ `lsh_optimization_comparison.csv` - 优化对比数据
- ✅ `lsh_optimization_visualization_en.png` - LSH优化可视化（英文）
- ✅ `lsh_parameter_trends_en.png` - 参数趋势图（英文）

#### results/analysis/ 目录
- ✅ `bucket_distribution.png` - 桶分布分析
- ✅ `candidate_set_analysis.png` - 候选集分析
- ✅ `data_characteristics.png` - 数据特征分析
- ✅ `lsh_parameter_analysis_deepfashion_fixed.csv` - DeepFashion参数分析数据
- ✅ `scalability_analysis_deepfashion_fixed.csv` - 可扩展性分析数据

---

### 🗑️ 可以删除的文件（已过时/重复/中文版）

#### analysis/ 目录 - **旧版本分析报告**（已被Newest/替代）

```bash
# 删除命令
cd /Users/cradle/Documents/GitHub/SC4020IBS/src/LSHBF/analysis
rm -f analysis_report_en.md          # 旧的英文报告
rm -f analysis_report.md              # 旧的中文报告
rm -f lsh_parameter_analysis_en.csv   # 旧的参数分析CSV
rm -f lsh_parameter_analysis.csv      # 旧的中文CSV
rm -f lsh_parameter_analysis.png      # 旧的参数分析图
rm -f scalability_analysis_en.csv     # 旧的可扩展性CSV
rm -f scalability_analysis.csv        # 旧的中文CSV
rm -f scalability_analysis.png        # 旧的可扩展性图
```

**原因**: 这些都是早期版本，已被 `Newest/` 目录中的improved版本替代。

---

#### code/ 目录 - **过时/冗余代码**

```bash
# 删除命令
cd /Users/cradle/Documents/GitHub/SC4020IBS/src/LSHBF/code
rm -f parameter_analysis_improved.py  # 旧的参数分析（已被fixed版本替代）
rm -f lsh_deep_analysis.py            # 诊断用代码（问题已解决）
rm -f visualize_optimization.py       # 中文版可视化（如果只用英文）
```

**原因**: 
- `parameter_analysis_improved.py` → 已被 `parameter_analysis_fixed.py` 替代
- `lsh_deep_analysis.py` → 仅用于问题诊断，现已完成优化
- `visualize_optimization.py` → 中文版（有乱码），已有英文版

---

#### results/ 目录 - **中文版图表和旧的问题诊断**

```bash
# 删除中文版图表
cd /Users/cradle/Documents/GitHub/SC4020IBS/src/LSHBF/results/optimization
rm -f lsh_optimization_visualization.png  # 中文版（有英文版）
rm -f lsh_parameter_trends.png            # 中文版（有英文版）
rm -f pca_lsh_analysis.png                # 中文版有乱码（有英文版）

# 删除旧的问题诊断目录
cd /Users/cradle/Documents/GitHub/SC4020IBS/src/LSHBF/results
rm -rf Nah-Problem_Resolve/               # 早期问题诊断，已解决
```

**原因**: 
- 中文版图表都有对应的英文版（`_en.png`）
- `Nah-Problem_Resolve/` 是早期诊断4-6%准确率问题的，现已解决

---

### ⚠️ 可选删除（根据需求决定）

#### results/ 目录 - **其他数据集的结果**

```bash
# 如果只关注DeepFashion，可以删除Fashion-MNIST结果
cd /Users/cradle/Documents/GitHub/SC4020IBS/src/LSHBF/results
rm -f comparison_plots_fmnist_resnet50_vectors.npy_en.png
rm -f performance_report_fmnist_resnet50_vectors.npy_en.csv
```

**原因**: 这些是Fashion-MNIST数据集的结果，如果只用DeepFashion可以删除。

**保留理由**: 如果需要对比多个数据集的性能，建议保留。

---

## 📊 清理前后对比

### 清理前
```
analysis/: 12个文件（含Newest/子目录）
code/: 11个文件（含__pycache__）
results/: 18个文件（含子目录）
总计: ~41个文件
```

### 清理后
```
analysis/: 3个文件（仅保留Newest/）
code/: 7个文件（核心代码）
results/: 10个文件（英文版 + 关键数据）
总计: ~20个文件
```

**减少约50%的文件！**

---

## 🚀 一键清理脚本

创建自动清理脚本：

```bash
#!/bin/bash
# cleanup_old_files.sh

BASE_DIR="/Users/cradle/Documents/GitHub/SC4020IBS/src/LY-Results"

echo "开始清理旧文件..."

# 1. 清理 analysis/ 旧文件
cd "$BASE_DIR/analysis"
rm -f analysis_report_en.md analysis_report.md
rm -f lsh_parameter_analysis_en.csv lsh_parameter_analysis.csv lsh_parameter_analysis.png
rm -f scalability_analysis_en.csv scalability_analysis.csv scalability_analysis.png
echo "✓ 已清理 analysis/ 旧文件"

# 2. 清理 code/ 过时代码
cd "$BASE_DIR/code"
rm -f parameter_analysis_improved.py lsh_deep_analysis.py visualize_optimization.py
echo "✓ 已清理 code/ 过时代码"

# 3. 清理 results/ 中文版和旧目录
cd "$BASE_DIR/results/optimization"
rm -f lsh_optimization_visualization.png lsh_parameter_trends.png pca_lsh_analysis.png
echo "✓ 已清理中文版图表"

cd "$BASE_DIR/results"
rm -rf Nah-Problem_Resolve/
echo "✓ 已删除旧问题诊断目录"

echo ""
echo "========================================="
echo "清理完成！"
echo "========================================="
echo "保留的核心文件:"
echo "  - code/: 7个核心Python文件"
echo "  - results/optimization/: 英文版图表和数据"
echo "  - results/analysis/: 深度分析结果"
echo "  - analysis/Newest/: 最新分析报告"
echo ""
echo "已删除:"
echo "  - 旧的分析报告（8个文件）"
echo "  - 过时代码（3个文件）"
echo "  - 中文版图表（3个文件）"
echo "  - 问题诊断目录（1个目录）"
echo "========================================="
```

---

## 📝 最终保留的文件结构

```
LY-Results/
├── analysis/
│   └── Newest/
│       ├── analysis_report_improved.md
│       ├── lsh_parameter_analysis_improved.csv
│       ├── lsh_parameter_analysis_improved.png
│       ├── scalability_analysis_improved.csv
│       └── scalability_analysis_improved.png
│
├── code/
│   ├── brute_force_search.py          ← Ground truth
│   ├── lsh_search.py                  ← 原始LSH
│   ├── lsh_optimized.py               ← 优化LSH（核心）
│   ├── test_pca_lsh.py                ← PCA测试（核心）
│   ├── parameter_analysis_fixed.py    ← 参数分析
│   ├── unified_evaluation_en.py       ← 统一评估
│   └── visualize_pca_lsh_en.py       ← 英文可视化
│
├── results/
│   ├── analysis/
│   │   ├── bucket_distribution.png
│   │   ├── candidate_set_analysis.png
│   │   ├── data_characteristics.png
│   │   ├── lsh_parameter_analysis_deepfashion_fixed.csv
│   │   └── scalability_analysis_deepfashion_fixed.csv
│   │
│   ├── optimization/
│   │   ├── pca_lsh_analysis_en.png                  ← 主要图表
│   │   ├── pca_dimension_analysis.csv
│   │   ├── pca_512_config_analysis.csv
│   │   ├── lsh_optimization_comparison.csv
│   │   ├── lsh_optimization_visualization_en.png
│   │   └── lsh_parameter_trends_en.png
│   │
│   ├── comparison_plots_inshop_clip_vectors_gallery.npy_en.png
│   └── performance_report_inshop_clip_vectors_gallery.npy_en.csv
│
└── reports/
    ├── lsh_deep_analysis_DeepFashion.md
    ├── lsh_analysis_deepfashion_fixed.md
    ├── lsh_optimization_report.md
    └── pca_dimensionality_reduction_report.md       ← 最新报告
```

---

## 💡 建议

1. **先备份再删除**: 建议先创建一个备份
   ```bash
   cd /Users/cradle/Documents/GitHub/SC4020IBS/src
   tar -czf LSHBF-backup-$(date +%Y%m%d).tar.gz LSHBF/
   ```

2. **分阶段清理**: 可以先删除明显过时的文件，观察一段时间后再删除可选文件

3. **保留文档**: reports/ 目录的所有markdown文件建议保留，便于后续参考

4. **Git提交**: 清理后记得提交到Git，便于回退
   ```bash
   git add .
   git commit -m "Clean up outdated files, keep only final PCA+LSH solution"
   ```

---

**生成时间**: 2025-10-15
**基于方案**: PCA(512维) + 优化LSH v2 (50表, 6位, 5探针)

