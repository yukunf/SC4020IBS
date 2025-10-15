#!/bin/bash
# 更细致的清理脚本 - 删除诊断代码、中间结果、冗余文件

BASE_DIR="/Users/cradle/Documents/GitHub/SC4020IBS/src/LY-Results"

echo "======================================"
echo "开始更细致的清理..."
echo "======================================"
echo ""

DELETED_COUNT=0

# 1. 清理 code/ 诊断和冗余代码
echo "1. 清理 code/ 诊断和冗余代码..."
cd "$BASE_DIR/code"

CODE_TO_DELETE=(
    "lsh_deep_analysis.py"              # 诊断用，问题已解决
    "parameter_analysis_improved.py"    # 已被fixed版本替代
    "visualize_optimization.py"         # 中文版，有英文版
)

for file in "${CODE_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ 删除: $file"
        rm -f "$file"
        ((DELETED_COUNT++))
    fi
done

# 删除 __pycache__
if [ -d "__pycache__" ]; then
    echo "  ✓ 删除: __pycache__/"
    rm -rf "__pycache__"
    ((DELETED_COUNT++))
fi

echo ""

# 2. 清理 results/ 中间分析图表（保留最终PCA结果）
echo "2. 清理 results/analysis/ 中间分析图表..."
cd "$BASE_DIR/results/analysis"

ANALYSIS_TO_DELETE=(
    "bucket_distribution.png"          # 中间诊断图
    "candidate_set_analysis.png"       # 中间诊断图
    "data_characteristics.png"         # 中间诊断图
)

for file in "${ANALYSIS_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ 删除: $file"
        rm -f "$file"
        ((DELETED_COUNT++))
    fi
done

echo ""

# 3. 清理 results/optimization/ 候选集分析（已包含在PCA分析中）
echo "3. 清理 results/optimization/ 冗余分析..."
cd "$BASE_DIR/results/optimization"

OPT_TO_DELETE=(
    "lsh_candidate_set_analysis_en.csv"
    "lsh_candidate_set_analysis_en.png"
)

for file in "${OPT_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ 删除: $file"
        rm -f "$file"
        ((DELETED_COUNT++))
    fi
done

echo ""

# 4. 清理 results/ Fashion-MNIST相关（如果只关注DeepFashion）
echo "4. 清理 Fashion-MNIST数据集结果（保留DeepFashion）..."
cd "$BASE_DIR/results"

FMNIST_TO_DELETE=(
    "comparison_plots_fmnist_resnet50_vectors.npy_en.png"
    "performance_report_fmnist_resnet50_vectors.npy_en.csv"
)

for file in "${FMNIST_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ 删除: $file"
        rm -f "$file"
        ((DELETED_COUNT++))
    fi
done

echo ""

# 5. 清理临时脚本
echo "5. 清理临时脚本..."
cd "$BASE_DIR"

if [ -f "conservative_cleanup.sh" ]; then
    echo "  ✓ 删除: conservative_cleanup.sh"
    rm -f "conservative_cleanup.sh"
    ((DELETED_COUNT++))
fi

echo ""

echo "======================================"
echo "细致清理完成！"
echo "======================================"
echo "本次删除: $DELETED_COUNT 个文件/目录"
echo ""
echo "✅ 最终保留的文件:"
echo ""
echo "📁 code/ (核心代码 - 7个)"
echo "  ├── brute_force_search.py"
echo "  ├── lsh_search.py"
echo "  ├── lsh_optimized.py              ⭐ Multi-Probe LSH"
echo "  ├── test_pca_lsh.py               ⭐ PCA测试"
echo "  ├── parameter_analysis_fixed.py"
echo "  ├── unified_evaluation_en.py"
echo "  └── visualize_pca_lsh_en.py       ⭐ 英文可视化"
echo ""
echo "📁 analysis/Newest/ (最新分析 - 5个)"
echo "  ├── analysis_report_improved.md"
echo "  ├── lsh_parameter_analysis_improved.csv"
echo "  ├── lsh_parameter_analysis_improved.png"
echo "  ├── scalability_analysis_improved.csv"
echo "  └── scalability_analysis_improved.png"
echo ""
echo "📁 results/analysis/ (深度分析数据 - 2个)"
echo "  ├── lsh_parameter_analysis_deepfashion_fixed.csv"
echo "  └── scalability_analysis_deepfashion_fixed.csv"
echo ""
echo "📁 results/optimization/ (优化结果 - 6个)"
echo "  ├── pca_lsh_analysis_en.png       ⭐⭐⭐ 主图表"
echo "  ├── pca_dimension_analysis.csv"
echo "  ├── pca_512_config_analysis.csv"
echo "  ├── lsh_optimization_comparison.csv"
echo "  ├── lsh_optimization_visualization_en.png"
echo "  └── lsh_parameter_trends_en.png"
echo ""
echo "📁 results/ (DeepFashion结果 - 2个)"
echo "  ├── comparison_plots_inshop_clip_vectors_gallery.npy_en.png"
echo "  └── performance_report_inshop_clip_vectors_gallery.npy_en.csv"
echo ""
echo "📁 reports/ (详细报告 - 4个)"
echo "  ├── lsh_deep_analysis_DeepFashion.md"
echo "  ├── lsh_analysis_deepfashion_fixed.md"
echo "  ├── lsh_optimization_report.md"
echo "  └── pca_dimensionality_reduction_report.md  ⭐ 最终报告"
echo ""
echo "🗑️ 共删除类别:"
echo "  - 诊断代码（3个）"
echo "  - 中间分析图表（3个）"
echo "  - 候选集分析（2个）"
echo "  - Fashion-MNIST结果（2个）"
echo "  - 临时脚本（1个）"
echo "  - __pycache__（1个）"
echo ""
echo "💡 核心文件已精简至最小必要集合！"
echo "======================================"

