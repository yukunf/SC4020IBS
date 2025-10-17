"""
Visualize PCA Dimensionality Reduction + LSH Optimization (English Version)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /path/to/project/src
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
# Load results
results_dir = os.path.join(PROJECT_DIR,'doc/report/LSHBFreports','results/optimization')
df_dim = pd.read_csv(f'{results_dir}/pca_dimension_analysis.csv')
df_512 = pd.read_csv(f'{results_dir}/pca_512_config_analysis.csv')

# Translate config names to English
config_translation = {
    '原始LSH (10表,12位)': 'Original LSH\n(10T, 12B)',
    '优化v1 (40表,8位,3探针)': 'Opt v1\n(40T, 8B, 3P)',
    '优化v2 (50表,6位,5探针)': 'Opt v2\n(50T, 6B, 5P)',
    '优化v3 (30表,8位,4探针)': 'Opt v3\n(30T, 8B, 4P)',
    '512维专用 (20表,10位,3探针)': '512D Special\n(20T, 10B, 3P)'
}
df_512['config'] = df_512['config'].map(config_translation)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))

colors_dim = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']
colors_config = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4']

# ============================================================================
# 1. Dimension vs Accuracy
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
ax1.plot(df_dim['dimension'], df_dim['accuracy_at_50'], 'o-', 
         linewidth=2.5, markersize=12, color='#4ECDC4', markeredgecolor='black', markeredgewidth=1.5)
ax1.set_xlabel('Dimension', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy@50 (%)', fontsize=12, fontweight='bold')
ax1.set_title('Dimension vs Accuracy', fontsize=13, fontweight='bold', pad=10)
ax1.invert_xaxis()
ax1.grid(True, alpha=0.3, linestyle='--')

# Add annotations
for i, row in df_dim.iterrows():
    ax1.annotate(f"{row['accuracy_at_50']:.1f}%", 
                (row['dimension'], row['accuracy_at_50']),
                textcoords="offset points", xytext=(0,10), ha='center', 
                fontsize=9, fontweight='bold')

# Highlight best
best_idx = df_dim['accuracy_at_50'].idxmax()
ax1.scatter(df_dim.loc[best_idx, 'dimension'], df_dim.loc[best_idx, 'accuracy_at_50'],
           s=400, facecolors='none', edgecolors='red', linewidths=3, zorder=10)

# ============================================================================
# 2. Dimension vs Query Time
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
ax2.plot(df_dim['dimension'], df_dim['query_time_ms'], 'o-', 
         linewidth=2.5, markersize=12, color='#FF6B6B', markeredgecolor='black', markeredgewidth=1.5)
ax2.set_xlabel('Dimension', fontsize=12, fontweight='bold')
ax2.set_ylabel('Query Time (ms)', fontsize=12, fontweight='bold')
ax2.set_title('Dimension vs Query Time', fontsize=13, fontweight='bold', pad=10)
ax2.invert_xaxis()
ax2.grid(True, alpha=0.3, linestyle='--')

# ============================================================================
# 3. Accuracy vs Variance Retained (Dual Axis)
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
ax3_twin = ax3.twinx()

line1 = ax3.plot(df_dim['dimension'], df_dim['accuracy_at_50'], 'o-', 
                linewidth=2.5, markersize=12, color='#4ECDC4', 
                markeredgecolor='black', markeredgewidth=1.5, label='Accuracy')

# Plot variance for dimensions with PCA (skip first row which is 2048 original)
line2 = ax3_twin.plot(df_dim['dimension'][1:], df_dim['variance_retained'][1:], 's-', 
                     linewidth=2.5, markersize=10, color='#FFA07A', 
                     markeredgecolor='black', markeredgewidth=1.5, label='Variance Retained')

ax3.set_xlabel('Dimension', fontsize=12, fontweight='bold')
ax3.set_ylabel('Accuracy@50 (%)', fontsize=12, fontweight='bold', color='#4ECDC4')
ax3_twin.set_ylabel('Variance Retained (%)', fontsize=12, fontweight='bold', color='#FFA07A')
ax3.set_title('Accuracy vs Variance Retained', fontsize=13, fontweight='bold', pad=10)
ax3.invert_xaxis()
ax3.grid(True, alpha=0.3, linestyle='--')

ax3.tick_params(axis='y', labelcolor='#4ECDC4')
ax3_twin.tick_params(axis='y', labelcolor='#FFA07A')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='best', fontsize=10)

# ============================================================================
# 4. 512D Configs: Accuracy Comparison
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
bars = ax4.barh(range(len(df_512)), df_512['accuracy'], 
               color=colors_config, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_yticks(range(len(df_512)))
ax4.set_yticklabels(df_512['config'].tolist(), fontsize=9)
ax4.set_xlabel('Accuracy@50 (%)', fontsize=12, fontweight='bold')
ax4.set_title('512D Configs: Accuracy Comparison', fontsize=13, fontweight='bold', pad=10)
ax4.grid(axis='x', alpha=0.3, linestyle='--')

for i, (bar, row) in enumerate(zip(bars, df_512.itertuples())):
    ax4.text(row.accuracy + 0.3, i, f"{row.accuracy:.2f}%", 
            va='center', fontsize=9, fontweight='bold')

# Highlight best
best_512_idx = df_512['accuracy'].idxmax()
bars[best_512_idx].set_edgecolor('gold')
bars[best_512_idx].set_linewidth(3)

# ============================================================================
# 5. 512D Configs: Query Time Comparison
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
bars = ax5.barh(range(len(df_512)), df_512['query_time'], 
               color=colors_config, alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.set_yticks(range(len(df_512)))
ax5.set_yticklabels(df_512['config'].tolist(), fontsize=9)
ax5.set_xlabel('Query Time (ms)', fontsize=12, fontweight='bold')
ax5.set_title('512D Configs: Query Time Comparison', fontsize=13, fontweight='bold', pad=10)
ax5.grid(axis='x', alpha=0.3, linestyle='--')

# ============================================================================
# 6. Summary
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Get best 512D result
best_512_row = df_512.iloc[best_512_idx]

# Original 2048D result (from previous experiments)
accuracy_2048 = 14.32
query_time_2048 = 16.70

# Calculate improvements
accuracy_improvement = best_512_row['accuracy'] - accuracy_2048
query_time_improvement = query_time_2048 - best_512_row['query_time']
speedup = query_time_2048 / best_512_row['query_time']

# Get PCA stats for 512D
pca_512_row = df_dim[df_dim['dimension'] == 512].iloc[0]
variance_retained = pca_512_row['variance_retained']

summary_text = f"""
PCA DIMENSIONALITY REDUCTION SUMMARY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best Dimension: 512
Variance Retained: {variance_retained:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2048D (Original):
  Accuracy: {accuracy_2048:.2f}%
  Query Time: {query_time_2048:.2f} ms

512D (PCA Reduced):
  Accuracy: {best_512_row['accuracy']:.2f}%
  Query Time: {best_512_row['query_time']:.2f} ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Improvements:
  Accuracy: {accuracy_improvement:+.2f}%
  Query Time: {query_time_improvement:+.2f} ms
  Speedup: {speedup:.1f}x faster
  Dimension: 75% reduction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recommendation:
✓ Use PCA(512D) + LSH!
  Reason: Higher accuracy
          and faster speed
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#FFFFCC', alpha=0.9, edgecolor='black', linewidth=2))

# ============================================================================
# Main title and layout
# ============================================================================
plt.suptitle('PCA Dimensionality Reduction + LSH Optimization Analysis\nDeepFashion Dataset', 
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_path = f'{results_dir}/pca_lsh_analysis_en.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {output_path}")

plt.show()

print("\n" + "="*70)
print("English version visualization completed!")
print("="*70)

