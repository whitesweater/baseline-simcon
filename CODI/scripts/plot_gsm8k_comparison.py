#!/usr/bin/env python
"""
绘制 4 种模型在 GSM8K 数据集上的性能对比柱状图
模型: baseModel, euclidean, geodesic, hyperbolic
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_colors import GSM8K_COLORS, BEST_BADGE_COLOR, BEST_BADGE_BG

# 设置更美观的样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

# 数据来自 all_results.csv
models = ['BaseModel', 'Euclidean', 'Geodesic', 'Hyperbolic']
accuracies = [
    0.5322213798332069,  # baseModel
    0.5610310841546626,  # euclidean
    0.533737680060652,   # geodesic
    0.535253980288097,   # hyperbolic
]

# 转换为百分比
accuracies_pct = [acc * 100 for acc in accuracies]

# 使用统一配色
colors = GSM8K_COLORS

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#FFFFFF')
ax.set_facecolor('#FFFFFF')

# 绘制柱状图 - 圆角效果通过较粗的边框模拟
bars = ax.bar(models, accuracies_pct, color=colors, edgecolor='white', linewidth=2, width=0.6)

# 找出最佳模型索引
best_idx = np.argmax(accuracies_pct)

# 在每个柱子上添加数值标签
for i, (bar, acc) in enumerate(zip(bars, accuracies_pct)):
    height = bar.get_height()
    # 最佳模型加标记在数字旁边
    if i == best_idx:
        label = f'{acc:.2f}%  '
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='medium', color='#333333')
        # Best 徽章紧贴数字右侧
        ax.annotate(' Best',
                    xy=(bar.get_x() + bar.get_width() / 2 + 0.18, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='left', va='bottom',
                    fontsize=8, fontweight='bold', color=BEST_BADGE_COLOR,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=BEST_BADGE_BG, 
                              edgecolor=BEST_BADGE_COLOR, linewidth=1.2))
    else:
        label = f'{acc:.2f}%'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='medium', color='#333333')

# 设置标题和标签
ax.set_title('Model Performance Comparison on GSM8K', 
             fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.set_xlabel('Model', fontsize=12, color='#555555', labelpad=10)
ax.set_ylabel('Accuracy (%)', fontsize=12, color='#555555', labelpad=10)

# 设置 y 轴范围
ax.set_ylim(0, max(accuracies_pct) * 1.15)

# 美化坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#DDDDDD')
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(colors='#555555', labelsize=10)

# 添加淡色网格线
ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='#CCCCCC')
ax.set_axisbelow(True)

# 调整布局
plt.tight_layout()

# 保存图片
output_path = '/data/yhao/baseline/CODI/results_useful/plots/gsm8k_model_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#FFFFFF')
print(f"图片已保存至: {output_path}")

# 显示图片
plt.show()
