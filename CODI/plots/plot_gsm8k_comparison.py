#!/usr/bin/env python
"""
绘制 4 种模型在 GSM8K 数据集上的性能对比柱状图
模型: baseModel, euclidean, geodesic, hyperbolic
"""

import matplotlib.pyplot as plt
import numpy as np
from color_config import COLOR_LIST, BAR_EDGE_COLOR, GRID_ALPHA

# 设置更美观的样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

# 数据来自 all_results.csv
models = ['Basemodel', 'Euclidean', 'Geodesic', 'Hyperbolic']
accuracies = [
    0.5322213798332069,  # baseModel
    0.5610310841546626,  # euclidean
    0.533737680060652,   # geodesic
    0.535253980288097,   # hyperbolic
]

# 转换为百分比
accuracies_pct = [acc * 100 for acc in accuracies]

# 使用统一配色
colors = COLOR_LIST[:4]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#FFFFFF')
ax.set_facecolor('#FFFFFF')

# 绘制柱状图
bars = ax.bar(models, accuracies_pct, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=1.2, width=0.6)

# 找出最佳模型索引
best_idx = np.argmax(accuracies_pct)

# 在每个柱子上添加数值标签
for i, (bar, acc) in enumerate(zip(bars, accuracies_pct)):
    height = bar.get_height()
    label = f'{acc:.2f}%'
    ax.annotate(label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#333333')

# 设置标签（无标题）
ax.set_xlabel('Model', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', labelpad=10)

# 设置 y 轴范围
distance = 0.5 * (max(accuracies_pct) - min(accuracies_pct))
ax.set_ylim(min(accuracies_pct) - distance, max(accuracies_pct) + distance)

# 美化坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#DDDDDD')
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(colors='#555555', labelsize=12)

# 添加淡色网格线
ax.yaxis.grid(True, linestyle='--', alpha=GRID_ALPHA, color='#CCCCCC')
ax.set_axisbelow(True)

# 调整布局
plt.tight_layout()

# 保存图片
output_path = 'results/gsm8k_model_comparison.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#FFFFFF')
print(f"图片已保存至: {output_path}")

plt.close()
