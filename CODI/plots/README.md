# 论文图表生成脚本

## 📁 文件结构

- `color_config.py` - 统一的颜色配置文件（5种主题颜色）
- `plot_ablation.py` - 消融实验可视化脚本
- `results/` - 生成的图表保存目录

## 🎨 颜色方案

所有图表使用统一的5种颜色：
1. **青绿色** (128, 216, 207) - Teal
2. **粉红色** (255, 159, 159) - Pink
3. **浅橙色** (255, 221, 147) - Orange
4. **浅蓝色** (153, 200, 254) - Blue
5. **深紫色** (152, 154, 202) - Purple

## 🚀 使用方法

### 生成消融实验图表

```bash
cd plots
python plot_ablation.py
```

### 在其他脚本中使用统一配色

```python
from color_config import COLOR_LIST, COLORS, LINE_COLOR

# 使用颜色列表
plt.bar(x, y, color=COLOR_LIST[0])

# 使用命名颜色
plt.plot(x, y, color=COLORS['purple'])
```

## 📊 可用图表类型

1. **柱状图** - `plot_gsm8k_ablation()`
2. **折线图** - `plot_gsm8k_ablation_line()`
3. **组合图** - `plot_gsm8k_ablation_combo()` (柱状+折线)

## 📝 添加新图表

在 `plots/` 目录下创建新的脚本文件，导入 `color_config` 使用统一配色。
