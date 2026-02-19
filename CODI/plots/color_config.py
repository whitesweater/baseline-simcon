"""
统一的颜色配置文件 - 用于所有论文图表
"""

# 5种主题颜色 RGB
COLORS_RGB = {
    'teal': (128, 216, 207),      # 青绿色
    'pink': (255, 159, 159),       # 粉红色
    'orange': (255, 221, 147),     # 浅橙色
    'blue': (153, 200, 254),       # 浅蓝色
    'purple': (152, 154, 202)      # 深紫色
}

# 归一化到 [0, 1] 用于 matplotlib
COLORS = {
    name: tuple(c/255 for c in rgb)
    for name, rgb in COLORS_RGB.items()
}

# 颜色列表（按顺序）
COLOR_LIST = [
    COLORS['teal'],
    COLORS['pink'],
    COLORS['orange'],
    COLORS['blue'],
    COLORS['purple']
]

# 常用配色方案
LINE_COLOR = COLORS['purple']  # 折线图默认颜色
BAR_EDGE_COLOR = 'black'       # 柱状图边框颜色
GRID_ALPHA = 0.7               # 网格透明度
BAR_ALPHA = 0.75               # 柱状图透明度（组合图用）
