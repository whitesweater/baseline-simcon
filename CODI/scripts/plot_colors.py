#!/usr/bin/env python
"""
统一的颜色配置文件
用于所有绘图脚本的配色管理
"""

# =============================================================================
# 多巴胺色系 - 明亮鲜艳活泼
# =============================================================================
DOPAMINE_COLORS = {
    'coral':     '#FF6B6B',  # 珊瑚红
    'teal':      '#4ECDC4',  # 蒂芙尼蓝
    'yellow':    '#FFE66D',  # 柠檬黄
    'mint':      '#95E1D3',  # 薄荷绿
}

# =============================================================================
# 模型配色映射 (统一管理)
# =============================================================================
MODEL_COLORS = {
    # GSM8K comparison 模型
    'baseModel':   '#FF6B6B',  # 珊瑚红
    'euclidean':   '#4ECDC4',  # 蒂芙尼蓝
    'geodesic':    '#FFE66D',  # 柠檬黄
    'hyperbolic':  '#95E1D3',  # 薄荷绿
    
    # Latent sweep 模型 (使用相同配色)
    'codi':          '#FF6B6B',  # 珊瑚红
    'codi_sircl':    '#4ECDC4',  # 蒂芙尼蓝
    'simcon':        '#FFE66D',  # 柠檬黄
    'simcon_sircl':  '#95E1D3',  # 薄荷绿
}

# 有序列表 (用于柱状图等需要固定顺序的场景)
GSM8K_MODELS = ['baseModel', 'euclidean', 'geodesic', 'hyperbolic']
GSM8K_COLORS = [MODEL_COLORS[m] for m in GSM8K_MODELS]

LATENT_SWEEP_MODELS = ['codi', 'codi_sircl', 'simcon', 'simcon_sircl']
LATENT_SWEEP_COLORS = [MODEL_COLORS[m] for m in LATENT_SWEEP_MODELS]

# 标记样式
MODEL_MARKERS = {
    'codi':          'o',
    'codi_sircl':    's',
    'simcon':        '^',
    'simcon_sircl':  'D',
}

# =============================================================================
# 辅助颜色
# =============================================================================
BEST_BADGE_COLOR = '#4ECDC4'  # Best 标记使用蒂芙尼蓝
BEST_BADGE_BG = '#E8FAF8'     # Best 标记背景
