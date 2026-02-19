#!/usr/bin/env python3
"""
从 latent_sweep_gsm8k 结果目录提取数据并画图

横坐标: latent token num (1-18)
纵坐标: GSM8K 准确率

图1: codi vs codi_sircl
图2: simcon vs sircl
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 添加 plots 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'plots'))
from color_config import COLOR_LIST, COLORS

# 模型显示名称映射
MODEL_DISPLAY_NAMES = {
    'codi': 'CODI',
    'codi_sircl': 'CODI+SIRCL',
    'simcon': 'Sim-CoT',
    'simcon_sircl': 'Sim-CoT+SIRCL'
}

# 模型颜色映射（使用用户指定的两组颜色）
# 第一组：#F1703E 和 #4B93EF
# 第二组：#9E44B4 和 #69B261
MODEL_COLORS = {
    'codi': '#F1703E',
    'codi_sircl': '#4B93EF',
    'simcon': '#9E44B4',
    'simcon_sircl': '#69B261'
}

# 模型标记样式
MODEL_MARKERS = {
    'codi': 'o',
    'codi_sircl': 's',
    'simcon': '^',
    'simcon_sircl': 'D'
}

# 设置中文字体支持（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def collect_results(results_dir: str) -> pd.DataFrame:
    """从各个 latent_N 目录收集结果"""
    all_data = []
    
    results_path = Path(results_dir)
    
    for latent_dir in sorted(results_path.glob("latent_*")):
        if not latent_dir.is_dir():
            continue
        
        # 提取 latent 数值
        try:
            latent_num = int(latent_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        
        # 读取 all_results.csv
        csv_path = latent_dir / "summary" / "all_results.csv"
        if not csv_path.exists():
            print(f"警告: {csv_path} 不存在")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            df['latent_num'] = latent_num
            all_data.append(df)
        except Exception as e:
            print(f"读取 {csv_path} 失败: {e}")
            continue
    
    if not all_data:
        raise ValueError(f"未找到任何结果数据: {results_dir}")
    
    return pd.concat(all_data, ignore_index=True)


def plot_latent_sweep(df: pd.DataFrame, output_dir: str):
    """生成两个图: codi组 和 simcon组"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 按模型和 latent_num 分组，取平均准确率
    summary = df.groupby(['model', 'latent_num'])['accuracy'].mean().reset_index()
    
    # 打印数据摘要
    print("\n" + "="*60)
    print("数据摘要")
    print("="*60)
    pivot = summary.pivot(index='latent_num', columns='model', values='accuracy')
    print(pivot.to_string())
    
    # 保存数据到 CSV
    csv_path = os.path.join(output_dir, "latent_sweep_summary.csv")
    pivot.to_csv(csv_path)
    print(f"\n数据已保存: {csv_path}")
    
    # =========================================================================
    # 图1: codi vs codi_sircl
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    models_group1 = ['codi', 'codi_sircl']
    colors_group1 = [MODEL_COLORS['codi'], MODEL_COLORS['codi_sircl']]
    markers_group1 = [MODEL_MARKERS['codi'], MODEL_MARKERS['codi_sircl']]
    
    for model, color, marker in zip(models_group1, colors_group1, markers_group1):
        if model in summary['model'].values:
            model_data = summary[summary['model'] == model].sort_values('latent_num')
            ax1.plot(model_data['latent_num'], model_data['accuracy'] * 100, 
                    marker=marker, color=color, linewidth=2, markersize=8,
                    label=MODEL_DISPLAY_NAMES[model], alpha=0.9)
    
    ax1.set_xlabel('Latent Tokens Number', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 19))
    ax1.set_xlim(0.5, 18.5)
    
    # 设置 Y 轴范围
    y_min = summary[summary['model'].isin(models_group1)]['accuracy'].min() * 100 - 5
    y_max = summary[summary['model'].isin(models_group1)]['accuracy'].max() * 100 + 5
    ax1.set_ylim(max(0, y_min), min(100, y_max))
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "latent_sweep_codi_group.png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"图1已保存: {fig1_path}")
    plt.close(fig1)
    
    # =========================================================================
    # 图2: simcon vs sircl
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    models_group2 = ['simcon', 'simcon_sircl']
    colors_group2 = [MODEL_COLORS['simcon'], MODEL_COLORS['simcon_sircl']]
    markers_group2 = [MODEL_MARKERS['simcon'], MODEL_MARKERS['simcon_sircl']]
    
    for model, color, marker in zip(models_group2, colors_group2, markers_group2):
        if model in summary['model'].values:
            model_data = summary[summary['model'] == model].sort_values('latent_num')
            ax2.plot(model_data['latent_num'], model_data['accuracy'] * 100, 
                    marker=marker, color=color, linewidth=2, markersize=8,
                    label=MODEL_DISPLAY_NAMES[model], alpha=0.9)
    
    ax2.set_xlabel('Latent Tokens Number', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 19))
    ax2.set_xlim(0.5, 18.5)
    
    # 设置 Y 轴范围
    if summary[summary['model'].isin(models_group2)].shape[0] > 0:
        y_min = summary[summary['model'].isin(models_group2)]['accuracy'].min() * 100 - 5
        y_max = summary[summary['model'].isin(models_group2)]['accuracy'].max() * 100 + 5
        ax2.set_ylim(max(0, y_min), min(100, y_max))
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "latent_sweep_simcon_group.png")
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"图2已保存: {fig2_path}")
    plt.close(fig2)
    
    # =========================================================================
    # 图3: 所有模型对比 (合并图)
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    all_models = ['codi', 'codi_sircl', 'simcon', 'simcon_sircl']
    all_colors = [MODEL_COLORS[m] for m in all_models]
    all_markers = [MODEL_MARKERS[m] for m in all_models]
    
    for model, color, marker in zip(all_models, all_colors, all_markers):
        if model in summary['model'].values:
            model_data = summary[summary['model'] == model].sort_values('latent_num')
            ax3.plot(model_data['latent_num'], model_data['accuracy'] * 100, 
                    marker=marker, color=color, linewidth=2, markersize=8,
                    label=MODEL_DISPLAY_NAMES[model], alpha=0.9)
    
    ax3.set_xlabel('Latent Token Number', fontsize=12)
    ax3.set_ylabel('GSM8K Accuracy (%)', fontsize=12)
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(1, 19))
    ax3.set_xlim(0.5, 18.5)
    
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "latent_sweep_all_models.png")
    fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"图3已保存: {fig3_path}")
    plt.close(fig3)
    
    return summary


def main():
    # 配置路径
    results_dir = "/data/yhao/baseline/CODI/results/latent_sweep_gsm8k"
    output_dir = "/data/yhao/baseline/CODI/plots/results"
    
    print("="*60)
    print("Latent Sweep 结果可视化")
    print("="*60)
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    
    # 收集数据
    print("\n正在收集数据...")
    df = collect_results(results_dir)
    print(f"共收集到 {len(df)} 条记录")
    print(f"模型: {df['model'].unique().tolist()}")
    print(f"Latent 范围: {df['latent_num'].min()} - {df['latent_num'].max()}")
    
    # 画图
    print("\n正在生成图表...")
    summary = plot_latent_sweep(df, output_dir)
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)
    print(f"\n输出文件:")
    print(f"  - {output_dir}/latent_sweep_summary.csv")
    print(f"  - {output_dir}/latent_sweep_codi_group.png")
    print(f"  - {output_dir}/latent_sweep_simcon_group.png")
    print(f"  - {output_dir}/latent_sweep_all_models.png")


if __name__ == "__main__":
    main()
