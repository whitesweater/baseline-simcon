#!/usr/bin/env python3
"""
Latent Token 矩阵秩分析脚本

分析不同数量的 latent token 构成的矩阵的秩（rank）
- 去除最后一个 token
- 将剩余 token 看作矩阵 (num_tokens x embedding_dim)
- 计算矩阵的秩

用法:
    python analyze_matrix_rank.py --latent_nums 7 17 18
    python analyze_matrix_rank.py --all  # 分析所有可用的 latent 数量
"""

import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

# 可视化
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 结果目录
BASE_DIR = Path("/data/yhao/baseline/CODI/results/latent_sweep_gsm8k")
MODELS = ["codi", "codi_sircl", "simcon", "simcon_sircl"]


def load_latents(latent_num: int, model: str = "codi") -> Optional[np.ndarray]:
    """
    加载指定 latent 数量的数据
    
    Returns:
        np.ndarray: shape (num_samples, num_iterations, embedding_dim)
    """
    latents_file = BASE_DIR / f"latent_{latent_num}" / "models" / model / "gsm8k" / "run_0" / "latents.json"
    
    if not latents_file.exists():
        print(f"File not found: {latents_file}")
        return None
    
    with open(latents_file) as f:
        data = json.load(f)
    
    latents = np.array(data['latents'], dtype=np.float32)
    print(f"Loaded latent_{latent_num}/{model}: shape={latents.shape}")
    return latents


def compute_matrix_rank(matrix: np.ndarray, tol: float = None) -> int:
    """
    计算矩阵的秩
    
    Args:
        matrix: 2D numpy array
        tol: 奇异值的截断阈值，默认为 max(M,N) * eps * max(singular_values)
    
    Returns:
        int: 矩阵的秩
    """
    return np.linalg.matrix_rank(matrix, tol=tol)


def compute_effective_rank(matrix: np.ndarray) -> float:
    """
    计算有效秩 (effective rank)
    
    有效秩 = exp(H(σ))，其中 H(σ) 是归一化奇异值的熵
    这是一个连续的秩度量，对噪声更鲁棒
    
    Args:
        matrix: 2D numpy array
    
    Returns:
        float: 有效秩
    """
    # 计算奇异值
    s = np.linalg.svd(matrix, compute_uv=False)
    
    # 归一化（概率分布）
    s_normalized = s / (s.sum() + 1e-10)
    
    # 去除零值
    s_normalized = s_normalized[s_normalized > 1e-10]
    
    # 计算熵
    entropy = -np.sum(s_normalized * np.log(s_normalized + 1e-10))
    
    # 有效秩
    effective_rank = np.exp(entropy)
    
    return effective_rank


def compute_singular_values(matrix: np.ndarray) -> np.ndarray:
    """计算奇异值"""
    return np.linalg.svd(matrix, compute_uv=False)


def analyze_single_sample(latents: np.ndarray, remove_last: bool = True) -> Dict:
    """
    分析单个样本的 latent token 矩阵
    
    Args:
        latents: shape (num_iterations, embedding_dim)
        remove_last: 是否去除最后一个 token
    
    Returns:
        Dict: 包含秩、有效秩、奇异值等信息
    """
    if remove_last:
        matrix = latents[:-1, :]  # 去除最后一个 token
    else:
        matrix = latents
    
    num_tokens, dim = matrix.shape
    
    # 计算秩
    rank = compute_matrix_rank(matrix)
    effective_rank = compute_effective_rank(matrix)
    
    # 计算奇异值
    singular_values = compute_singular_values(matrix)
    
    # 奇异值统计
    sv_max = singular_values.max()
    sv_min = singular_values.min()
    sv_mean = singular_values.mean()
    sv_std = singular_values.std()
    
    # 条件数
    condition_number = sv_max / (sv_min + 1e-10)
    
    # 能量分布（前 k 个奇异值占总能量的比例）
    total_energy = (singular_values ** 2).sum()
    energy_ratio_50 = (singular_values[:min(len(singular_values), num_tokens//2)] ** 2).sum() / (total_energy + 1e-10)
    energy_ratio_90 = 0.0
    cumsum = np.cumsum(singular_values ** 2)
    for i, c in enumerate(cumsum):
        if c / (total_energy + 1e-10) >= 0.9:
            energy_ratio_90 = i + 1
            break
    
    return {
        'num_tokens': num_tokens,
        'rank': rank,
        'effective_rank': effective_rank,
        'sv_max': sv_max,
        'sv_min': sv_min,
        'sv_mean': sv_mean,
        'sv_std': sv_std,
        'condition_number': condition_number,
        'energy_ratio_50': energy_ratio_50,
        'components_for_90_energy': energy_ratio_90,
        'singular_values': singular_values
    }


def analyze_latent_rank(latent_num: int, model: str = "codi", 
                        num_samples: int = None) -> Dict:
    """
    分析指定 latent 数量的矩阵秩
    
    Args:
        latent_num: latent token 数量
        model: 模型名称
        num_samples: 采样数量，None 表示使用全部样本
    
    Returns:
        Dict: 汇总统计信息
    """
    latents = load_latents(latent_num, model)
    if latents is None:
        return None
    
    n_samples = latents.shape[0]
    if num_samples is not None:
        n_samples = min(n_samples, num_samples)
        indices = np.random.choice(latents.shape[0], n_samples, replace=False)
        latents = latents[indices]
    
    # 对每个样本计算统计量
    all_stats = []
    for i in range(n_samples):
        stats = analyze_single_sample(latents[i], remove_last=True)
        all_stats.append(stats)
    
    # 聚合统计
    num_tokens = all_stats[0]['num_tokens']
    ranks = [s['rank'] for s in all_stats]
    effective_ranks = [s['effective_rank'] for s in all_stats]
    condition_numbers = [s['condition_number'] for s in all_stats]
    components_for_90 = [s['components_for_90_energy'] for s in all_stats]
    
    # 收集所有奇异值用于分析
    all_sv = np.array([s['singular_values'] for s in all_stats])
    
    result = {
        'latent_num': latent_num,
        'model': model,
        'num_samples': n_samples,
        'num_tokens': num_tokens,
        'rank_mean': np.mean(ranks),
        'rank_std': np.std(ranks),
        'rank_min': np.min(ranks),
        'rank_max': np.max(ranks),
        'effective_rank_mean': np.mean(effective_ranks),
        'effective_rank_std': np.std(effective_ranks),
        'condition_number_mean': np.mean(condition_numbers),
        'condition_number_std': np.std(condition_numbers),
        'components_for_90_energy_mean': np.mean(components_for_90),
        'singular_values_mean': all_sv.mean(axis=0),
        'singular_values_std': all_sv.std(axis=0),
    }
    
    return result


def print_results(results: List[Dict]):
    """打印结果表格"""
    print("\n" + "="*100)
    print("Latent Token 矩阵秩分析结果")
    print("="*100)
    
    print(f"\n{'Latent Num':<12} {'Model':<15} {'Tokens':<8} {'Rank Mean':<12} {'Rank Std':<10} "
          f"{'Eff Rank':<12} {'Cond Num':<15} {'90% Energy Components':<20}")
    print("-"*100)
    
    for r in results:
        if r is None:
            continue
        print(f"{r['latent_num']:<12} {r['model']:<15} {r['num_tokens']:<8} "
              f"{r['rank_mean']:<12.2f} {r['rank_std']:<10.2f} "
              f"{r['effective_rank_mean']:<12.2f} {r['condition_number_mean']:<15.2f} "
              f"{r['components_for_90_energy_mean']:<20.2f}")
    
    print("="*100)


def plot_rank_comparison(results: List[Dict], output_dir: str = None):
    """
    绘制不同 latent 数量的秩比较图
    """
    if output_dir is None:
        output_dir = BASE_DIR / "rank_analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 按模型分组
    models = list(set(r['model'] for r in results if r is not None))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#80d8cf', '#ff9f9f', '#ffdd93', '#99c8fe', '#989aca']
    
    for model_idx, model in enumerate(models):
        model_results = [r for r in results if r is not None and r['model'] == model]
        model_results.sort(key=lambda x: x['latent_num'])
        
        latent_nums = [r['latent_num'] for r in model_results]
        num_tokens = [r['num_tokens'] for r in model_results]
        ranks = [r['rank_mean'] for r in model_results]
        rank_stds = [r['rank_std'] for r in model_results]
        effective_ranks = [r['effective_rank_mean'] for r in model_results]
        
        color = colors[model_idx % len(colors)]
        
        # Plot 1: Rank vs Latent Num
        ax1 = axes[0, 0]
        ax1.errorbar(latent_nums, ranks, yerr=rank_stds, label=model, 
                    color=color, marker='o', capsize=3, linewidth=2)
        ax1.set_xlabel('Latent Number (Config)')
        ax1.set_ylabel('Matrix Rank')
        ax1.set_title('Matrix Rank vs Latent Number')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Effective Rank vs Latent Num
        ax2 = axes[0, 1]
        ax2.plot(latent_nums, effective_ranks, label=model, 
                color=color, marker='s', linewidth=2)
        ax2.set_xlabel('Latent Number (Config)')
        ax2.set_ylabel('Effective Rank')
        ax2.set_title('Effective Rank vs Latent Number')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rank / NumTokens ratio
        ax3 = axes[1, 0]
        rank_ratio = [r / t for r, t in zip(ranks, num_tokens)]
        ax3.plot(latent_nums, rank_ratio, label=model, 
                color=color, marker='^', linewidth=2)
        ax3.set_xlabel('Latent Number (Config)')
        ax3.set_ylabel('Rank / Num Tokens')
        ax3.set_title('Rank Ratio (Rank / Tokens) vs Latent Number')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rank vs Actual Num Tokens (去除最后一个)
        ax4 = axes[1, 1]
        ax4.plot(num_tokens, ranks, label=model, 
                color=color, marker='d', linewidth=2)
        ax4.plot([min(num_tokens), max(num_tokens)], 
                [min(num_tokens), max(num_tokens)], 
                'k--', alpha=0.5, label='y=x (full rank)')
        ax4.set_xlabel('Num Tokens (after removing last)')
        ax4.set_ylabel('Matrix Rank')
        ax4.set_title('Rank vs Actual Token Count')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / "rank_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def plot_singular_values(results: List[Dict], output_dir: str = None):
    """
    绘制奇异值分布对比图
    """
    if output_dir is None:
        output_dir = BASE_DIR / "rank_analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 只绘制第一个模型的不同 latent 数量对比
    model = results[0]['model'] if results else 'codi'
    model_results = [r for r in results if r is not None and r['model'] == model]
    model_results.sort(key=lambda x: x['latent_num'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_results)))
    
    # Plot 1: Singular values (normalized)
    ax1 = axes[0]
    for i, r in enumerate(model_results):
        sv = r['singular_values_mean']
        sv_normalized = sv / (sv.sum() + 1e-10)
        ax1.plot(range(1, len(sv)+1), sv_normalized, 
                label=f"latent_{r['latent_num']} (tokens={r['num_tokens']})", 
                color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Component Index')
    ax1.set_ylabel('Normalized Singular Value')
    ax1.set_title(f'Normalized Singular Value Distribution ({model})')
    ax1.legend(loc='upper right')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative energy
    ax2 = axes[1]
    for i, r in enumerate(model_results):
        sv = r['singular_values_mean']
        sv_sq = sv ** 2
        cumsum = np.cumsum(sv_sq) / (sv_sq.sum() + 1e-10)
        ax2.plot(range(1, len(cumsum)+1), cumsum, 
                label=f"latent_{r['latent_num']} (tokens={r['num_tokens']})", 
                color=colors[i], linewidth=2)
    
    ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% energy')
    ax2.axhline(y=0.99, color='orange', linestyle='--', alpha=0.5, label='99% energy')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Energy Ratio')
    ax2.set_title(f'Cumulative Energy Distribution ({model})')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / f"singular_values_{model}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Latent Token 矩阵秩分析')
    parser.add_argument('--latent_nums', type=int, nargs='+', default=[7, 17],
                        help='要分析的 latent 数量列表')
    parser.add_argument('--all', action='store_true',
                        help='分析所有可用的 latent 数量 (1-18)')
    parser.add_argument('--models', type=str, nargs='+', default=['codi'],
                        help='要分析的模型列表')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='采样数量，默认使用全部样本')
    parser.add_argument('--no_plot', action='store_true',
                        help='不生成图表')
    
    args = parser.parse_args()
    
    if args.all:
        latent_nums = list(range(1, 19))
    else:
        latent_nums = args.latent_nums
    
    print(f"Analyzing latent numbers: {latent_nums}")
    print(f"Models: {args.models}")
    
    # 分析
    all_results = []
    for latent_num in latent_nums:
        for model in args.models:
            result = analyze_latent_rank(latent_num, model, args.num_samples)
            if result is not None:
                all_results.append(result)
    
    # 打印结果
    print_results(all_results)
    
    # 绘图
    if not args.no_plot and all_results:
        plot_rank_comparison(all_results)
        plot_singular_values(all_results)
    
    return all_results


if __name__ == "__main__":
    main()
