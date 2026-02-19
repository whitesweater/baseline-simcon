#!/usr/bin/env python3
"""
Latent Token 矩阵秩分析脚本

分析三个不同 latent token 数量的模型:
- latent_7: 7 tokens (from codi model)
- latent_17: 16 tokens (from 16long model)  
- latent_33: 32 tokens (from 32long model)

计算：
1. 矩阵秩 (Matrix Rank) - 去除最后一个 token 后
2. 有效秩 (Effective Rank) - 基于奇异值熵的连续秩度量
3. 奇异值分布

运行: python analyze_rank_quick.py
"""

import json
import numpy as np
import os
import sys

def compute_effective_rank(matrix):
    """计算有效秩 = exp(entropy of normalized singular values)"""
    s = np.linalg.svd(matrix, compute_uv=False)
    s_norm = s / (s.sum() + 1e-10)
    s_norm = s_norm[s_norm > 1e-10]
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
    return np.exp(entropy)

def analyze_file(name, path, n_samples=50):
    """分析单个文件的矩阵秩"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"Path: {path[:80]}...")
    
    if not os.path.exists(path):
        print("  FILE NOT FOUND!")
        return None
    
    # 加载数据
    print("  Loading data...")
    with open(path) as f:
        data = json.load(f)
    
    latents = np.array(data['latents'], dtype=np.float32)
    num_samples, num_iters, dim = latents.shape
    print(f"  Shape: {num_samples} samples x {num_iters} iterations x {dim} dim")
    
    # 去除最后一个 token
    latents_no_last = latents[:, :-1, :]
    num_tokens = num_iters - 1
    print(f"  Tokens after removing last: {num_tokens}")
    
    del data  # 释放内存
    
    # 随机采样
    np.random.seed(42)
    n = min(num_samples, n_samples)
    indices = np.random.choice(num_samples, n, replace=False)
    
    # 计算秩
    print(f"  Computing ranks for {n} samples...")
    ranks = []
    eff_ranks = []
    
    for idx, i in enumerate(indices):
        matrix = latents_no_last[i]  # (num_tokens, dim)
        
        # 标准秩
        rank = np.linalg.matrix_rank(matrix)
        ranks.append(rank)
        
        # 有效秩
        eff_rank = compute_effective_rank(matrix)
        eff_ranks.append(eff_rank)
        
        if (idx + 1) % 10 == 0:
            print(f"    Progress: {idx + 1}/{n}")
    
    # 统计
    result = {
        'name': name,
        'num_tokens': num_tokens,
        'rank_mean': np.mean(ranks),
        'rank_std': np.std(ranks),
        'rank_min': int(np.min(ranks)),
        'rank_max': int(np.max(ranks)),
        'eff_rank_mean': np.mean(eff_ranks),
        'eff_rank_std': np.std(eff_ranks),
        'ratio': np.mean(ranks) / num_tokens
    }
    
    print(f"\n  === Results ===")
    print(f"  Number of tokens: {num_tokens}")
    print(f"  Rank: {result['rank_mean']:.2f} ± {result['rank_std']:.2f} (min={result['rank_min']}, max={result['rank_max']})")
    print(f"  Effective Rank: {result['eff_rank_mean']:.2f} ± {result['eff_rank_std']:.2f}")
    print(f"  Rank / Tokens: {result['ratio']:.4f}")
    
    return result


def main():
    # 文件路径
    files = {
        "latent_7 (codi)": "/data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_7/models/codi/gsm8k/run_0/latents.json",
        "latent_17 (16long)": "/data/yhao/baseline/CODI/results/16long/models/decoder-trajectory-euclidean-16long_Llama-3.2-1B-Instruct_ep_12_lr_0.0008_seed_11_checkpoint-35988/gsm8k/run_0/latents.json",
        "latent_33 (32long)": "/data/yhao/baseline/CODI/results/32long/models/decoder-trajectory-euclidean-32long_Llama-3.2-1B-Instruct_ep_12_lr_0.0008_seed_11_checkpoint-32989/gsm8k/run_0/latents.json",
    }
    
    print("="*80)
    print("Latent Token Matrix Rank Analysis")
    print("="*80)
    
    # 分析每个文件
    results = []
    for name, path in files.items():
        result = analyze_file(name, path, n_samples=50)
        if result:
            results.append(result)
    
    # 汇总表
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<22} {'Tokens':<8} {'Rank':<16} {'Eff Rank':<16} {'Rank/Tokens'}")
    print("-"*80)
    for r in results:
        rank_str = f"{r['rank_mean']:.0f}±{r['rank_std']:.1f}"
        eff_str = f"{r['eff_rank_mean']:.2f}±{r['eff_rank_std']:.2f}"
        print(f"{r['name']:<22} {r['num_tokens']:<8} {rank_str:<16} {eff_str:<16} {r['ratio']:.4f}")
    print("="*80)
    
    # 分析结论
    print("\n" + "="*80)
    print("ANALYSIS CONCLUSIONS")
    print("="*80)
    if results:
        full_rank = [r for r in results if r['rank_mean'] == r['num_tokens']]
        if len(full_rank) == len(results):
            print("✓ All models achieve FULL RANK in the latent token matrices")
            print("  This means each latent token contributes linearly independent information")
        else:
            for r in results:
                if r['rank_mean'] < r['num_tokens']:
                    print(f"✗ {r['name']}: rank ({r['rank_mean']:.1f}) < tokens ({r['num_tokens']})")
                    print(f"  Some latent tokens may be linearly dependent")
        
        print("\n  Effective Rank comparison:")
        for r in results:
            print(f"  - {r['name']}: {r['eff_rank_mean']:.2f} / {r['num_tokens']} = {r['eff_rank_mean']/r['num_tokens']:.2%} of tokens")
    
    return results


if __name__ == "__main__":
    results = main()
