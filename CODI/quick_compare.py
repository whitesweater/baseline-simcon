#!/usr/bin/env python3
"""
快速查看测试结果对比

数据结构：
- 每次运行生成 N 个 batch（默认 11 个，代表不同配置）
- batch 字段每次运行都是 0-10（重复），用行号确定 run_id
"""

import json
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

BATCHES_PER_RUN = 11


def load_results(results_dir: str = "results", batches_per_run: int = BATCHES_PER_RUN) -> Dict[str, pd.DataFrame]:
    """加载所有结果文件"""
    results_path = Path(results_dir)
    results = {}
    
    files = {
        'accel': 'accel_gsm8k.jsonl',
        'action': 'action_gsm8k.jsonl',
        'geodesic': 'geodesic_gsm8k.jsonl',
        'radius': 'radius_gsm8k.jsonl',
    }
    
    for name, filename in files.items():
        filepath = results_path / filename
        data = []
        if filepath.exists():
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        
        if data:
            df = pd.DataFrame(data)
            # 用行号计算 run_id 和 config_id
            df['row_idx'] = range(len(df))
            df['config_id'] = df['row_idx'] % batches_per_run
            df['run_id'] = df['row_idx'] // batches_per_run
            results[name] = df
        else:
            results[name] = pd.DataFrame()
    
    return results


def print_run_summary(results: Dict[str, pd.DataFrame], run_id: int):
    """打印单次运行的摘要"""
    print(f"\n【Run {run_id}】各配置对比:")
    
    for metric_name, df in results.items():
        if df.empty:
            continue
        
        run_df = df[df['run_id'] == run_id].sort_values('config_id')
        if run_df.empty:
            continue
        
        print(f"\n  [{metric_name.upper()}]")
        
        if metric_name == 'accel':
            cols = ['config_id', 'accel_mean', 'accel_max', 'accel_std']
        elif metric_name == 'action':
            cols = ['config_id', 'total_mean', 'kinetic_mean', 'potential_mean']
        elif metric_name == 'geodesic':
            cols = ['config_id', 'deviation_mean', 'deviation_max']
        elif metric_name == 'radius':
            cols = ['config_id', 'radius_mean', 'radius_max']
        else:
            cols = ['config_id'] + [c for c in run_df.columns if c not in ['batch', 'run_id', 'config_id', 'row_idx', 'batch_size', 'num_latent_steps']]
        
        cols = [c for c in cols if c in run_df.columns]
        print(run_df[cols].round(4).to_string(index=False))


def print_cross_run_summary(results: Dict[str, pd.DataFrame], batches_per_run: int):
    """打印跨运行的摘要"""
    n_runs = max(df['run_id'].max() + 1 for df in results.values() if not df.empty)
    
    if n_runs < 2:
        return
    
    print(f"\n{'='*100}")
    print(f"【跨运行对比】- 共 {n_runs} 次运行")
    print(f"{'='*100}")
    
    metrics_cols = {
        'accel': 'accel_mean',
        'action': 'total_mean',
        'geodesic': 'deviation_mean',
        'radius': 'radius_mean',
    }
    
    for metric_name, col in metrics_cols.items():
        if results[metric_name].empty:
            continue
        
        df = results[metric_name]
        
        print(f"\n[{metric_name.upper()}] - {col}")
        
        rows = []
        for config_id in range(batches_per_run):
            row = {'config': config_id}
            config_df = df[df['config_id'] == config_id].sort_values('run_id')
            
            for i, (_, record) in enumerate(config_df.iterrows()):
                row[f'run_{i}'] = record[col]
            
            # 计算跨运行统计
            values = [row.get(f'run_{i}') for i in range(n_runs) if f'run_{i}' in row]
            if len(values) > 1:
                row['mean'] = np.mean(values)
                row['std'] = np.std(values)
            
            rows.append(row)
        
        summary_df = pd.DataFrame(rows)
        print(summary_df.round(4).to_string(index=False))


def export_csv(results: Dict[str, pd.DataFrame], results_dir: str = "results"):
    """导出 CSV"""
    results_path = Path(results_dir)
    
    for metric_name, df in results.items():
        if df.empty:
            continue
        
        csv_path = results_path / f'{metric_name}_detailed.csv'
        export_cols = ['run_id', 'config_id'] + [c for c in df.columns 
                      if c not in ['batch', 'run_id', 'config_id', 'row_idx', 'batch_size', 'num_latent_steps']]
        df[export_cols].to_csv(csv_path, index=False, float_format='%.6f')
        print(f"✓ {csv_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='快速查看测试结果对比')
    parser.add_argument('--results_dir', type=str, default='results1',
                       help='结果目录 (默认: results)')
    parser.add_argument('--batches_per_run', '-n', type=int, default=BATCHES_PER_RUN,
                       help=f'每次运行的 batch 数 (默认: {BATCHES_PER_RUN})')
    parser.add_argument('--run', '-r', type=int, default=None,
                       help='只显示指定运行的结果')
    parser.add_argument('--cross', action='store_true',
                       help='显示跨运行对比')
    parser.add_argument('--csv', action='store_true',
                       help='导出 CSV')
    args = parser.parse_args()
    
    results = load_results(args.results_dir, args.batches_per_run)
    
    if not any(not df.empty for df in results.values()):
        print("⚠ 没有找到任何结果文件！")
        return
    
    n_runs = max(df['run_id'].max() + 1 for df in results.values() if not df.empty)
    total_rows = max(len(df) for df in results.values() if not df.empty)
    
    print(f"\n{'='*100}")
    print(f"结果概览: {n_runs} 次运行 × {args.batches_per_run} 个配置 = {total_rows} 行")
    print(f"{'='*100}")
    
    if args.run is not None:
        print_run_summary(results, args.run)
    else:
        for run_id in range(n_runs):
            print_run_summary(results, run_id)
    
    if args.cross or n_runs > 1:
        print_cross_run_summary(results, args.batches_per_run)
    
    if args.csv:
        print("\n导出 CSV:")
        export_csv(results, args.results_dir)
    
    print(f"\n提示:")
    print(f"  --run N     只查看第 N 次运行")
    print(f"  --cross     显示跨运行对比")
    print(f"  --csv       导出 CSV 文件")
    print(f"  visualize_results.py  生成完整图表\n")


if __name__ == '__main__':
    main()
