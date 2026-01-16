#!/usr/bin/env python3
"""
测试结果分析和可视化脚本

数据结构：
- 每次运行 test.py 生成 N 个 batch（默认 11 个，代表不同配置）
- batch 字段每次运行都是 0-10（重复），需要用行号来确定 run_id
- 运行 M 次后，总共有 M * N 行
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ========== 配置 ==========
BATCHES_PER_RUN = 11  # 每次运行生成的 batch 数量

# ========== 字体配置 ==========
def setup_font():
    """配置字体"""
    chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'Source Han Sans SC']
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return True
    return False

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
setup_font()


class ResultsAnalyzer:
    def __init__(self, results_dir: str = "results", batches_per_run: int = BATCHES_PER_RUN):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.batches_per_run = batches_per_run
        
        self.metrics_files = {
            'accel': 'accel_gsm8k.jsonl',
            'action': 'action_gsm8k.jsonl',
            'geodesic': 'geodesic_gsm8k.jsonl',
            'radius': 'radius_gsm8k.jsonl',
        }
    
    def load_results(self) -> Dict[str, pd.DataFrame]:
        """加载所有结果文件，用行号计算 run_id 和 config_id"""
        results = {}
        for metric_name, filename in self.metrics_files.items():
            filepath = self.results_dir / filename
            if filepath.exists():
                data = []
                with open(filepath, 'r') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                if data:
                    df = pd.DataFrame(data)
                    # 用行号计算 run_id 和 config_id（而不是 batch 字段）
                    df['row_idx'] = range(len(df))
                    df['config_id'] = df['row_idx'] % self.batches_per_run
                    df['run_id'] = df['row_idx'] // self.batches_per_run
                    results[metric_name] = df
                else:
                    results[metric_name] = pd.DataFrame()
            else:
                results[metric_name] = pd.DataFrame()
        return results
    
    def get_run_count(self, results: Dict[str, pd.DataFrame]) -> int:
        """获取运行次数 M"""
        for df in results.values():
            if not df.empty:
                return df['run_id'].max() + 1
        return 0
    
    def export_csv(self, results: Dict[str, pd.DataFrame]):
        """导出 CSV 文件"""
        print("\n导出 CSV 文件:")
        
        # 1. 每个指标的详细 CSV
        for metric_name, df in results.items():
            if df.empty:
                continue
            
            csv_path = self.results_dir / f'{metric_name}_detailed.csv'
            export_cols = ['run_id', 'config_id'] + [c for c in df.columns 
                          if c not in ['batch', 'run_id', 'config_id', 'row_idx', 'batch_size', 'num_latent_steps']]
            df[export_cols].to_csv(csv_path, index=False, float_format='%.6f')
            print(f"  ✓ {csv_path}")
        
        # 2. 汇总 CSV
        self._export_summary_csv(results)
        
        # 3. 跨运行对比 CSV
        self._export_cross_run_csv(results)
    
    def _export_summary_csv(self, results: Dict[str, pd.DataFrame]):
        """导出汇总 CSV"""
        n_runs = self.get_run_count(results)
        
        summary_rows = []
        for run_id in range(n_runs):
            for config_id in range(self.batches_per_run):
                row = {'run_id': run_id, 'config_id': config_id}
                
                for metric_name, df in results.items():
                    if df.empty:
                        continue
                    
                    mask = (df['run_id'] == run_id) & (df['config_id'] == config_id)
                    if mask.any():
                        record = df[mask].iloc[0]
                        
                        if metric_name == 'accel':
                            row['accel_mean'] = record.get('accel_mean')
                            row['accel_max'] = record.get('accel_max')
                        elif metric_name == 'action':
                            row['action_total'] = record.get('total_mean')
                            row['action_kinetic'] = record.get('kinetic_mean')
                        elif metric_name == 'geodesic':
                            row['geodesic_mean'] = record.get('deviation_mean')
                        elif metric_name == 'radius':
                            row['radius_mean'] = record.get('radius_mean')
                
                summary_rows.append(row)
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            csv_path = self.results_dir / 'summary_all.csv'
            summary_df.to_csv(csv_path, index=False, float_format='%.6f')
            print(f"  ✓ {csv_path} (完整汇总)")
    
    def _export_cross_run_csv(self, results: Dict[str, pd.DataFrame]):
        """导出跨运行对比 CSV"""
        n_runs = self.get_run_count(results)
        
        for metric_name, df in results.items():
            if df.empty:
                continue
            
            # 选择主要指标列
            if metric_name == 'accel':
                value_col = 'accel_mean'
            elif metric_name == 'action':
                value_col = 'total_mean'
            elif metric_name == 'geodesic':
                value_col = 'deviation_mean'
            elif metric_name == 'radius':
                value_col = 'radius_mean'
            else:
                continue
            
            # 创建 pivot table
            pivot_df = df.pivot_table(
                index='config_id', 
                columns='run_id', 
                values=value_col,
                aggfunc='first'
            )
            pivot_df.columns = [f'run_{i}' for i in pivot_df.columns]
            pivot_df = pivot_df.reset_index()
            
            # 添加统计列
            run_cols = [c for c in pivot_df.columns if c.startswith('run_')]
            if len(run_cols) > 1:
                pivot_df['mean_across_runs'] = pivot_df[run_cols].mean(axis=1)
                pivot_df['std_across_runs'] = pivot_df[run_cols].std(axis=1)
            
            csv_path = self.results_dir / f'{metric_name}_cross_run.csv'
            pivot_df.to_csv(csv_path, index=False, float_format='%.6f')
            print(f"  ✓ {csv_path} (跨运行对比)")
    
    def print_summary(self, results: Dict[str, pd.DataFrame]):
        """打印汇总信息"""
        n_runs = self.get_run_count(results)
        total_rows = max(len(df) for df in results.values() if not df.empty)
        
        print(f"\n{'='*100}")
        print(f"结果分析 - {self.results_dir.absolute()}")
        print(f"{'='*100}")
        print(f"  运行次数 (M): {n_runs}")
        print(f"  每次配置数 (N): {self.batches_per_run}")
        print(f"  总行数: {total_rows}")
        
        # 打印每次运行的摘要
        for run_id in range(n_runs):
            print(f"\n【Run {run_id}】")
            for metric_name, df in results.items():
                if df.empty:
                    continue
                
                run_df = df[df['run_id'] == run_id]
                if run_df.empty:
                    continue
                
                if metric_name == 'accel':
                    col = 'accel_mean'
                elif metric_name == 'action':
                    col = 'total_mean'
                elif metric_name == 'geodesic':
                    col = 'deviation_mean'
                elif metric_name == 'radius':
                    col = 'radius_mean'
                else:
                    continue
                
                print(f"  {metric_name:10s}: mean={run_df[col].mean():.4f}, "
                      f"std={run_df[col].std():.4f}, "
                      f"min={run_df[col].min():.4f}, max={run_df[col].max():.4f}")
        
        # 打印跨运行的配置对比
        if n_runs > 1:
            print(f"\n{'='*100}")
            print("【跨运行配置对比】")
            print(f"{'='*100}")
            
            for metric_name, df in results.items():
                if df.empty:
                    continue
                
                if metric_name == 'accel':
                    col = 'accel_mean'
                elif metric_name == 'action':
                    col = 'total_mean'
                elif metric_name == 'geodesic':
                    col = 'deviation_mean'
                elif metric_name == 'radius':
                    col = 'radius_mean'
                else:
                    continue
                
                print(f"\n[{metric_name.upper()}]")
                for config_id in range(self.batches_per_run):
                    config_df = df[df['config_id'] == config_id].sort_values('run_id')
                    if not config_df.empty:
                        values = config_df[col].values
                        print(f"  Config {config_id:2d}: " + 
                              ", ".join([f"run{i}={v:.4f}" for i, v in enumerate(values)]) +
                              f" | mean={values.mean():.4f}, std={values.std():.4f}")
    
    def plot_within_run(self, results: Dict[str, pd.DataFrame]):
        """绘制单次运行内的配置对比"""
        n_runs = self.get_run_count(results)
        
        for run_id in range(n_runs):
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Run {run_id} - Config Comparison', fontsize=16, fontweight='bold')
            
            # Acceleration
            if not results['accel'].empty:
                ax = axes[0, 0]
                df = results['accel'][results['accel']['run_id'] == run_id].sort_values('config_id')
                if not df.empty:
                    ax.bar(df['config_id'], df['accel_mean'], alpha=0.8, color='steelblue')
                    if 'accel_std' in df.columns:
                        ax.errorbar(df['config_id'], df['accel_mean'], yerr=df['accel_std'], 
                                   fmt='none', color='black', capsize=3)
                    ax.set_xlabel('Config ID')
                    ax.set_ylabel('Acceleration Mean')
                    ax.set_title('Acceleration by Config')
                    ax.set_xticks(range(self.batches_per_run))
                    ax.grid(True, alpha=0.3, axis='y')
            
            # Action
            if not results['action'].empty:
                ax = axes[0, 1]
                df = results['action'][results['action']['run_id'] == run_id].sort_values('config_id')
                if not df.empty:
                    x = np.arange(len(df))
                    width = 0.25
                    ax.bar(x - width, df['kinetic_mean'], width, label='Kinetic', alpha=0.8)
                    ax.bar(x, df['potential_mean'], width, label='Potential', alpha=0.8)
                    ax.bar(x + width, df['total_mean'], width, label='Total', alpha=0.8)
                    ax.set_xlabel('Config ID')
                    ax.set_ylabel('Action Value')
                    ax.set_title('Action Components by Config')
                    ax.set_xticks(x)
                    ax.set_xticklabels(df['config_id'])
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
            
            # Geodesic
            if not results['geodesic'].empty:
                ax = axes[1, 0]
                df = results['geodesic'][results['geodesic']['run_id'] == run_id].sort_values('config_id')
                if not df.empty:
                    ax.bar(df['config_id'], df['deviation_mean'], alpha=0.8, color='forestgreen')
                    ax.set_xlabel('Config ID')
                    ax.set_ylabel('Geodesic Deviation')
                    ax.set_title('Geodesic Deviation by Config')
                    ax.set_xticks(range(self.batches_per_run))
                    ax.grid(True, alpha=0.3, axis='y')
            
            # Radius
            if not results['radius'].empty:
                ax = axes[1, 1]
                df = results['radius'][results['radius']['run_id'] == run_id].sort_values('config_id')
                if not df.empty:
                    ax.bar(df['config_id'], df['radius_mean'], alpha=0.8, color='coral')
                    if 'radius_threshold' in df.columns:
                        ax.axhline(y=df['radius_threshold'].iloc[0], color='r', 
                                  linestyle='--', label='Threshold', linewidth=2)
                        ax.legend()
                    ax.set_xlabel('Config ID')
                    ax.set_ylabel('Radius')
                    ax.set_title('Radius by Config')
                    ax.set_xticks(range(self.batches_per_run))
                    ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            output_path = self.results_dir / f'within_run_{run_id}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Run {run_id} 内部对比: {output_path}")
            plt.close()
    
    def plot_cross_run_lines(self, results: Dict[str, pd.DataFrame]):
        """绘制跨运行折线图（每条线=一次运行）"""
        n_runs = self.get_run_count(results)
        
        if n_runs < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Cross-Run Comparison (M={n_runs} runs)', fontsize=16, fontweight='bold')
        
        metrics_info = [
            ('accel', 'accel_mean', 'Acceleration Mean', axes[0, 0]),
            ('action', 'total_mean', 'Action Total', axes[0, 1]),
            ('geodesic', 'deviation_mean', 'Geodesic Deviation', axes[1, 0]),
            ('radius', 'radius_mean', 'Radius Mean', axes[1, 1]),
        ]
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_runs))
        
        for metric_name, col, title, ax in metrics_info:
            if results[metric_name].empty:
                continue
            
            df = results[metric_name]
            
            for run_id in range(n_runs):
                run_df = df[df['run_id'] == run_id].sort_values('config_id')
                ax.plot(run_df['config_id'], run_df[col], 'o-', 
                       label=f'Run {run_id}', color=colors[run_id],
                       linewidth=2, markersize=6, alpha=0.8)
            
            ax.set_xlabel('Config ID')
            ax.set_ylabel(title)
            ax.set_title(f'{title} - Cross Run')
            ax.set_xticks(range(self.batches_per_run))
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.results_dir / 'cross_run_lines.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 跨运行折线图: {output_path}")
        plt.close()
    
    def plot_cross_run_by_config(self, results: Dict[str, pd.DataFrame]):
        """绘制每个配置在各次运行中的变化（条形图）"""
        n_runs = self.get_run_count(results)
        
        if n_runs < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Config Performance Across {n_runs} Runs', fontsize=16, fontweight='bold')
        
        metrics_info = [
            ('accel', 'accel_mean', 'Acceleration Mean', axes[0, 0]),
            ('action', 'total_mean', 'Action Total', axes[0, 1]),
            ('geodesic', 'deviation_mean', 'Geodesic Deviation', axes[1, 0]),
            ('radius', 'radius_mean', 'Radius Mean', axes[1, 1]),
        ]
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_runs))
        
        for metric_name, col, title, ax in metrics_info:
            if results[metric_name].empty:
                continue
            
            df = results[metric_name]
            
            x = np.arange(self.batches_per_run)
            width = 0.8 / n_runs
            
            for run_id in range(n_runs):
                run_df = df[df['run_id'] == run_id].sort_values('config_id')
                offset = (run_id - n_runs/2 + 0.5) * width
                ax.bar(x + offset, run_df[col], width, 
                      label=f'Run {run_id}', color=colors[run_id], alpha=0.8)
            
            ax.set_xlabel('Config ID')
            ax.set_ylabel(title)
            ax.set_title(f'{title} - All Runs')
            ax.set_xticks(x)
            ax.set_xticklabels(range(self.batches_per_run))
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.results_dir / 'cross_run_bars.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 跨运行条形图: {output_path}")
        plt.close()
    
    def plot_heatmaps(self, results: Dict[str, pd.DataFrame]):
        """绘制热力图"""
        n_runs = self.get_run_count(results)
        
        if n_runs < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Metrics Heatmap (Config x Run)', fontsize=16, fontweight='bold')
        
        metrics_info = [
            ('accel', 'accel_mean', 'Acceleration Mean', axes[0, 0]),
            ('action', 'total_mean', 'Action Total', axes[0, 1]),
            ('geodesic', 'deviation_mean', 'Geodesic Deviation', axes[1, 0]),
            ('radius', 'radius_mean', 'Radius Mean', axes[1, 1]),
        ]
        
        for metric_name, col, title, ax in metrics_info:
            if results[metric_name].empty:
                continue
            
            df = results[metric_name]
            pivot = df.pivot_table(index='config_id', columns='run_id', values=col, aggfunc='first')
            
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                       cbar_kws={'label': title})
            ax.set_xlabel('Run ID')
            ax.set_ylabel('Config ID')
            ax.set_title(title)
        
        plt.tight_layout()
        output_path = self.results_dir / 'heatmap_config_run.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 热力图: {output_path}")
        plt.close()
    
    def plot_boxplots(self, results: Dict[str, pd.DataFrame]):
        """绘制箱线图"""
        n_runs = self.get_run_count(results)
        
        if n_runs < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Config Stability Across Runs', fontsize=16, fontweight='bold')
        
        metrics_info = [
            ('accel', 'accel_mean', 'Acceleration Mean', axes[0, 0]),
            ('action', 'total_mean', 'Action Total', axes[0, 1]),
            ('geodesic', 'deviation_mean', 'Geodesic Deviation', axes[1, 0]),
            ('radius', 'radius_mean', 'Radius Mean', axes[1, 1]),
        ]
        
        for metric_name, col, title, ax in metrics_info:
            if results[metric_name].empty:
                continue
            
            df = results[metric_name]
            box_data = [df[df['config_id'] == i][col].values for i in range(self.batches_per_run)]
            
            bp = ax.boxplot(box_data, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Config ID')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Distribution per Config')
            ax.set_xticklabels(range(self.batches_per_run))
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.results_dir / 'boxplot_stability.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 箱线图: {output_path}")
        plt.close()
    
    def plot_run_summary(self, results: Dict[str, pd.DataFrame]):
        """绘制每次运行的汇总统计对比"""
        n_runs = self.get_run_count(results)
        
        if n_runs < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Run-Level Summary Statistics', fontsize=16, fontweight='bold')
        
        metrics_info = [
            ('accel', 'accel_mean', 'Acceleration', axes[0, 0]),
            ('action', 'total_mean', 'Action', axes[0, 1]),
            ('geodesic', 'deviation_mean', 'Geodesic', axes[1, 0]),
            ('radius', 'radius_mean', 'Radius', axes[1, 1]),
        ]
        
        for metric_name, col, title, ax in metrics_info:
            if results[metric_name].empty:
                continue
            
            df = results[metric_name]
            
            # 计算每次运行的统计量
            run_stats = df.groupby('run_id')[col].agg(['mean', 'std', 'min', 'max']).reset_index()
            
            x = run_stats['run_id']
            ax.bar(x, run_stats['mean'], yerr=run_stats['std'], 
                  alpha=0.8, color='steelblue', capsize=5, label='Mean ± Std')
            ax.scatter(x, run_stats['min'], marker='v', color='red', s=50, label='Min', zorder=5)
            ax.scatter(x, run_stats['max'], marker='^', color='green', s=50, label='Max', zorder=5)
            
            ax.set_xlabel('Run ID')
            ax.set_ylabel(title)
            ax.set_title(f'{title} - Run Statistics')
            ax.set_xticks(x)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.results_dir / 'run_summary_stats.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 运行汇总图: {output_path}")
        plt.close()
    
    def run(self):
        """执行完整分析流程"""
        results = self.load_results()
        
        if not any(not df.empty for df in results.values()):
            print("⚠ 没有找到任何结果文件！")
            return
        
        n_runs = self.get_run_count(results)
        
        # 打印摘要
        self.print_summary(results)
        
        # 导出 CSV
        self.export_csv(results)
        
        # 绘制图表
        print("\n绘制图表:")
        
        # 1. 每次运行内的配置对比
        self.plot_within_run(results)
        
        # 2. 跨运行对比图表
        if n_runs > 1:
            self.plot_cross_run_lines(results)      # 折线图
            self.plot_cross_run_by_config(results)  # 条形图
            self.plot_heatmaps(results)             # 热力图
            self.plot_boxplots(results)             # 箱线图
            self.plot_run_summary(results)          # 运行汇总统计
        
        print(f"\n{'='*60}")
        print("✓ 分析完成！")
        print(f"  - CSV 文件: {self.results_dir}/*.csv")
        print(f"  - 图表文件: {self.results_dir}/*.png")
        print(f"{'='*60}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='分析和可视化测试结果')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='结果文件所在目录 (默认: results)')
    parser.add_argument('--batches_per_run', '-n', type=int, default=BATCHES_PER_RUN,
                       help=f'每次运行的 batch 数量 (默认: {BATCHES_PER_RUN})')
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(
        results_dir=args.results_dir,
        batches_per_run=args.batches_per_run
    )
    analyzer.run()


if __name__ == '__main__':
    main()
