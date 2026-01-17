#!/usr/bin/env python3
"""
结果分析和可视化脚本 (新版)

支持新的结果目录结构：
results/
├── models/{model}/{dataset}/run_{i}/
├── datasets/{dataset}/all_models.csv
└── summary/
    ├── all_results.csv
    └── comparison_matrix.csv

用法：
    python analyze_results.py                    # 分析所有结果
    python analyze_results.py --plot             # 生成可视化图表
    python analyze_results.py --model euclidean  # 只分析指定模型
    python analyze_results.py --dataset gsm8k    # 只分析指定数据集
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Warning: matplotlib/seaborn not installed, plotting disabled")


# ============================================================
# 配置
# ============================================================
CODI_RESULT_DIR = os.environ.get("CODI_RESULT_DIR", "./results")

DATASET_NAMES = {
    "gsm8k": "GSM8K",
    "gsm-hard": "GSM-Hard",
    "svamp": "SVAMP",
    "multi-arith": "MultiArith",
    "commonsense": "CommonSenseQA",
}


# ============================================================
# 字体配置
# ============================================================
def setup_plot():
    """配置绘图"""
    if not HAS_PLOT:
        return
    
    chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            break
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    sns.set_style("whitegrid")


# ============================================================
# 结果加载
# ============================================================
class ResultsLoader:
    """结果加载器"""
    
    def __init__(self, results_dir: str = None):
        self.results_dir = Path(results_dir or CODI_RESULT_DIR)
        self.models_dir = self.results_dir / "models"
        self.datasets_dir = self.results_dir / "datasets"
        self.summary_dir = self.results_dir / "summary"
    
    def load_all_results(self) -> pd.DataFrame:
        """加载所有结果"""
        all_results_path = self.summary_dir / "all_results.csv"
        
        if all_results_path.exists():
            return pd.read_csv(all_results_path)
        
        # 如果汇总文件不存在，从各个 metrics.json 重建
        print("汇总文件不存在，从各个结果文件重建...")
        return self._rebuild_all_results()
    
    def _rebuild_all_results(self) -> pd.DataFrame:
        """从各个结果文件重建汇总"""
        results = []
        
        if not self.models_dir.exists():
            print(f"模型结果目录不存在: {self.models_dir}")
            return pd.DataFrame()
        
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                dataset_name = dataset_dir.name
                
                for run_dir in dataset_dir.iterdir():
                    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                        continue
                    
                    metrics_path = run_dir / "metrics.json"
                    if metrics_path.exists():
                        with open(metrics_path) as f:
                            metrics = json.load(f)
                            results.append(metrics)
        
        if results:
            df = pd.DataFrame(results)
            # 保存汇总
            self.summary_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.summary_dir / "all_results.csv", index=False)
            return df
        
        return pd.DataFrame()
    
    def get_models(self) -> List[str]:
        """获取所有模型名称"""
        if not self.models_dir.exists():
            return []
        return [d.name for d in self.models_dir.iterdir() if d.is_dir()]
    
    def get_datasets(self) -> List[str]:
        """获取所有数据集名称"""
        df = self.load_all_results()
        if df.empty:
            return []
        return df['dataset'].unique().tolist()


# ============================================================
# 结果分析
# ============================================================
class ResultsAnalyzer:
    """结果分析器"""
    
    def __init__(self, results_dir: str = None):
        self.loader = ResultsLoader(results_dir)
        self.results_dir = Path(results_dir or CODI_RESULT_DIR)
    
    def print_summary(self, model: str = None, dataset: str = None):
        """打印结果摘要"""
        df = self.loader.load_all_results()
        
        if df.empty:
            print("没有找到任何结果")
            return
        
        # 筛选
        if model:
            df = df[df['model'] == model]
        if dataset:
            df = df[df['dataset'] == dataset]
        
        if df.empty:
            print("筛选后没有结果")
            return
        
        print("\n" + "="*100)
        print("结果摘要")
        print("="*100)
        print(f"总测试数: {len(df)}")
        print(f"模型数: {df['model'].nunique()}")
        print(f"数据集数: {df['dataset'].nunique()}")
        print()
        
        # 生成对比矩阵
        self._print_comparison_matrix(df)
        
        # 按模型打印详细结果
        self._print_model_details(df)
        
        # 按数据集打印排名
        self._print_dataset_rankings(df)
    
    def _print_comparison_matrix(self, df: pd.DataFrame):
        """打印对比矩阵"""
        print("\n" + "-"*100)
        print("模型 × 数据集 准确率矩阵 (平均值)")
        print("-"*100)
        
        pivot = df.pivot_table(
            values='accuracy',
            index='model',
            columns='dataset',
            aggfunc='mean'
        )
        
        # 添加平均值列
        pivot['AVG'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('AVG', ascending=False)
        
        # 格式化为百分比
        formatted = pivot.applymap(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-")
        print(formatted.to_string())
        
        # 保存到 CSV
        pivot.to_csv(self.results_dir / "summary" / "comparison_matrix_latest.csv", float_format='%.4f')
    
    def _print_model_details(self, df: pd.DataFrame):
        """打印每个模型的详细结果"""
        print("\n" + "-"*100)
        print("各模型详细结果")
        print("-"*100)
        
        for model in sorted(df['model'].unique()):
            model_df = df[df['model'] == model]
            
            print(f"\n【{model}】")
            
            for dataset in sorted(model_df['dataset'].unique()):
                dataset_df = model_df[model_df['dataset'] == dataset]
                
                acc_mean = dataset_df['accuracy'].mean()
                acc_std = dataset_df['accuracy'].std()
                n_runs = len(dataset_df)
                
                std_str = f"±{acc_std*100:.2f}%" if n_runs > 1 else ""
                display_name = DATASET_NAMES.get(dataset, dataset)
                
                print(f"  {display_name:15s}: {acc_mean*100:6.2f}% {std_str:10s} ({n_runs} runs)")
    
    def _print_dataset_rankings(self, df: pd.DataFrame):
        """打印每个数据集的模型排名"""
        print("\n" + "-"*100)
        print("各数据集模型排名")
        print("-"*100)
        
        for dataset in sorted(df['dataset'].unique()):
            dataset_df = df[df['dataset'] == dataset]
            
            ranking = dataset_df.groupby('model')['accuracy'].mean().sort_values(ascending=False)
            
            display_name = DATASET_NAMES.get(dataset, dataset)
            print(f"\n【{display_name}】")
            
            for i, (model, acc) in enumerate(ranking.items(), 1):
                medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
                print(f"  {medal} {i}. {model:30s}: {acc*100:.2f}%")
    
    def generate_plots(self, output_dir: str = None):
        """生成可视化图表"""
        if not HAS_PLOT:
            print("matplotlib/seaborn 未安装，无法生成图表")
            return
        
        setup_plot()
        
        df = self.loader.load_all_results()
        if df.empty:
            print("没有数据可绘制")
            return
        
        output_dir = Path(output_dir or (self.results_dir / "plots"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n生成可视化图表...")
        
        # 1. 🏆 冠军展示图 - 最直观的模型排名
        self._plot_champion_ranking(df, output_dir)
        
        # 2. 🎯 雷达图 - 多维度对比
        self._plot_radar_chart(df, output_dir)
        
        # 3. 🔥 增强热力图 - 带排名标注
        self._plot_enhanced_heatmap(df, output_dir)
        
        # 4. 📊 分组条形图 - 每个数据集的模型对比
        self._plot_grouped_bars(df, output_dir)
        
        # 5. 🏅 胜负矩阵 - 模型两两对比
        self._plot_win_matrix(df, output_dir)
        
        # 6. 📈 综合评分图
        self._plot_comprehensive_score(df, output_dir)
        
        # 7. 原有图表
        self._plot_heatmap(df, output_dir)
        self._plot_bar_comparison(df, output_dir)
        self._plot_per_dataset(df, output_dir)
        
        print(f"\n✅ 所有图表已保存到: {output_dir}")
    
    def _plot_champion_ranking(self, df: pd.DataFrame, output_dir: Path):
        """🏆 冠军排名图 - 最直观展示谁最强"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 计算每个模型的平均准确率
        model_avg = df.groupby('model')['accuracy'].mean().sort_values(ascending=False) * 100
        
        # 颜色：金银铜 + 其他
        colors = []
        for i in range(len(model_avg)):
            if i == 0:
                colors.append('#FFD700')  # 金
            elif i == 1:
                colors.append('#C0C0C0')  # 银
            elif i == 2:
                colors.append('#CD7F32')  # 铜
            else:
                colors.append('#4A90D9')  # 蓝
        
        bars = ax.barh(range(len(model_avg)), model_avg.values, color=colors, 
                      edgecolor='black', linewidth=1.5)
        
        # 添加奖牌图标和数值
        for i, (bar, val) in enumerate(zip(bars, model_avg.values)):
            # 排名标记
            rank_mark = "[1st]" if i == 0 else ("[2nd]" if i == 1 else ("[3rd]" if i == 2 else ""))
            
            # 在条形内部显示模型名和准确率
            model_name = model_avg.index[i].replace("trained_", "")
            ax.text(val - 2, bar.get_y() + bar.get_height()/2,
                   f'{rank_mark} {model_name}: {val:.1f}%',
                   va='center', ha='right', fontsize=14, fontweight='bold',
                   color='white' if val > 30 else 'black')
            
            # 与第一名的差距
            if i > 0:
                gap = model_avg.values[0] - val
                ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                       f'(-{gap:.1f}%)',
                       va='center', ha='left', fontsize=11, color='red', alpha=0.8)
        
        ax.set_yticks([])
        ax.set_xlabel('Average Accuracy (%)', fontsize=14)
        ax.set_xlim(0, 100)
        ax.set_title('*** MODEL CHAMPIONSHIP RANKING ***\n(Average across all datasets)', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # 添加背景网格
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # 添加冠军高亮框
        ax.axhspan(-0.5, 0.5, color='gold', alpha=0.1)
        
        plt.tight_layout()
        plt.savefig(output_dir / "champion_ranking.png", dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"  [OK] Champion Ranking: {output_dir / 'champion_ranking.png'}")
    
    def _plot_radar_chart(self, df: pd.DataFrame, output_dir: Path):
        """🎯 雷达图 - 多维度展示各模型能力"""
        from math import pi
        
        # 计算每个模型在每个数据集上的准确率
        pivot = df.pivot_table(
            values='accuracy',
            index='model',
            columns='dataset',
            aggfunc='mean'
        ) * 100
        
        # 按平均准确率排序，只展示前 6 个模型
        pivot['avg'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('avg', ascending=False).head(6)
        pivot = pivot.drop('avg', axis=1)
        
        categories = [DATASET_NAMES.get(c, c) for c in pivot.columns]
        N = len(categories)
        
        # 计算角度
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
        
        # 颜色
        colors = plt.cm.Set1(np.linspace(0, 1, len(pivot)))
        
        for idx, (model, row) in enumerate(pivot.iterrows()):
            values = row.values.tolist()
            values += values[:1]  # 闭合
            
            model_short = model.replace("trained_", "")
            ax.plot(angles, values, 'o-', linewidth=2.5, label=model_short, 
                   color=colors[idx], markersize=8)
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        # 设置角度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12, fontweight='bold')
        
        # 设置径向范围
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=10)
        
        ax.set_title('MODEL CAPABILITY RADAR CHART\n', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_dir / "radar_chart.png", dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"  [OK] Radar Chart: {output_dir / 'radar_chart.png'}")
    
    def _plot_enhanced_heatmap(self, df: pd.DataFrame, output_dir: Path):
        """🔥 增强热力图 - 带排名标注"""
        pivot = df.pivot_table(
            values='accuracy',
            index='model',
            columns='dataset',
            aggfunc='mean'
        ) * 100
        
        # 按平均值排序
        pivot['AVG'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('AVG', ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.8)))
        
        # 绘制热力图
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=50,
            ax=ax,
            linewidths=2,
            linecolor='white',
            cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.8},
            annot_kws={'size': 12, 'weight': 'bold'}
        )
        
        # 添加每列的排名标注（在单元格内添加排名）
        for col_idx, col in enumerate(pivot.columns):
            col_ranks = pivot[col].rank(ascending=False)
            for row_idx, (model, rank) in enumerate(col_ranks.items()):
                if rank <= 3:
                    rank_symbol = "*" if rank == 1 else ("**" if rank == 2 else "***")
                    # 在单元格角落添加排名标记
                    ax.text(col_idx + 0.9, row_idx + 0.15, rank_symbol, 
                           fontsize=10, ha='right', va='top')
        
        # 美化
        ax.set_title('MODEL PERFORMANCE HEATMAP WITH RANKINGS\n(* = 1st, ** = 2nd, *** = 3rd)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        
        # 简化模型名
        yticks = [label.get_text().replace("trained_", "") for label in ax.get_yticklabels()]
        ax.set_yticklabels(yticks, fontsize=11)
        
        # 简化数据集名
        xticks = [DATASET_NAMES.get(label.get_text(), label.get_text()) 
                 for label in ax.get_xticklabels()]
        ax.set_xticklabels(xticks, fontsize=11, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / "enhanced_heatmap.png", dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"  [OK] Enhanced Heatmap: {output_dir / 'enhanced_heatmap.png'}")
    
    def _plot_grouped_bars(self, df: pd.DataFrame, output_dir: Path):
        """📊 分组条形图 - 每个数据集的模型对比"""
        pivot = df.pivot_table(
            values='accuracy',
            index='model',
            columns='dataset',
            aggfunc='mean'
        ) * 100
        
        # 按平均值排序
        pivot['avg'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('avg', ascending=False)
        pivot = pivot.drop('avg', axis=1)
        
        n_models = len(pivot)
        n_datasets = len(pivot.columns)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(n_datasets)
        width = 0.8 / n_models
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))
        
        for i, (model, row) in enumerate(pivot.iterrows()):
            offset = (i - n_models/2 + 0.5) * width
            model_short = model.replace("trained_", "")
            bars = ax.bar(x + offset, row.values, width, label=model_short, 
                         color=colors[i], edgecolor='black', linewidth=0.5)
            
            # 标注最高分
            for j, (val, dataset) in enumerate(zip(row.values, pivot.columns)):
                # 检查是否是该数据集的最高分
                if val == pivot[dataset].max():
                    ax.text(j + offset, val + 1, 'BEST', ha='center', fontsize=8, 
                           fontweight='bold', color='red')
                # 显示具体数值
                ax.text(
                    j + offset,
                    val + 1.5,
                    f"{val:.1f}%",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='black'
                )
        
        ax.set_xlabel('Dataset', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_title('MODEL COMPARISON BY DATASET\n(BEST = Top performer on that dataset)', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_NAMES.get(c, c) for c in pivot.columns], fontsize=12)
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right', fontsize=10, ncol=2)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "grouped_bars.png", dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"  [OK] Grouped Bars: {output_dir / 'grouped_bars.png'}")
    
    def _plot_win_matrix(self, df: pd.DataFrame, output_dir: Path):
        """🏅 胜负矩阵 - 模型两两对比谁赢得更多"""
        pivot = df.pivot_table(
            values='accuracy',
            index='model',
            columns='dataset',
            aggfunc='mean'
        )
        
        models = pivot.index.tolist()
        n = len(models)
        
        # 计算胜负矩阵
        win_matrix = np.zeros((n, n))
        
        for i, model_a in enumerate(models):
            for j, model_b in enumerate(models):
                if i != j:
                    # 统计 model_a 赢 model_b 的数据集数量
                    wins = (pivot.loc[model_a] > pivot.loc[model_b]).sum()
                    win_matrix[i, j] = wins
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制热力图
        mask = np.eye(n, dtype=bool)
        sns.heatmap(
            win_matrix,
            annot=True,
            fmt='.0f',
            cmap='RdYlGn',
            center=len(pivot.columns)/2,
            ax=ax,
            mask=mask,
            linewidths=2,
            linecolor='white',
            cbar_kws={'label': 'Number of Datasets Won'},
            annot_kws={'size': 14, 'weight': 'bold'}
        )
        
        # 设置标签
        short_names = [m.replace("trained_", "") for m in models]
        ax.set_xticklabels(short_names, fontsize=11, rotation=45, ha='right')
        ax.set_yticklabels(short_names, fontsize=11, rotation=0)
        ax.set_xlabel('Opponent', fontsize=14)
        ax.set_ylabel('Model', fontsize=14)
        ax.set_title(f'WIN MATRIX: Row beats Column on N datasets\n(Total datasets: {len(pivot.columns)})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "win_matrix.png", dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"  [OK] Win Matrix: {output_dir / 'win_matrix.png'}")
    
    def _plot_comprehensive_score(self, df: pd.DataFrame, output_dir: Path):
        """📈 综合评分图 - 多维度加权评分"""
        pivot = df.pivot_table(
            values='accuracy',
            index='model',
            columns='dataset',
            aggfunc='mean'
        ) * 100
        
        # 计算多个指标
        scores = pd.DataFrame(index=pivot.index)
        
        # 1. 平均准确率
        scores['avg_acc'] = pivot.mean(axis=1)
        
        # 2. 最低准确率（最差表现）
        scores['min_acc'] = pivot.min(axis=1)
        
        # 3. 一致性（标准差的反面）
        scores['consistency'] = 100 - pivot.std(axis=1)
        
        # 4. 数据集冠军数
        scores['championships'] = (pivot == pivot.max()).sum(axis=1) * 10
        
        # 综合得分
        scores['total_score'] = (
            scores['avg_acc'] * 0.5 +
            scores['min_acc'] * 0.2 +
            scores['consistency'] * 0.2 +
            scores['championships'] * 0.1
        )
        
        scores = scores.sort_values('total_score', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：综合得分条形图
        ax1 = axes[0]
        colors = ['#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32' if i == 2 else '#4A90D9' 
                 for i in range(len(scores))]
        
        bars = ax1.barh(range(len(scores)), scores['total_score'], color=colors,
                       edgecolor='black', linewidth=1.5)
        
        for i, (idx, row) in enumerate(scores.iterrows()):
            model_short = idx.replace("trained_", "")
            rank_mark = "[1st]" if i == 0 else ("[2nd]" if i == 1 else ("[3rd]" if i == 2 else ""))
            ax1.text(row['total_score'] + 1, i, 
                    f'{rank_mark} {model_short}: {row["total_score"]:.1f}',
                    va='center', fontsize=12, fontweight='bold')
        
        ax1.set_yticks([])
        ax1.set_xlabel('Comprehensive Score', fontsize=14)
        ax1.set_title('COMPREHENSIVE RANKING\n(Avg×0.5 + Min×0.2 + Consistency×0.2 + Champs×0.1)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 100)
        ax1.xaxis.grid(True, linestyle='--', alpha=0.7)
        
        # 右图：各维度雷达图（简化版柱状对比）
        ax2 = axes[1]
        
        # 取前3名的各维度数据
        top3 = scores.head(3)
        metrics = ['avg_acc', 'min_acc', 'consistency']
        metric_names = ['Average\nAccuracy', 'Worst\nDataset', 'Consistency']
        
        x = np.arange(len(metrics))
        width = 0.25
        
        rank_labels = ['1st', '2nd', '3rd']
        for i, (model, row) in enumerate(top3.iterrows()):
            model_short = model.replace("trained_", "")
            offset = (i - 1) * width
            color = ['#FFD700', '#C0C0C0', '#CD7F32'][i]
            ax2.bar(x + offset, [row[m] for m in metrics], width, 
                   label=f'[{rank_labels[i]}] {model_short}', color=color, edgecolor='black')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_names, fontsize=12)
        ax2.set_ylabel('Score', fontsize=14)
        ax2.set_title('TOP 3 MODELS BREAKDOWN', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=11)
        ax2.set_ylim(0, 100)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle('COMPREHENSIVE MODEL EVALUATION', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "comprehensive_score.png", dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"  [OK] Comprehensive Score: {output_dir / 'comprehensive_score.png'}")
    
    def _plot_heatmap(self, df: pd.DataFrame, output_dir: Path):
        """绘制热力图"""
        pivot = df.pivot_table(
            values='accuracy',
            index='model',
            columns='dataset',
            aggfunc='mean'
        ) * 100  # 转为百分比
        
        pivot = pivot.sort_values(pivot.columns[0], ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=50,
            ax=ax,
            cbar_kws={'label': 'Accuracy (%)'}
        )
        
        ax.set_title('Model × Dataset Accuracy Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Model')
        
        plt.tight_layout()
        plt.savefig(output_dir / "heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {output_dir / 'heatmap.png'}")
    
    def _plot_bar_comparison(self, df: pd.DataFrame, output_dir: Path):
        """绘制条形图对比"""
        # 计算每个模型的平均准确率
        model_avg = df.groupby('model')['accuracy'].mean().sort_values(ascending=True) * 100
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(model_avg) * 0.4)))
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(model_avg)))
        bars = ax.barh(model_avg.index, model_avg.values, color=colors)
        
        # 添加数值标签
        for bar, val in zip(bars, model_avg.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}%', va='center', fontsize=10)
        
        ax.set_xlabel('Average Accuracy (%)')
        ax.set_title('Model Performance Comparison (All Datasets)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {output_dir / 'model_comparison.png'}")
    
    def _plot_per_dataset(self, df: pd.DataFrame, output_dir: Path):
        """为每个数据集绘制模型对比图"""
        datasets = df['dataset'].unique()
        n_datasets = len(datasets)
        
        fig, axes = plt.subplots(1, n_datasets, figsize=(4*n_datasets, 6), sharey=True)
        
        if n_datasets == 1:
            axes = [axes]
        
        for ax, dataset in zip(axes, sorted(datasets)):
            dataset_df = df[df['dataset'] == dataset]
            model_acc = dataset_df.groupby('model')['accuracy'].agg(['mean', 'std']) * 100
            model_acc = model_acc.sort_values('mean', ascending=True)
            
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(model_acc)))
            
            ax.barh(model_acc.index, model_acc['mean'], 
                   xerr=model_acc['std'].fillna(0), 
                   color=colors, capsize=3)
            
            display_name = DATASET_NAMES.get(dataset, dataset)
            ax.set_title(display_name, fontsize=12, fontweight='bold')
            ax.set_xlim(0, 100)
            ax.set_xlabel('Accuracy (%)')
        
        plt.suptitle('Model Performance by Dataset', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "per_dataset_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {output_dir / 'per_dataset_comparison.png'}")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="分析和可视化测试结果")
    parser.add_argument("--results-dir", "-r", default=CODI_RESULT_DIR, help="结果目录")
    parser.add_argument("--model", "-m", default=None, help="只分析指定模型")
    parser.add_argument("--dataset", "-d", default=None, help="只分析指定数据集")
    parser.add_argument("--plot", "-p", action="store_true", help="生成可视化图表")
    parser.add_argument("--output", "-o", default=None, help="图表输出目录")
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results_dir)
    
    # 打印摘要
    analyzer.print_summary(model=args.model, dataset=args.dataset)
    
    # 生成图表
    if args.plot:
        analyzer.generate_plots(args.output)


if __name__ == "__main__":
    main()
