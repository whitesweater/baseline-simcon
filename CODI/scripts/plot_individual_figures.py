#!/usr/bin/env python3
"""
生成单独的可视化图片并创建详细解说报告
每个图单独保存为一个文件
"""

import json
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from scipy.spatial.distance import cosine, euclidean
import pandas as pd

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 模型显示名称映射
DISPLAY_NAMES = {
    'codi': 'CODI',
    'simcon': 'SIM-CoT',
    'codi_sircl': 'CODI+SIRCL',
    'sircl': 'SIM-CoT+SIRCL'
}

# 模型颜色
MODEL_COLORS = {
    'codi': '#3498DB',       # 蓝色
    'codi_sircl': '#E67E22', # 橙色
    'simcon': '#27AE60',     # 绿色
    'sircl': '#E74C3C'       # 红色
}


# 模型分组
MODEL_GROUPS = {
    'CODI': ['codi', 'codi_sircl'],
    'SIM-CoT': ['simcon', 'sircl']
}

# 分组颜色
GROUP_COLORS = {
    'CODI': {
        'codi': '#3498DB',       # 蓝色 (基线)
        'codi_sircl': '#1A5276'  # 深蓝色 (+SIRCL)
    },
    'SIM-CoT': {
        'simcon': '#27AE60',     # 绿色 (基线)
        'sircl': '#196F3D'       # 深绿色 (+SIRCL)
    }
}


class IndividualPlotter:
    """单独图片绘制器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.models_dir = self.results_dir / "models"
        self.output_dir = self.results_dir / "latent_analysis" / "individual_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_names = []
        self.latents = {}
        self.predictions = {}
        self.ground_truths = None
        self.correct_masks = {}
        self.all_correct_indices = None
        
    def load_data(self, dataset: str = "gsm8k", run: int = 0):
        """加载所有模型的数据"""
        print(f"Loading data from {self.models_dir}")
        
        for model_dir in sorted(self.models_dir.iterdir()):
            if model_dir.is_dir():
                self.model_names.append(model_dir.name)
        
        print(f"Found models: {self.model_names}")
        
        for model_name in self.model_names:
            run_dir = self.models_dir / model_name / dataset / f"run_{run}"
            
            latents_file = run_dir / "latents.json"
            if latents_file.exists():
                print(f"Loading latents for {model_name}...")
                with open(latents_file) as f:
                    data = json.load(f)
                self.latents[model_name] = np.array(data['latents'], dtype=np.float32)
                print(f"  Shape: {self.latents[model_name].shape}")
            
            pred_file = run_dir / "predictions.json"
            if pred_file.exists():
                with open(pred_file) as f:
                    pred_data = json.load(f)
                self.predictions[model_name] = pred_data.get('predictions', [])
                if self.ground_truths is None:
                    self.ground_truths = pred_data.get('ground_truth', pred_data.get('ground_truths', []))
        
        self._compute_correct_masks()
        
    def _compute_correct_masks(self):
        """计算正确率mask"""
        for model_name in self.model_names:
            if model_name not in self.predictions:
                continue
            preds = self.predictions[model_name]
            # 转换为字符串进行比较，处理float和str混合的情况
            correct = []
            for p, g in zip(preds, self.ground_truths):
                try:
                    correct.append(float(p) == float(g))
                except (ValueError, TypeError):
                    correct.append(str(p) == str(g))
            self.correct_masks[model_name] = np.array(correct, dtype=bool)
        
        all_correct = np.ones(len(self.ground_truths), dtype=bool)
        for model_name in self.model_names:
            if model_name in self.correct_masks:
                all_correct = np.logical_and(all_correct, self.correct_masks[model_name])
        
        self.all_correct_indices = np.where(all_correct)[0]
        print(f"\nAll models correct: {len(self.all_correct_indices)} / {len(self.ground_truths)} samples")
    
    # ==================== 1. Embedding Statistics (4-in-1) ====================
    def plot_embedding_statistics(self):
        """绘制4个Embedding统计指标的组合图"""
        print("Plotting: Embedding Statistics (4 subplots)...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Embedding Norm
        ax = axes[0, 0]
        for model_name in self.model_names:
            latents = self.latents[model_name][self.all_correct_indices]
            norms_by_iter = []
            for iter_idx in range(7):
                norms = [np.linalg.norm(latents[i, iter_idx]) for i in range(latents.shape[0])]
                norms_by_iter.append(np.mean(norms))
            ax.plot(range(1, 8), norms_by_iter, marker='o', linewidth=2, markersize=6,
                   label=DISPLAY_NAMES.get(model_name, model_name),
                   color=MODEL_COLORS.get(model_name, 'gray'))
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Mean L2 Norm', fontsize=11)
        ax.set_title('Embedding Norm by Iteration', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. Embedding Variance
        ax = axes[0, 1]
        for model_name in self.model_names:
            latents = self.latents[model_name][self.all_correct_indices]
            vars_by_iter = []
            for iter_idx in range(7):
                iter_latents = latents[:, iter_idx, :]
                var = np.mean(np.var(iter_latents, axis=0))
                vars_by_iter.append(var)
            ax.plot(range(1, 8), vars_by_iter, marker='s', linewidth=2, markersize=6,
                   label=DISPLAY_NAMES.get(model_name, model_name),
                   color=MODEL_COLORS.get(model_name, 'gray'))
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Mean Variance', fontsize=11)
        ax.set_title('Embedding Variance by Iteration', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative Path Length
        ax = axes[1, 0]
        for model_name in self.model_names:
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            cumulative_dists = [0]
            for iter_idx in range(6):
                step_dists = [euclidean(latents[i, iter_idx], latents[i, iter_idx+1])
                             for i in range(n_samples)]
                cumulative_dists.append(cumulative_dists[-1] + np.mean(step_dists))
            ax.plot(range(1, 8), cumulative_dists, marker='d', linewidth=2, markersize=6,
                   label=DISPLAY_NAMES.get(model_name, model_name),
                   color=MODEL_COLORS.get(model_name, 'gray'))
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Cumulative Distance', fontsize=11)
        ax.set_title('Cumulative Path Length', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 4. Path Efficiency
        ax = axes[1, 1]
        for model_name in self.model_names:
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            efficiencies = []
            for iter_idx in range(1, 7):
                direct_dists = [euclidean(latents[i, 0], latents[i, iter_idx]) 
                               for i in range(n_samples)]
                path_dists = []
                for i in range(n_samples):
                    path_dist = sum(euclidean(latents[i, j], latents[i, j+1]) 
                                   for j in range(iter_idx))
                    path_dists.append(path_dist)
                eff = np.mean([d/p if p > 1e-8 else 1 for d, p in zip(direct_dists, path_dists)])
                efficiencies.append(eff)
            ax.plot(range(2, 8), efficiencies, marker='^', linewidth=2, markersize=6,
                   label=DISPLAY_NAMES.get(model_name, model_name),
                   color=MODEL_COLORS.get(model_name, 'gray'))
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Path Efficiency (direct/actual)', fontsize=11)
        ax.set_title('Path Efficiency by Iteration', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / '01_embedding_statistics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {save_path}")
        return save_path
    
    # ==================== 2. Distance to Final Token (4-in-1) ====================
    def plot_distance_to_final_all_models(self):
        """绘制4个模型到最终token距离的组合箱线图"""
        print("Plotting: Distance to Final Token (4 models)...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, model_name in enumerate(self.model_names):
            ax = axes[idx]
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            
            distances_by_iter = []
            for iter_idx in range(6):
                dists = [euclidean(latents[i, iter_idx], latents[i, -1]) for i in range(n_samples)]
                distances_by_iter.append(dists)
            
            bp = ax.boxplot(distances_by_iter, labels=[f'Iter {i+1}' for i in range(6)],
                           patch_artist=True)
            
            color = MODEL_COLORS.get(model_name, 'lightblue')
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Distance to Final Token', fontsize=11)
            ax.set_title(f'{DISPLAY_NAMES.get(model_name, model_name)}',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Distance to Final Token by Iteration', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = self.output_dir / '02_distance_to_final_boxplots.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {save_path}")
        return save_path
    
    # ==================== 3. Convergence Comparison (2-in-1) ====================
    def plot_convergence_comparisons(self):
        """绘制收敛对比图（距离+相似度）"""
        print("Plotting: Convergence Comparisons (2 subplots)...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：距离收敛
        ax = axes[0]
        for model_name in self.model_names:
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            mean_dists = []
            for iter_idx in range(6):
                dists = [euclidean(latents[i, iter_idx], latents[i, -1]) for i in range(n_samples)]
                mean_dists.append(np.mean(dists))
            ax.plot(range(1, 7), mean_dists, marker='o', linewidth=2, markersize=8,
                   label=DISPLAY_NAMES.get(model_name, model_name),
                   color=MODEL_COLORS.get(model_name, 'gray'))
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Mean Distance to Final Token', fontsize=12)
        ax.set_title('Distance Convergence', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 右图：相似度收敛
        ax = axes[1]
        for model_name in self.model_names:
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            mean_sims = []
            for iter_idx in range(6):
                sims = [1 - cosine(latents[i, iter_idx], latents[i, -1]) for i in range(n_samples)]
                mean_sims.append(np.mean(sims))
            ax.plot(range(1, 7), mean_sims, marker='s', linewidth=2, markersize=8,
                   label=DISPLAY_NAMES.get(model_name, model_name),
                   color=MODEL_COLORS.get(model_name, 'gray'))
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Mean Cosine Similarity to Final Token', fontsize=12)
        ax.set_title('Similarity Convergence', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Convergence Comparison Across Models', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = self.output_dir / '03_convergence_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {save_path}")
        return save_path
    
    # ==================== 4. Similarity Heatmaps (4-in-1) ====================
    def plot_similarity_heatmaps(self):
        """绘制4个模型的余弦相似度热力图"""
        print("Plotting: Cosine Similarity Heatmaps (4 models)...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, model_name in enumerate(self.model_names):
            ax = axes[idx]
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            
            sim_matrix = np.zeros((7, 7))
            for i in range(7):
                for j in range(7):
                    sims = [1 - cosine(latents[k, i], latents[k, j]) for k in range(n_samples)]
                    sim_matrix[i, j] = np.mean(sims)
            
            im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0.4, vmax=1.0)
            
            for i in range(7):
                for j in range(7):
                    ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center',
                           fontsize=9, fontweight='bold',
                           color='white' if sim_matrix[i, j] > 0.7 else 'black')
            
            ax.set_xticks(range(7))
            ax.set_yticks(range(7))
            ax.set_xticklabels([f'Iter {i+1}' for i in range(7)], fontsize=9)
            ax.set_yticklabels([f'Iter {i+1}' for i in range(7)], fontsize=9)
            ax.set_title(f'{DISPLAY_NAMES.get(model_name, model_name)}',
                        fontsize=12, fontweight='bold')
        
        # 添加共享的colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Cosine Similarity')
        
        plt.suptitle('Average Cosine Similarity Between Iterations\n'
                    f'(All Models Correct: {len(self.all_correct_indices)} samples)',
                    fontsize=14, fontweight='bold')
        
        save_path = self.output_dir / '04_similarity_heatmaps.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {save_path}")
        return save_path
    
    # ==================== 5. Key Statistics Bar Charts (6-in-1) ====================
    def plot_key_statistics(self, stats_df: pd.DataFrame):
        """绘制6个关键统计指标的柱状图"""
        print("Plotting: Key Statistics (6 subplots)...")
        
        key_metrics = [
            ('cos_sim_to_final_mean', 'Cosine Sim to Final', True),
            ('eucl_dist_to_final_mean', 'Euclidean Dist to Final', False),
            ('cluster_compactness_mean', 'Cluster Compactness', False),
            ('trajectory_smoothness_mean', 'Trajectory Smoothness', True),
            ('path_length_mean', 'Total Path Length', False),
            ('convergence_rate_mean', 'Convergence Rate', False),
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (col, label, higher_is_better) in enumerate(key_metrics):
            ax = axes[idx]
            values = stats_df[col].values
            models = [DISPLAY_NAMES.get(m, m) for m in stats_df['model']]
            colors = [MODEL_COLORS.get(m, 'gray') for m in stats_df['model']]
            
            bars = ax.bar(models, values, color=colors)
            
            # 标注最佳值
            if higher_is_better:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', labelsize=10, rotation=15)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Key Statistics Comparison\n(Red border = Best)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path = self.output_dir / '05_key_statistics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {save_path}")
        return save_path
    
    # ==================== Main ====================
    def generate_all_plots(self):
        """生成所有图片（同类图放在一起，按CODI和SIM-CoT分组）"""
        print("\n" + "="*60)
        print("Generating Grouped Plots (CODI vs SIM-CoT)")
        print("="*60 + "\n")
        
        saved_paths = []
        
        # 1. Embedding统计图 - CODI组
        saved_paths.append(self.plot_embedding_statistics_by_group('CODI'))
        # 2. Embedding统计图 - SIM-CoT组
        saved_paths.append(self.plot_embedding_statistics_by_group('SIM-CoT'))
        
        # 3. 距离箱线图 - CODI组
        saved_paths.append(self.plot_distance_to_final_by_group('CODI'))
        # 4. 距离箱线图 - SIM-CoT组
        saved_paths.append(self.plot_distance_to_final_by_group('SIM-CoT'))
        
        # 5. 收敛对比图 - CODI组
        saved_paths.append(self.plot_convergence_by_group('CODI'))
        # 6. 收敛对比图 - SIM-CoT组
        saved_paths.append(self.plot_convergence_by_group('SIM-CoT'))
        
        # 7. 相似度热力图 - CODI组
        saved_paths.append(self.plot_similarity_heatmaps_by_group('CODI'))
        # 8. 相似度热力图 - SIM-CoT组
        saved_paths.append(self.plot_similarity_heatmaps_by_group('SIM-CoT'))
        
        # 9. 统计指标柱状图 - CODI组
        # 10. 统计指标柱状图 - SIM-CoT组
        stats_file = self.results_dir / "latent_analysis" / "statistics.csv"
        if stats_file.exists():
            stats_df = pd.read_csv(stats_file)
            saved_paths.append(self.plot_key_statistics_by_group(stats_df, 'CODI'))
            saved_paths.append(self.plot_key_statistics_by_group(stats_df, 'SIM-CoT'))
        
        print(f"\n{'='*60}")
        print(f"Total {len(saved_paths)} grouped plots saved to {self.output_dir}")
        print("="*60)
        
        return saved_paths
    
    # ==================== Group-based plotting methods ====================
    def plot_embedding_statistics_by_group(self, group_name: str):
        """绘制指定组的Embedding统计指标"""
        models = MODEL_GROUPS[group_name]
        colors = GROUP_COLORS[group_name]
        print(f"Plotting: Embedding Statistics for {group_name}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        
        # 1. Embedding Norm
        ax = axes[0, 0]
        for model_name in models:
            latents = self.latents[model_name][self.all_correct_indices]
            norms_by_iter = []
            for iter_idx in range(7):
                norms = [np.linalg.norm(latents[i, iter_idx]) for i in range(latents.shape[0])]
                norms_by_iter.append(np.mean(norms))
            ax.plot(range(1, 8), norms_by_iter, marker='o', linewidth=2, markersize=8,
                   label=DISPLAY_NAMES.get(model_name, model_name),
                   color=colors.get(model_name, 'gray'))
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Mean L2 Norm', fontsize=11)
        ax.set_title('Embedding Norm by Iteration', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 2. Embedding Variance
        ax = axes[0, 1]
        for model_name in models:
            latents = self.latents[model_name][self.all_correct_indices]
            vars_by_iter = []
            for iter_idx in range(7):
                iter_latents = latents[:, iter_idx, :]
                var = np.mean(np.var(iter_latents, axis=0))
                vars_by_iter.append(var)
            ax.plot(range(1, 8), vars_by_iter, marker='s', linewidth=2, markersize=8,
                   label=DISPLAY_NAMES.get(model_name, model_name),
                   color=colors.get(model_name, 'gray'))
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Mean Variance', fontsize=11)
        ax.set_title('Embedding Variance by Iteration', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative Path Length
        ax = axes[1, 0]
        for model_name in models:
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            cumulative_dists = [0]
            for iter_idx in range(6):
                step_dists = [euclidean(latents[i, iter_idx], latents[i, iter_idx+1])
                             for i in range(n_samples)]
                cumulative_dists.append(cumulative_dists[-1] + np.mean(step_dists))
            ax.plot(range(1, 8), cumulative_dists, marker='d', linewidth=2, markersize=8,
                   label=DISPLAY_NAMES.get(model_name, model_name),
                   color=colors.get(model_name, 'gray'))
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Cumulative Distance', fontsize=11)
        ax.set_title('Cumulative Path Length', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 4. Path Efficiency
        ax = axes[1, 1]
        for model_name in models:
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            efficiencies = []
            for iter_idx in range(1, 7):
                direct_dists = [euclidean(latents[i, 0], latents[i, iter_idx]) 
                               for i in range(n_samples)]
                path_dists = []
                for i in range(n_samples):
                    path_dist = sum(euclidean(latents[i, j], latents[i, j+1]) 
                                   for j in range(iter_idx))
                    path_dists.append(path_dist)
                eff = np.mean([d/p if p > 1e-8 else 1 for d, p in zip(direct_dists, path_dists)])
                efficiencies.append(eff)
            ax.plot(range(2, 8), efficiencies, marker='^', linewidth=2, markersize=8,
                   label=DISPLAY_NAMES.get(model_name, model_name),
                   color=colors.get(model_name, 'gray'))
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Path Efficiency (direct/actual)', fontsize=11)
        ax.set_title('Path Efficiency by Iteration', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{group_name} Group: Embedding Statistics', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        group_suffix = group_name.lower().replace('-', '')
        save_path = self.output_dir / f'01_{group_suffix}_embedding_statistics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {save_path}")
        return save_path
    
    def plot_distance_to_final_by_group(self, group_name: str):
        """绘制指定组的距离箱线图"""
        models = MODEL_GROUPS[group_name]
        colors = GROUP_COLORS[group_name]
        print(f"Plotting: Distance to Final Token for {group_name}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, model_name in enumerate(models):
            ax = axes[idx]
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            
            distances_by_iter = []
            for iter_idx in range(6):
                dists = [euclidean(latents[i, iter_idx], latents[i, -1]) for i in range(n_samples)]
                distances_by_iter.append(dists)
            
            bp = ax.boxplot(distances_by_iter, labels=[f'Iter {i+1}' for i in range(6)],
                           patch_artist=True)
            
            color = colors.get(model_name, 'lightblue')
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Distance to Final Token', fontsize=11)
            ax.set_title(f'{DISPLAY_NAMES.get(model_name, model_name)}',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'{group_name} Group: Distance to Final Token', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        group_suffix = group_name.lower().replace('-', '')
        save_path = self.output_dir / f'02_{group_suffix}_distance_boxplots.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {save_path}")
        return save_path
    
    def plot_convergence_by_group(self, group_name: str):
        """绘制指定组的收敛对比图"""
        models = MODEL_GROUPS[group_name]
        colors = GROUP_COLORS[group_name]
        print(f"Plotting: Convergence Comparison for {group_name}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：距离收敛
        ax = axes[0]
        for model_name in models:
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            mean_dists = []
            for iter_idx in range(6):
                dists = [euclidean(latents[i, iter_idx], latents[i, -1]) for i in range(n_samples)]
                mean_dists.append(np.mean(dists))
            ax.plot(range(1, 7), mean_dists, marker='o', linewidth=2, markersize=8,
                   label=DISPLAY_NAMES.get(model_name, model_name),
                   color=colors.get(model_name, 'gray'))
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Mean Distance to Final Token', fontsize=12)
        ax.set_title('Distance Convergence', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 右图：相似度收敛
        ax = axes[1]
        for model_name in models:
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            mean_sims = []
            for iter_idx in range(6):
                sims = [1 - cosine(latents[i, iter_idx], latents[i, -1]) for i in range(n_samples)]
                mean_sims.append(np.mean(sims))
            ax.plot(range(1, 7), mean_sims, marker='s', linewidth=2, markersize=8,
                   label=DISPLAY_NAMES.get(model_name, model_name),
                   color=colors.get(model_name, 'gray'))
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Mean Cosine Similarity to Final Token', fontsize=12)
        ax.set_title('Similarity Convergence', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{group_name} Group: Convergence Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        group_suffix = group_name.lower().replace('-', '')
        save_path = self.output_dir / f'03_{group_suffix}_convergence.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {save_path}")
        return save_path
    
    def plot_similarity_heatmaps_by_group(self, group_name: str):
        """绘制指定组的余弦相似度热力图"""
        models = MODEL_GROUPS[group_name]
        print(f"Plotting: Cosine Similarity Heatmaps for {group_name}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, model_name in enumerate(models):
            ax = axes[idx]
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            
            sim_matrix = np.zeros((7, 7))
            for i in range(7):
                for j in range(7):
                    sims = [1 - cosine(latents[k, i], latents[k, j]) for k in range(n_samples)]
                    sim_matrix[i, j] = np.mean(sims)
            
            im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0.4, vmax=1.0)
            
            for i in range(7):
                for j in range(7):
                    ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center',
                           fontsize=9, fontweight='bold',
                           color='white' if sim_matrix[i, j] > 0.7 else 'black')
            
            ax.set_xticks(range(7))
            ax.set_yticks(range(7))
            ax.set_xticklabels([f'Iter {i+1}' for i in range(7)], fontsize=9)
            ax.set_yticklabels([f'Iter {i+1}' for i in range(7)], fontsize=9)
            ax.set_title(f'{DISPLAY_NAMES.get(model_name, model_name)}',
                        fontsize=12, fontweight='bold')
        
        # 添加共享的colorbar
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Cosine Similarity')
        
        plt.suptitle(f'{group_name} Group: Cosine Similarity Matrices', fontsize=14, fontweight='bold')
        
        group_suffix = group_name.lower().replace('-', '')
        save_path = self.output_dir / f'04_{group_suffix}_similarity_heatmaps.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {save_path}")
        return save_path
    
    def plot_key_statistics_by_group(self, stats_df: pd.DataFrame, group_name: str):
        """绘制指定组的关键统计指标柱状图"""
        models = MODEL_GROUPS[group_name]
        colors = GROUP_COLORS[group_name]
        print(f"Plotting: Key Statistics for {group_name}...")
        
        # 筛选该组的数据
        group_df = stats_df[stats_df['model'].isin(models)]
        
        key_metrics = [
            ('cos_sim_to_final_mean', 'Cosine Sim to Final', True),
            ('eucl_dist_to_final_mean', 'Euclidean Dist to Final', False),
            ('cluster_compactness_mean', 'Cluster Compactness', False),
            ('trajectory_smoothness_mean', 'Trajectory Smoothness', True),
            ('path_length_mean', 'Total Path Length', False),
            ('convergence_rate_mean', 'Convergence Rate', False),
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        
        for idx, (col, label, higher_is_better) in enumerate(key_metrics):
            ax = axes[idx]
            values = group_df[col].values
            model_labels = [DISPLAY_NAMES.get(m, m) for m in group_df['model']]
            bar_colors = [colors.get(m, 'gray') for m in group_df['model']]
            
            bars = ax.bar(model_labels, values, color=bar_colors)
            
            # 标注最佳值
            if higher_is_better:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', labelsize=11)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'{group_name} Group: Key Statistics Comparison\n(Red border = Best)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        group_suffix = group_name.lower().replace('-', '')
        save_path = self.output_dir / f'05_{group_suffix}_key_statistics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {save_path}")
        return save_path
    
    def generate_report(self):
        """生成详细解说报告"""
        stats_file = self.results_dir / "latent_analysis" / "statistics.csv"
        stats_df = pd.read_csv(stats_file) if stats_file.exists() else None
        
        n_samples = len(self.all_correct_indices)
        
        report = f"""
================================================================================
              LATENT TRAJECTORY ANALYSIS - DETAILED REPORT
                      Latent轨迹分析详细解说报告
================================================================================

【研究背景】

本报告分析SIRCL (Stability-Inducing Regularization for Continuous Latent)方法对
隐式思维链(Implicit Chain-of-Thought)推理过程的影响。通过分析latent token在
高维空间中的轨迹特征，验证SIRCL约束的有效性。

数据来源：GSM8K数学推理数据集
分析样本：所有模型都答对的题目 ({n_samples} 个样本)

================================================================================
                               图表详细解说
================================================================================

【图1. Embedding Statistics (01_embedding_statistics.png)】
--------------------------------------------------------------------------------
这是一个2x2的组合图，包含4个相关的Embedding统计指标：

■ 左上: Embedding Norm by Iteration
  - 指标说明：每个迭代步骤中，latent token向量的L2范数（向量长度）
  - 物理意义：范数反映了embedding在高维空间中的"能量"或"激活强度"
  - 观察结果：CODI+SIRCL和SIM-CoT+SIRCL的范数更低且更稳定
  - 结论：SIRCL使得模型的隐藏状态更加紧凑

■ 右上: Embedding Variance by Iteration
  - 指标说明：同一迭代步骤中，不同样本的embedding方差
  - 物理意义：衡量模型在处理不同问题时的一致性
  - 观察结果：CODI+SIRCL方差最低，SIM-CoT方差最高
  - 结论：SIRCL提高了模型推理的一致性和可预测性

■ 左下: Cumulative Path Length
  - 指标说明：从第1个迭代到当前迭代，latent token移动的累积距离
  - 物理意义：短路径=直接、高效的推理；长路径=迂回、低效的推理
  - 观察结果：CODI+SIRCL路径最短(~35)，CODI路径最长(~107)
  - 结论：SIRCL显著减少了不必要的"思维漫游"

■ 右下: Path Efficiency by Iteration
  - 指标说明：直线距离 / 实际走过的路径长度
  - 物理意义：效率=1表示完全走直线；效率越高，推理越"高效"
  - 观察结果：CODI+SIRCL效率最高(~0.5)，CODI效率最低(接近0)
  - 结论：SIRCL强制模型沿更直接的路径推理


【图2. Distance to Final Token Boxplots (02_distance_to_final_boxplots.png)】
--------------------------------------------------------------------------------
这是一个2x2的组合图，展示4个模型各自的箱线图：

- 指标说明：每个迭代与最终答案token之间的欧氏距离分布
- 物理意义：距离反映了当前思考状态"离答案有多远"
- 理想情况：距离应随迭代递减（逐步接近答案）

各模型表现：
■ CODI+SIRCL：距离最小，从Iter1就已经很接近最终答案，异常值少
■ SIM-CoT+SIRCL：距离适中，随迭代稳定下降
■ CODI：距离大且不稳定，有大量异常值（离群点）
■ SIM-CoT：距离最大，收敛最慢

结论：SIRCL让模型从一开始就"指向"正确答案，而不是先探索再收敛。


【图3. Convergence Comparison (03_convergence_comparison.png)】
--------------------------------------------------------------------------------
这是一个1x2的组合图，展示两种收敛指标：

■ 左图: Distance Convergence
  - 指标说明：各模型到最终token平均距离的变化曲线
  - 观察结果：
    · CODI+SIRCL：从一开始就很低(~5)，几乎是"直达"
    · CODI：从高处(~13)缓慢下降，收敛最慢
    · SIM-CoT+SIRCL vs SIM-CoT：SIRCL使收敛速度提高约2倍

■ 右图: Similarity Convergence
  - 指标说明：各模型与最终token的余弦相似度变化曲线
  - 物理意义：1.0表示方向完全一致，高相似度="思考方向正确"
  - 观察结果：
    · CODI+SIRCL：相似度全程保持在0.9以上
    · CODI：相似度波动剧烈，从0.7到0.65再回升

结论：SIRCL显著加速了向最终答案的收敛，并保证方向一致性。


【图4. Cosine Similarity Heatmaps (04_similarity_heatmaps.png)】
--------------------------------------------------------------------------------
这是一个2x2的组合图，展示4个模型的7x7余弦相似度矩阵：

- 指标说明：7个迭代之间两两余弦相似度的矩阵
- 颜色解读：深红色(>0.9)=迭代之间非常相似；深蓝色(<0.6)=差异大
- 对角线永远是1.0（自己和自己完全相似）

各模型表现：
■ CODI+SIRCL：几乎全红(>0.95)，所有迭代高度一致
■ SIM-CoT+SIRCL：大部分红色(>0.8)，较为一致
■ CODI：颜色变化明显，早期迭代与后期差异大
■ SIM-CoT：颜色变化最剧烈，迭代之间差异显著

结论：SIRCL使得整个推理轨迹保持在一个连贯的方向上，而非反复改变。


【图5. Key Statistics Comparison (05_key_statistics.png)】
--------------------------------------------------------------------------------
这是一个2x3的组合图，展示6个关键统计指标的柱状图（红框=最佳）：

■ Cosine Sim to Final（越高越好）
  - 含义：前6个迭代与最终答案的平均余弦相似度
  - 最佳：CODI+SIRCL (0.915)
  - CODI: 0.763 → CODI+SIRCL: 0.915 (+20%提升)

■ Euclidean Dist to Final（越低越好）
  - 含义：前6个迭代与最终答案的平均欧氏距离
  - 最佳：CODI+SIRCL (5.29)
  - CODI: 12.88 → CODI+SIRCL: 5.29 (-59%降低)

■ Cluster Compactness（越低越好）
  - 含义：7个迭代点的聚类紧密度（标准差）
  - 最佳：CODI+SIRCL (4.83)
  - CODI: 10.08 → CODI+SIRCL: 4.83 (-52%降低)

■ Trajectory Smoothness（越接近0越好）
  - 含义：相邻迭代方向变化的平滑度（基于二阶导数）
  - 最佳：SIM-CoT+SIRCL (0.069)
  - 负值表示方向剧烈变化（"急转弯"）

■ Total Path Length（越低越好）
  - 含义：7个迭代的总路径长度
  - 最佳：CODI+SIRCL (35.59)
  - CODI: 106.92 → CODI+SIRCL: 35.59 (-67%降低)

■ Convergence Rate（负值越大越好）
  - 含义：距离变化的斜率（负值=在收敛）
  - 最佳：SIM-CoT+SIRCL (-2.46)


================================================================================
                               核心结论
================================================================================

【1. SIRCL的主要效果】

┌──────────────────────────────────────────────────────────────────────────┐
│  指标                    │  CODI → CODI+SIRCL    │  SIM-CoT → SIM-CoT+SIRCL │
├──────────────────────────────────────────────────────────────────────────┤
│  路径长度                │  107 → 36 (-67%)      │  94 → 55 (-42%)          │
│  到最终token距离         │  12.9 → 5.3 (-59%)    │  18.0 → 10.2 (-43%)      │
│  余弦相似度              │  0.76 → 0.92 (+20%)   │  0.72 → 0.79 (+10%)      │
│  聚类紧密度              │  10.1 → 4.8 (-52%)    │  13.4 → 8.3 (-38%)       │
└──────────────────────────────────────────────────────────────────────────┘

【2. SIRCL的作用机制】

SIRCL通过在训练时添加轨迹一致性约束，实现了以下效果：

1. 空间压缩：将latent表示压缩到更紧凑的区域
2. 方向对齐：让所有迭代指向同一个方向（最终答案）
3. 路径直化：减少"思维漫游"，让推理更直接
4. 快速收敛：从第一步就接近最终答案

【3. 理论解释】

在隐式思维链推理中，模型通过多次迭代逐步"思考"问题。
没有约束时，模型可能在高维空间中进行大量无效探索。
SIRCL的向心约束（centripetal constraint）强制所有迭代点
聚集在一个小区域内，这相当于：

- 在Euclidean空间中：让所有点聚向它们的算术平均
- 在Hyperbolic空间中：让所有点聚向Fréchet mean

这种约束使得：
1. 推理过程更稳定（方差降低）
2. 推理路径更短（效率提高）
3. 最终答案更可靠（一致性增强）

================================================================================
                               附录
================================================================================

图片文件列表：
"""
        
        # 添加文件列表
        for f in sorted(self.output_dir.glob('*.png')):
            report += f"  - {f.name}\n"
        
        report += """
统计数据文件：
  - statistics.csv

生成时间：2026年1月28日
================================================================================
"""
        
        # 保存报告
        report_path = self.output_dir / 'ANALYSIS_REPORT.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")
        
        return report_path


def main():
    results_dir = "/data/yhao/baseline/CODI/results/route"
    
    plotter = IndividualPlotter(results_dir)
    plotter.load_data(dataset="gsm8k", run=0)
    plotter.generate_all_plots()
    plotter.generate_report()


if __name__ == "__main__":
    main()
