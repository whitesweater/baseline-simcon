#!/usr/bin/env python3
"""
Latent Token Analysis and Visualization Script
分析和可视化不同模型的latent token分布，包括：
1. 筛选所有模型都答对的题目
2. 以最后一个token为中心，分析其他6个token的分布
3. 使用t-SNE/UMAP/PCA等降维方法投影到2D/3D空间
4. 统计分析：余弦相似度、聚类紧密度、轨迹一致性等
"""

import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 可视化
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# 降维
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed, UMAP will be skipped")

# 统计
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import spearmanr, pearsonr
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 模型显示名称映射
DISPLAY_NAMES = {
    'codi': 'CODI',
    'simcon': 'SIM-CoT',
    'codi_sircl': 'CODI+SIRCL',
    'sircl': 'SIM-CoT+SIRCL'
}

class LatentAnalyzer:
    """Latent Token分析器"""
    
    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: 结果目录路径，包含models/子目录
        """
        self.results_dir = Path(results_dir)
        self.models_dir = self.results_dir / "models"
        self.output_dir = self.results_dir / "latent_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        self.model_names = []
        self.latents = {}  # {model_name: np.array (n_samples, n_iterations, embedding_dim)}
        self.predictions = {}  # {model_name: list}
        self.ground_truths = None  # list (共享)
        self.correct_masks = {}  # {model_name: np.array bool}
        self.all_correct_indices = None  # 所有模型都答对的题目索引
        
    def load_data(self, dataset: str = "gsm8k", run: int = 0):
        """加载所有模型的数据"""
        print(f"Loading data from {self.models_dir}")
        
        # 发现所有模型
        for model_dir in sorted(self.models_dir.iterdir()):
            if model_dir.is_dir():
                self.model_names.append(model_dir.name)
        
        print(f"Found models: {self.model_names}")
        
        # 加载每个模型的数据
        for model_name in self.model_names:
            run_dir = self.models_dir / model_name / dataset / f"run_{run}"
            
            # 加载latents
            latents_file = run_dir / "latents.json"
            if latents_file.exists():
                print(f"Loading latents for {model_name}...")
                with open(latents_file) as f:
                    data = json.load(f)
                self.latents[model_name] = np.array(data['latents'], dtype=np.float32)
                print(f"  Shape: {self.latents[model_name].shape}")
            
            # 加载predictions
            pred_file = run_dir / "predictions.json"
            if pred_file.exists():
                with open(pred_file) as f:
                    pred_data = json.load(f)
                self.predictions[model_name] = pred_data['predictions']
                
                # ground_truth只需加载一次
                if self.ground_truths is None:
                    self.ground_truths = [float(g) for g in pred_data['ground_truth']]
        
        # 计算每个模型的正确率mask
        for model_name in self.model_names:
            preds = self.predictions[model_name]
            correct = []
            for p, g in zip(preds, self.ground_truths):
                try:
                    p_val = float(p) if not isinstance(p, (int, float)) else p
                    g_val = float(g) if not isinstance(g, (int, float)) else g
                    is_correct = abs(p_val - g_val) < 1e-6 or (g_val != 0 and abs(p_val - g_val) / abs(g_val) < 1e-4)
                except (ValueError, TypeError):
                    is_correct = str(p).strip() == str(g).strip()
                correct.append(is_correct)
            correct = np.array(correct)
            self.correct_masks[model_name] = correct
            acc = correct.mean() * 100
            print(f"  {model_name} accuracy: {acc:.2f}%")
        
        # 找出所有模型都答对的题目
        all_correct = np.ones(len(self.ground_truths), dtype=bool)
        for model_name in self.model_names:
            all_correct &= self.correct_masks[model_name]
        
        self.all_correct_indices = np.where(all_correct)[0]
        print(f"\nQuestions all models answered correctly: {len(self.all_correct_indices)}/{len(self.ground_truths)}")
        
    def compute_relative_latents(self) -> Dict[str, np.ndarray]:
        """
        计算相对于最后一个token的latent表示
        Returns:
            {model_name: np.array (n_correct_samples, n_iterations-1, embedding_dim)}
            其中每个向量是相对于最后一个token的偏移
        """
        relative_latents = {}
        
        for model_name in self.model_names:
            latents = self.latents[model_name][self.all_correct_indices]  # (n, 7, dim)
            final_token = latents[:, -1:, :]  # (n, 1, dim)
            # 计算相对位置（其他token - 最后一个token）
            relative = latents[:, :-1, :] - final_token  # (n, 6, dim)
            relative_latents[model_name] = relative
            
        return relative_latents
    
    def compute_statistics(self) -> pd.DataFrame:
        """
        计算统计指标，凸显约束方法使token聚集的优势
        """
        stats = []
        
        for model_name in self.model_names:
            latents = self.latents[model_name][self.all_correct_indices]  # (n, 7, dim)
            n_samples, n_iters, dim = latents.shape
            
            model_stats = {'model': model_name}
            
            # ========== 1. 余弦相似度统计 ==========
            # 计算每个样本中，所有token对之间的余弦相似度
            cos_sims_to_final = []  # 其他token与最后一个token的相似度
            cos_sims_consecutive = []  # 相邻token之间的相似度
            cos_sims_all_pairs = []  # 所有token对之间的相似度
            
            for i in range(n_samples):
                sample_latents = latents[i]  # (7, dim)
                final_token = sample_latents[-1]
                
                # 与最后一个token的相似度
                for j in range(n_iters - 1):
                    sim = 1 - cosine(sample_latents[j], final_token)
                    cos_sims_to_final.append(sim)
                
                # 相邻token的相似度
                for j in range(n_iters - 1):
                    sim = 1 - cosine(sample_latents[j], sample_latents[j+1])
                    cos_sims_consecutive.append(sim)
                
                # 所有token对
                for j in range(n_iters):
                    for k in range(j+1, n_iters):
                        sim = 1 - cosine(sample_latents[j], sample_latents[k])
                        cos_sims_all_pairs.append(sim)
            
            model_stats['cos_sim_to_final_mean'] = np.mean(cos_sims_to_final)
            model_stats['cos_sim_to_final_std'] = np.std(cos_sims_to_final)
            model_stats['cos_sim_consecutive_mean'] = np.mean(cos_sims_consecutive)
            model_stats['cos_sim_all_pairs_mean'] = np.mean(cos_sims_all_pairs)
            
            # ========== 2. 欧氏距离统计 ==========
            eucl_dists_to_final = []
            eucl_dists_consecutive = []
            
            for i in range(n_samples):
                sample_latents = latents[i]
                final_token = sample_latents[-1]
                
                for j in range(n_iters - 1):
                    dist = euclidean(sample_latents[j], final_token)
                    eucl_dists_to_final.append(dist)
                    
                for j in range(n_iters - 1):
                    dist = euclidean(sample_latents[j], sample_latents[j+1])
                    eucl_dists_consecutive.append(dist)
            
            model_stats['eucl_dist_to_final_mean'] = np.mean(eucl_dists_to_final)
            model_stats['eucl_dist_to_final_std'] = np.std(eucl_dists_to_final)
            model_stats['eucl_dist_consecutive_mean'] = np.mean(eucl_dists_consecutive)
            
            # ========== 3. 聚类紧密度 (Cluster Compactness) ==========
            # 计算每个样本中所有token到其中心的平均距离
            compactness_scores = []
            for i in range(n_samples):
                sample_latents = latents[i]  # (7, dim)
                centroid = sample_latents.mean(axis=0)  # (dim,)
                dists = [euclidean(sample_latents[j], centroid) for j in range(n_iters)]
                compactness_scores.append(np.mean(dists))
            
            model_stats['cluster_compactness_mean'] = np.mean(compactness_scores)
            model_stats['cluster_compactness_std'] = np.std(compactness_scores)
            
            # ========== 4. 轨迹平滑度 (Trajectory Smoothness) ==========
            # 计算相邻token之间角度变化的一致性
            smoothness_scores = []
            for i in range(n_samples):
                sample_latents = latents[i]
                # 计算连续的方向向量
                directions = []
                for j in range(n_iters - 1):
                    direction = sample_latents[j+1] - sample_latents[j]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-8:
                        directions.append(direction / norm)
                
                # 计算相邻方向向量之间的夹角余弦（越接近1越平滑）
                if len(directions) >= 2:
                    angle_consistencies = []
                    for j in range(len(directions) - 1):
                        cos_angle = np.dot(directions[j], directions[j+1])
                        angle_consistencies.append(cos_angle)
                    smoothness_scores.append(np.mean(angle_consistencies))
            
            model_stats['trajectory_smoothness_mean'] = np.mean(smoothness_scores) if smoothness_scores else 0
            model_stats['trajectory_smoothness_std'] = np.std(smoothness_scores) if smoothness_scores else 0
            
            # ========== 5. 轨迹长度 (Total Path Length) ==========
            path_lengths = []
            for i in range(n_samples):
                sample_latents = latents[i]
                total_length = 0
                for j in range(n_iters - 1):
                    total_length += euclidean(sample_latents[j], sample_latents[j+1])
                path_lengths.append(total_length)
            
            model_stats['path_length_mean'] = np.mean(path_lengths)
            model_stats['path_length_std'] = np.std(path_lengths)
            
            # ========== 6. 收敛速度 (Convergence Rate) ==========
            # 计算每一步与最终token的距离，看距离是否递减
            convergence_rates = []
            for i in range(n_samples):
                sample_latents = latents[i]
                final_token = sample_latents[-1]
                dists = [euclidean(sample_latents[j], final_token) for j in range(n_iters - 1)]
                # 使用线性回归斜率作为收敛速度
                if len(dists) > 1:
                    x = np.arange(len(dists))
                    slope = np.polyfit(x, dists, 1)[0]
                    convergence_rates.append(slope)
            
            model_stats['convergence_rate_mean'] = np.mean(convergence_rates)  # 负值表示收敛
            model_stats['convergence_rate_std'] = np.std(convergence_rates)
            
            # ========== 7. 半径分布 (Radius Distribution) ==========
            # 计算所有token到最终token的距离分布
            radii = []
            for i in range(n_samples):
                sample_latents = latents[i]
                final_token = sample_latents[-1]
                for j in range(n_iters - 1):
                    radii.append(euclidean(sample_latents[j], final_token))
            
            model_stats['radius_mean'] = np.mean(radii)
            model_stats['radius_std'] = np.std(radii)
            model_stats['radius_max'] = np.max(radii)
            model_stats['radius_90pct'] = np.percentile(radii, 90)
            
            # ========== 8. 方差解释比例 (通过PCA) ==========
            # 计算latent空间的有效维度
            all_latents_flat = latents.reshape(-1, dim)
            pca = PCA(n_components=min(50, dim))
            pca.fit(all_latents_flat)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            model_stats['pca_dim_90pct'] = np.argmax(cumvar >= 0.9) + 1  # 解释90%方差需要的维度
            model_stats['pca_dim_95pct'] = np.argmax(cumvar >= 0.95) + 1
            
            stats.append(model_stats)
        
        df = pd.DataFrame(stats)
        return df
    
    def visualize_2d(self, method: str = 'tsne', perplexity: int = 30):
        """
        2D可视化：以最后一个token为中心，展示其他token的分布
        """
        print(f"\nGenerating 2D visualization using {method}...")
        
        # 准备数据
        all_data = []
        all_labels = []  # (model_idx, iter_idx)
        
        for model_idx, model_name in enumerate(self.model_names):
            latents = self.latents[model_name][self.all_correct_indices]  # (n, 7, dim)
            n_samples = latents.shape[0]
            
            for i in range(n_samples):
                for j in range(7):  # 7个iteration
                    all_data.append(latents[i, j])
                    all_labels.append((model_idx, j, i))  # (model, iteration, sample)
        
        all_data = np.array(all_data)
        print(f"  Data shape for dimensionality reduction: {all_data.shape}")
        
        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            embedded = reducer.fit_transform(all_data)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            embedded = reducer.fit_transform(all_data)
        elif method == 'mds':
            reducer = MDS(n_components=2, random_state=42, n_jobs=-1)
            embedded = reducer.fit_transform(all_data)
        elif method == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedded = reducer.fit_transform(all_data)
        else:
            print(f"  Method {method} not available, using PCA")
            reducer = PCA(n_components=2)
            embedded = reducer.fit_transform(all_data)
        
        # 创建图形 - 每个模型一个子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        colors = cm.viridis(np.linspace(0, 1, 7))  # 7个iteration的颜色
        iteration_labels = [f'Iter {i+1}' for i in range(6)] + ['Final (Iter 7)']
        
        for model_idx, model_name in enumerate(self.model_names):
            ax = axes[model_idx]
            
            # 提取该模型的数据点
            model_mask = np.array([l[0] == model_idx for l in all_labels])
            model_embedded = embedded[model_mask]
            model_labels = [l for l in all_labels if l[0] == model_idx]
            
            # 按iteration分组绘制
            for iter_idx in range(7):
                iter_mask = np.array([l[1] == iter_idx for l in model_labels])
                iter_points = model_embedded[iter_mask]
                
                alpha = 0.3 if iter_idx < 6 else 0.8
                size = 10 if iter_idx < 6 else 30
                marker = 'o' if iter_idx < 6 else '*'
                
                ax.scatter(iter_points[:, 0], iter_points[:, 1], 
                          c=[colors[iter_idx]], s=size, alpha=alpha,
                          label=iteration_labels[iter_idx], marker=marker)
            
            ax.set_title(f'{DISPLAY_NAMES.get(model_name, model_name)}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            if model_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        plt.suptitle(f'Latent Token Distribution (2D {method.upper()})\n'
                    f'All Models Correct: {len(self.all_correct_indices)} samples', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / f'latent_2d_{method}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
        
        return embedded, all_labels
    
    def visualize_3d(self, method: str = 'tsne', perplexity: int = 30):
        """
        3D可视化：球形展示
        """
        print(f"\nGenerating 3D visualization using {method}...")
        
        # 准备数据
        all_data = []
        all_labels = []
        
        for model_idx, model_name in enumerate(self.model_names):
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            
            for i in range(n_samples):
                for j in range(7):
                    all_data.append(latents[i, j])
                    all_labels.append((model_idx, j, i))
        
        all_data = np.array(all_data)
        
        # 降维到3D
        if method == 'tsne':
            reducer = TSNE(n_components=3, perplexity=perplexity, random_state=42)
            embedded = reducer.fit_transform(all_data)
        elif method == 'pca':
            reducer = PCA(n_components=3)
            embedded = reducer.fit_transform(all_data)
        elif method == 'mds':
            reducer = MDS(n_components=3, random_state=42, n_jobs=-1)
            embedded = reducer.fit_transform(all_data)
        elif method == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_components=3, random_state=42)
            embedded = reducer.fit_transform(all_data)
        else:
            reducer = PCA(n_components=3)
            embedded = reducer.fit_transform(all_data)
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        
        colors = cm.viridis(np.linspace(0, 1, 7))
        iteration_labels = [f'Iter {i+1}' for i in range(6)] + ['Final (Iter 7)']
        
        for model_idx, model_name in enumerate(self.model_names):
            ax = fig.add_subplot(2, 2, model_idx + 1, projection='3d')
            
            model_mask = np.array([l[0] == model_idx for l in all_labels])
            model_embedded = embedded[model_mask]
            model_labels = [l for l in all_labels if l[0] == model_idx]
            
            for iter_idx in range(7):
                iter_mask = np.array([l[1] == iter_idx for l in model_labels])
                iter_points = model_embedded[iter_mask]
                
                alpha = 0.3 if iter_idx < 6 else 0.8
                size = 10 if iter_idx < 6 else 50
                
                ax.scatter(iter_points[:, 0], iter_points[:, 1], iter_points[:, 2],
                          c=[colors[iter_idx]], s=size, alpha=alpha,
                          label=iteration_labels[iter_idx])
            
            ax.set_title(f'{DISPLAY_NAMES.get(model_name, model_name)}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
            if model_idx == 0:
                ax.legend(loc='upper left', fontsize=7)
        
        plt.suptitle(f'Latent Token Distribution (3D {method.upper()})\n'
                    f'All Models Correct: {len(self.all_correct_indices)} samples',
                    fontsize=14, fontweight='bold')
        
        save_path = self.output_dir / f'latent_3d_{method}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
        
        return embedded, all_labels
    
    def visualize_relative_distribution(self):
        """
        可视化相对于最后一个token的分布
        使用极坐标/径向图展示距离分布
        """
        print("\nGenerating relative distribution visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 第一行：距离分布（箱线图）
        for model_idx, model_name in enumerate(self.model_names):
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples, n_iters, dim = latents.shape
            
            # 计算每个iteration到final的距离
            distances_by_iter = []
            for iter_idx in range(n_iters - 1):
                dists = []
                for i in range(n_samples):
                    dist = euclidean(latents[i, iter_idx], latents[i, -1])
                    dists.append(dist)
                distances_by_iter.append(dists)
            
            ax = axes[0, model_idx] if model_idx < 3 else None
            if ax is not None:
                bp = ax.boxplot(distances_by_iter, patch_artist=True)
                colors = cm.Blues(np.linspace(0.3, 0.9, 6))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                ax.set_xticklabels([f'Iter {i+1}' for i in range(6)])
                ax.set_ylabel('Distance to Final Token')
                ax.set_title(f'{model_name}\nDistance to Final Token by Iteration')
        
        # 处理第4个模型
        if len(self.model_names) > 3:
            model_name = self.model_names[3]
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples, n_iters, dim = latents.shape
            
            distances_by_iter = []
            for iter_idx in range(n_iters - 1):
                dists = []
                for i in range(n_samples):
                    dist = euclidean(latents[i, iter_idx], latents[i, -1])
                    dists.append(dist)
                distances_by_iter.append(dists)
            
            ax = axes[1, 0]
            bp = ax.boxplot(distances_by_iter, patch_artist=True)
            colors = cm.Blues(np.linspace(0.3, 0.9, 6))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            ax.set_xticklabels([f'Iter {i+1}' for i in range(6)])
            ax.set_ylabel('Distance to Final Token')
            ax.set_title(f'{model_name}\nDistance to Final Token by Iteration')
        
        # 第二行中间：所有模型的平均距离对比
        ax = axes[1, 1]
        for model_idx, model_name in enumerate(self.model_names):
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            
            mean_dists = []
            for iter_idx in range(6):
                dists = [euclidean(latents[i, iter_idx], latents[i, -1]) 
                        for i in range(n_samples)]
                mean_dists.append(np.mean(dists))
            
            ax.plot(range(1, 7), mean_dists, marker='o', label=model_name, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Distance to Final Token')
        ax.set_title('Convergence Comparison Across Models')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 第二行右：余弦相似度对比
        ax = axes[1, 2]
        for model_idx, model_name in enumerate(self.model_names):
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            
            mean_sims = []
            for iter_idx in range(6):
                sims = [1 - cosine(latents[i, iter_idx], latents[i, -1]) 
                       for i in range(n_samples)]
                mean_sims.append(np.mean(sims))
            
            ax.plot(range(1, 7), mean_sims, marker='s', label=model_name, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Cosine Similarity to Final Token')
        ax.set_title('Similarity Convergence Across Models')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'relative_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_heatmap_comparison(self):
        """
        热力图对比：余弦相似度矩阵
        """
        print("\nGenerating similarity heatmap comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for model_idx, model_name in enumerate(self.model_names):
            ax = axes[model_idx]
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples, n_iters, dim = latents.shape
            
            # 计算平均余弦相似度矩阵
            sim_matrix = np.zeros((n_iters, n_iters))
            for i in range(n_iters):
                for j in range(n_iters):
                    sims = []
                    for k in range(n_samples):
                        sim = 1 - cosine(latents[k, i], latents[k, j])
                        sims.append(sim)
                    sim_matrix[i, j] = np.mean(sims)
            
            im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0.5, vmax=1.0)
            ax.set_xticks(range(n_iters))
            ax.set_yticks(range(n_iters))
            ax.set_xticklabels([f'Iter {i+1}' for i in range(n_iters)])
            ax.set_yticklabels([f'Iter {i+1}' for i in range(n_iters)])
            ax.set_title(f'{model_name}\nCosine Similarity Matrix', fontsize=11)
            
            # 添加数值标签
            for i in range(n_iters):
                for j in range(n_iters):
                    text = ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                                  ha='center', va='center', fontsize=8,
                                  color='white' if sim_matrix[i, j] > 0.75 else 'black')
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('Average Cosine Similarity Between Iterations\n'
                    f'(All Models Correct: {len(self.all_correct_indices)} samples)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'similarity_heatmap.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_statistics_comparison(self, stats_df: pd.DataFrame):
        """
        统计指标对比可视化
        """
        print("\nGenerating statistics comparison visualization...")
        
        # 选择关键指标进行可视化
        key_metrics = [
            ('cos_sim_to_final_mean', 'Cosine Sim to Final', True),  # (column, label, higher_is_better)
            ('eucl_dist_to_final_mean', 'Euclidean Dist to Final', False),
            ('cluster_compactness_mean', 'Cluster Compactness', False),
            ('trajectory_smoothness_mean', 'Trajectory Smoothness', True),
            ('path_length_mean', 'Total Path Length', False),
            ('convergence_rate_mean', 'Convergence Rate', False),  # 更负=更好
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.model_names)))
        
        for idx, (col, label, higher_is_better) in enumerate(key_metrics):
            ax = axes[idx]
            values = stats_df[col].values
            
            bars = ax.bar(stats_df['model'], values, color=colors)
            ax.set_ylabel(label)
            ax.set_title(label)
            
            # 标注最佳值
            if higher_is_better:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
            
            ax.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Key Statistics Comparison\n(Red border = Best)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'statistics_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_per_iteration_statistics(self):
        """
        每个iteration的详细统计对比
        """
        print("\nGenerating per-iteration statistics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 每个iteration的平均向量范数
        ax = axes[0, 0]
        for model_name in self.model_names:
            latents = self.latents[model_name][self.all_correct_indices]
            norms_by_iter = []
            for iter_idx in range(7):
                norms = [np.linalg.norm(latents[i, iter_idx]) for i in range(latents.shape[0])]
                norms_by_iter.append(np.mean(norms))
            ax.plot(range(1, 8), norms_by_iter, marker='o', label=model_name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean L2 Norm')
        ax.set_title('Embedding Norm by Iteration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 每个iteration的方差
        ax = axes[0, 1]
        for model_name in self.model_names:
            latents = self.latents[model_name][self.all_correct_indices]
            vars_by_iter = []
            for iter_idx in range(7):
                # 计算该iteration所有样本的方差
                iter_latents = latents[:, iter_idx, :]  # (n_samples, dim)
                var = np.mean(np.var(iter_latents, axis=0))
                vars_by_iter.append(var)
            ax.plot(range(1, 8), vars_by_iter, marker='s', label=model_name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Variance')
        ax.set_title('Embedding Variance by Iteration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 累积距离（到目前为止走过的总路程）
        ax = axes[1, 0]
        for model_name in self.model_names:
            latents = self.latents[model_name][self.all_correct_indices]
            n_samples = latents.shape[0]
            
            cumulative_dists = [0]
            for iter_idx in range(6):
                step_dists = [euclidean(latents[i, iter_idx], latents[i, iter_idx+1])
                             for i in range(n_samples)]
                cumulative_dists.append(cumulative_dists[-1] + np.mean(step_dists))
            
            ax.plot(range(1, 8), cumulative_dists, marker='d', label=model_name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cumulative Distance')
        ax.set_title('Cumulative Path Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 直线距离效率（直接距离 / 走过的路程）
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
                
                # 效率 = 直线距离 / 走过的路程
                eff = np.mean([d/p if p > 1e-8 else 1 for d, p in zip(direct_dists, path_dists)])
                efficiencies.append(eff)
            
            ax.plot(range(2, 8), efficiencies, marker='^', label=model_name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Path Efficiency (direct/actual)')
        ax.set_title('Path Efficiency by Iteration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'per_iteration_statistics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_single_question_trajectories(self, num_samples: int = 10, method: str = 'pca'):
        """
        单题轨迹可视化：对于选定的题目，展示每个模型的latent token轨迹
        
        这是核心对比图，能直观看到：
        - 每个模型的7个latent token如何演化
        - 轨迹是收敛还是发散
        - 路径是平滑还是曲折
        
        理想的轨迹特征（约束方法的优势）：
        1. 更紧密的聚集：所有点更靠近最终token（圆心）
        2. 更短的路径：从起点到终点的总路程更短
        3. 更平滑的轨迹：没有大幅度的跳跃和折返
        4. 更一致的方向：每一步都朝着最终目标前进
        """
        print(f"\nGenerating single question trajectory visualizations...")
        
        trajectory_dir = self.output_dir / 'trajectories'
        trajectory_dir.mkdir(exist_ok=True)
        
        # 随机选择样本（使用固定种子保证可重复）
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.all_correct_indices), 
                                         min(num_samples, len(self.all_correct_indices)), 
                                         replace=False)
        
        # 颜色配置：iteration从浅到深
        iter_colors = plt.cm.Blues(np.linspace(0.3, 1.0, 7))
        # 更新颜色方案以反映 Baseline/Ours 关系
        # SimCon组: 蓝色系
        # CODI组: 橙红色系
        model_colors = {
            'codi': '#F39C12',       # Baseline (Orange)
            'codi_sircl': '#D35400', # Ours (Dark Orange/Red)
            'simcon': '#3498DB',     # Baseline (Blue)
            'sircl': '#1F618D'       # Ours (Dark Blue)
        }
        model_markers = {'codi': 'o', 'codi_sircl': 's', 'simcon': '^', 'sircl': 'D'}
        
        for sample_idx in sample_indices:
            actual_idx = self.all_correct_indices[sample_idx]
            
            # 收集该题目所有模型的latent
            all_latents_for_question = []
            model_labels = []
            
            for model_name in self.model_names:
                latents = self.latents[model_name][actual_idx]  # (7, dim)
                all_latents_for_question.append(latents)
                model_labels.extend([model_name] * 7)
            
            all_latents_for_question = np.vstack(all_latents_for_question)  # (28, dim)
            
            # 降维到2D
            if method == 'pca':
                reducer = PCA(n_components=2)
            else:
                reducer = TSNE(n_components=2, perplexity=5, random_state=42)
            
            embedded_2d = reducer.fit_transform(all_latents_for_question)
            
            # 降维到3D
            if method == 'pca':
                reducer_3d = PCA(n_components=3)
            else:
                reducer_3d = PCA(n_components=3)  # 3D用PCA更稳定
            
            embedded_3d = reducer_3d.fit_transform(all_latents_for_question)
            
            # ==================== 2D轨迹图 (分模型单独保存) ====================
            # 为每个模型生成单独的图，并合并在一张图上（每个模型一个subplot）
            # 用户请求：只画2d 轨迹 ，然后 每种模型的轨迹分开画
            
            n_models = len(self.model_names)
            # 计算合适的行列数
            n_cols = min(n_models, 4)
            n_rows = (n_models + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            if n_models == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for model_idx, model_name in enumerate(self.model_names):
                ax = axes[model_idx]
                start_idx = model_idx * 7
                end_idx = start_idx + 7
                model_points = embedded_2d[start_idx:end_idx]
                
                # 画轨迹线
                ax.plot(model_points[:, 0], model_points[:, 1], 
                       color=model_colors.get(model_name, 'gray'),
                       linewidth=2, alpha=0.9, label='Trajectory')
                
                # 画每个iteration的点
                for i in range(7):
                    size = 120 if i == 6 else 60
                    marker = '*' if i == 6 else model_markers.get(model_name, 'o')
                    # 最后一个点用特殊的金色
                    color = '#FFD700' if i == 6 else iter_colors[i]
                    
                    sc = ax.scatter(model_points[i, 0], model_points[i, 1],
                              c=[color], s=size, marker=marker,
                              edgecolors='black', linewidths=1.0, zorder=5)
                    
                    # 标注iteration编号
                    ax.annotate(f'{i+1}', (model_points[i, 0], model_points[i, 1]),
                               xytext=(3, 3), textcoords='offset points', 
                               fontsize=9, fontweight='bold')
                
                ax.set_title(f'{DISPLAY_NAMES.get(model_name, model_name)}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # 统一坐标轴范围以便比较 (如果需要)
                # ax.set_xlim(x_min, x_max)
                # ax.set_ylim(y_min, y_max)
            
            # 隐藏多余的子图
            for i in range(n_models, len(axes)):
                axes[i].axis('off')
                
            plt.suptitle(f'Single Question Trajectory (2D {method.upper()}) - Question #{actual_idx}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            save_path = trajectory_dir / f'trajectory_q{actual_idx}_{method}_separate.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # 同时保留原来的合并对比图逻辑，但根据需求可能不需要了？
            # 用户说“每种模型的轨迹分开画”，上面的代码已经实现了分subplot画
            # 如果需要完全独立的文件，可以循环保存。这里用subplot比较好对比。
            
            # ==================== 3D轨迹图 (跳过) ====================
            # embedded_3d = reducer_3d.fit_transform(all_latents_for_question)
            # ... 代码省略 ...
            ax.grid(True, alpha=0.3)
            
            # 右图：以最终token为中心的相对位置图
            ax = axes[1]
            for model_idx, model_name in enumerate(self.model_names):
                start_idx = model_idx * 7
                end_idx = start_idx + 7
                model_points = embedded_2d[start_idx:end_idx]
                
                # 以最终token为中心
                final_point = model_points[-1]
                relative_points = model_points - final_point
                
                # 画轨迹线
                ax.plot(relative_points[:, 0], relative_points[:, 1],
                       color=model_colors.get(model_name, 'gray'),
                       linewidth=2, alpha=0.7, label=model_name)
                
                # 画点
                for i in range(7):
                    size = 150 if i == 6 else 50
                    marker = '*' if i == 6 else model_markers.get(model_name, 'o')
                    ax.scatter(relative_points[i, 0], relative_points[i, 1],
                              c=[iter_colors[i]], s=size, marker=marker,
                              edgecolors=model_colors.get(model_name, 'gray'),
                              linewidths=2, zorder=5)
            
            # 画参考圆（表示距离）
            for r in [2, 4, 6]:
                circle = plt.Circle((0, 0), r, fill=False, color='gray', 
                                   linestyle='--', alpha=0.3)
                ax.add_patch(circle)
            
            ax.set_xlabel('Relative Dimension 1')
            ax.set_ylabel('Relative Dimension 2')
            ax.set_title(f'Relative to Final Token (Center)\n'
                        f'Closer to center = Better convergence')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            
            plt.suptitle(f'Single Question Latent Trajectory (2D {method.upper()})\n'
                        f'Question #{actual_idx} - All models answered correctly',
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            save_path = trajectory_dir / f'trajectory_2d_q{actual_idx}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # ==================== 3D轨迹图 (已跳过) ====================
            # fig = plt.figure(figsize=(16, 7))
            
            # 左图：所有模型在同一坐标系
            ax = fig.add_subplot(121, projection='3d')
            for model_idx, model_name in enumerate(self.model_names):
                start_idx = model_idx * 7
                end_idx = start_idx + 7
                model_points = embedded_3d[start_idx:end_idx]
                
                # 画轨迹线
                ax.plot(model_points[:, 0], model_points[:, 1], model_points[:, 2],
                       color=model_colors.get(model_name, 'gray'),
                       linewidth=2, alpha=0.7, label=DISPLAY_NAMES.get(model_name, model_name))
                
                # 画点
                for i in range(7):
                    size = 100 if i == 6 else 30
                    ax.scatter(model_points[i, 0], model_points[i, 1], model_points[i, 2],
                              c=[iter_colors[i]], s=size,
                              edgecolors=model_colors.get(model_name, 'gray'),
                              linewidths=1, zorder=5)
            
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
            ax.set_title(f'All Models 3D Trajectory')
            ax.legend(loc='upper right', fontsize=8)
            
            # 右图：以最终token为中心
            ax = fig.add_subplot(122, projection='3d')
            for model_idx, model_name in enumerate(self.model_names):
                start_idx = model_idx * 7
                end_idx = start_idx + 7
                model_points = embedded_3d[start_idx:end_idx]
                
                final_point = model_points[-1]
                relative_points = model_points - final_point
                
                ax.plot(relative_points[:, 0], relative_points[:, 1], relative_points[:, 2],
                       color=model_colors.get(model_name, 'gray'),
                       linewidth=2, alpha=0.7, label=model_name)
                
                for i in range(7):
                    size = 100 if i == 6 else 30
                    ax.scatter(relative_points[i, 0], relative_points[i, 1], relative_points[i, 2],
                              c=[iter_colors[i]], s=size,
                              edgecolors=model_colors.get(model_name, 'gray'),
                              linewidths=1, zorder=5)
            
            # 画参考球面
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            for r in [2, 4]:
                x = r * np.outer(np.cos(u), np.sin(v))
                y = r * np.outer(np.sin(u), np.sin(v))
                z = r * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_wireframe(x, y, z, color='gray', alpha=0.1, linewidth=0.5)
            
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
            ax.set_title(f'Relative to Final Token (Center)')
            ax.legend(loc='upper right', fontsize=8)
            
            plt.suptitle(f'Single Question Latent Trajectory (3D)\n'
                        f'Question #{actual_idx}',
                        fontsize=12, fontweight='bold')
            
            save_path = trajectory_dir / f'trajectory_3d_q{actual_idx}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Saved {num_samples} trajectory visualizations to {trajectory_dir}")
        
        # ==================== 生成对比组合图 ====================
        self._generate_comparison_summary(sample_indices)
    
    def _generate_comparison_summary(self, sample_indices):
        """
        生成模型对比总结图：
        每道题单独一个图，更清晰地展示对比
        1. sircl vs simcon
        2. codi_sircl vs codi
        3. codi vs codi_sircl (CODI vs CODI+SIRCL)
        4. simcon vs sircl (SIM-CoT vs SIM-CoT+SIRCL)
        """
        print("  Generating comparison summary...")
        
        comparison_pairs = [
            ('sircl', 'simcon', 'SIM-CoT+SIRCL (Ours) vs SIM-CoT (Baseline)'),
            ('codi_sircl', 'codi', 'CODI+SIRCL (Ours) vs CODI (Baseline)'),
            ('codi', 'codi_sircl', 'CODI vs CODI+SIRCL'),
            ('simcon', 'sircl', 'SIM-CoT vs SIM-CoT+SIRCL')
        ]
        
        # 更新颜色方案以反映 Baseline/Ours 关系 (保持与上面一致)
        model_colors = {
            'codi': '#F39C12',       # Baseline (Orange)
            'codi_sircl': '#D35400', # Ours (Dark Orange/Red)
            'simcon': '#3498DB',     # Baseline (Blue)
            'sircl': '#1F618D'       # Ours (Dark Blue)
        }
        
        comparison_dir = self.output_dir / 'comparisons'
        comparison_dir.mkdir(exist_ok=True)
        
        for model_a, model_b, title in comparison_pairs:
            if model_a not in self.model_names or model_b not in self.model_names:
                continue
            
            pair_dir = comparison_dir / f'{model_a}_vs_{model_b}'
            pair_dir.mkdir(exist_ok=True)
            
            for sample_idx in sample_indices:
                actual_idx = self.all_correct_indices[sample_idx]
                
                # 获取两个模型的latent
                latents_a = self.latents[model_a][actual_idx]  # (7, dim)
                latents_b = self.latents[model_b][actual_idx]  # (7, dim)
                
                combined = np.vstack([latents_a, latents_b])  # (14, dim)
                
                # PCA降维到2D和3D
                pca_2d = PCA(n_components=2)
                embedded_2d = pca_2d.fit_transform(combined)
                
                pca_3d = PCA(n_components=3)
                embedded_3d = pca_3d.fit_transform(combined)
                
                points_a_2d = embedded_2d[:7]
                points_b_2d = embedded_2d[7:]
                points_a_3d = embedded_3d[:7]
                points_b_3d = embedded_3d[7:]
                
                # 计算统计指标
                dist_a = np.mean([euclidean(latents_a[i], latents_a[-1]) for i in range(6)])
                dist_b = np.mean([euclidean(latents_b[i], latents_b[-1]) for i in range(6)])
                path_a = sum(euclidean(latents_a[i], latents_a[i+1]) for i in range(6))
                path_b = sum(euclidean(latents_b[i], latents_b[i+1]) for i in range(6))
                cos_a = np.mean([1 - cosine(latents_a[i], latents_a[-1]) for i in range(6)])
                cos_b = np.mean([1 - cosine(latents_b[i], latents_b[-1]) for i in range(6)])
                
                # ==================== 2D对比图 ====================
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # 左图：原始轨迹
                ax = axes[0]
                
                # 模型A (我们的方法)
                ax.plot(points_a_2d[:, 0], points_a_2d[:, 1], 
                       color=model_colors[model_a], linewidth=3, alpha=0.8,
                       label=f'{model_a} (Ours)', zorder=3)
                for i in range(7):
                    size = 200 if i == 6 else 80
                    marker = '*' if i == 6 else 'o'
                    ax.scatter(points_a_2d[i, 0], points_a_2d[i, 1], 
                              c=[model_colors[model_a]], s=size, marker=marker,
                              edgecolors='white', linewidths=2, zorder=5)
                    # 标注iteration编号
                    ax.annotate(f'{i+1}', (points_a_2d[i, 0], points_a_2d[i, 1]),
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=10, fontweight='bold', color=model_colors[model_a])
                
                # 模型B (基线)
                ax.plot(points_b_2d[:, 0], points_b_2d[:, 1],
                       color=model_colors[model_b], linewidth=3, alpha=0.8,
                       label=f'{model_b} (Baseline)', linestyle='--', zorder=2)
                for i in range(7):
                    size = 200 if i == 6 else 80
                    marker = '*' if i == 6 else 's'
                    ax.scatter(points_b_2d[i, 0], points_b_2d[i, 1],
                              c=[model_colors[model_b]], s=size, marker=marker,
                              edgecolors='white', linewidths=2, zorder=4)
                    ax.annotate(f'{i+1}', (points_b_2d[i, 0], points_b_2d[i, 1]),
                               xytext=(-15, -10), textcoords='offset points',
                               fontsize=10, color=model_colors[model_b])
                
                ax.set_xlabel('PCA Dimension 1', fontsize=12)
                ax.set_ylabel('PCA Dimension 2', fontsize=12)
                ax.set_title('Original Trajectory', fontsize=14, fontweight='bold')
                ax.legend(loc='upper right', fontsize=11)
                ax.grid(True, alpha=0.3)
                
                # 右图：以Final为中心
                ax = axes[1]
                
                rel_a = points_a_2d - points_a_2d[-1]
                rel_b = points_b_2d - points_b_2d[-1]
                
                # 模型A
                ax.plot(rel_a[:, 0], rel_a[:, 1],
                       color=model_colors[model_a], linewidth=3, alpha=0.8,
                       label=f'{model_a} (Ours)', zorder=3)
                for i in range(7):
                    size = 200 if i == 6 else 80
                    marker = '*' if i == 6 else 'o'
                    ax.scatter(rel_a[i, 0], rel_a[i, 1],
                              c=[model_colors[model_a]], s=size, marker=marker,
                              edgecolors='white', linewidths=2, zorder=5)
                    ax.annotate(f'{i+1}', (rel_a[i, 0], rel_a[i, 1]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold', color=model_colors[model_a])
                
                # 模型B
                ax.plot(rel_b[:, 0], rel_b[:, 1],
                       color=model_colors[model_b], linewidth=3, alpha=0.8,
                       label=f'{model_b} (Baseline)', linestyle='--', zorder=2)
                for i in range(7):
                    size = 200 if i == 6 else 80
                    marker = '*' if i == 6 else 's'
                    ax.scatter(rel_b[i, 0], rel_b[i, 1],
                              c=[model_colors[model_b]], s=size, marker=marker,
                              edgecolors='white', linewidths=2, zorder=4)
                    ax.annotate(f'{i+1}', (rel_b[i, 0], rel_b[i, 1]),
                               xytext=(-15, -10), textcoords='offset points',
                               fontsize=10, color=model_colors[model_b])
                
                # 画参考圆
                max_dist = max(np.max(np.abs(rel_a)), np.max(np.abs(rel_b))) * 1.2
                for r in np.linspace(max_dist/4, max_dist, 4):
                    circle = plt.Circle((0, 0), r, fill=False, color='gray',
                                       linestyle=':', alpha=0.4)
                    ax.add_patch(circle)
                
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
                ax.set_aspect('equal')
                ax.set_xlabel('Relative Dimension 1', fontsize=12)
                ax.set_ylabel('Relative Dimension 2', fontsize=12)
                ax.set_title('Centered at Final Token (★)', fontsize=14, fontweight='bold')
                ax.legend(loc='upper right', fontsize=11)
                ax.grid(True, alpha=0.3)
                
                # 添加统计信息文本框
                stats_text = (
                    f"Statistics (lower is better):\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{model_a} (ours):\n"
                    f"  • Avg dist to final: {dist_a:.2f}\n"
                    f"  • Path length: {path_a:.2f}\n"
                    f"  • Cosine sim: {cos_a:.3f}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{model_b}:\n"
                    f"  • Avg dist to final: {dist_b:.2f}\n"
                    f"  • Path length: {path_b:.2f}\n"
                    f"  • Cosine sim: {cos_b:.3f}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Improvement:\n"
                    f"  • Distance: {(1 - dist_a/dist_b)*100:.1f}% ↓\n"
                    f"  • Path: {(1 - path_a/path_b)*100:.1f}% ↓"
                )
                
                fig.text(0.02, 0.02, stats_text, fontsize=9, 
                        family='monospace', verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.suptitle(f'{title} - Question #{actual_idx}\n'
                            f'● = Iterations 1-6 (thinking process), ★ = Iteration 7 (final answer token)',
                            fontsize=13, fontweight='bold')
                plt.tight_layout(rect=[0.15, 0.15, 1, 0.95])
                
                save_path = pair_dir / f'q{actual_idx}_2d.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # ==================== 3D对比图 ====================
                fig = plt.figure(figsize=(16, 7))
                
                # 左图：原始轨迹
                ax = fig.add_subplot(121, projection='3d')
                
                ax.plot(points_a_3d[:, 0], points_a_3d[:, 1], points_a_3d[:, 2],
                       color=model_colors[model_a], linewidth=3, alpha=0.8,
                       label=f'{model_a} (ours)')
                for i in range(7):
                    size = 150 if i == 6 else 50
                    ax.scatter(points_a_3d[i, 0], points_a_3d[i, 1], points_a_3d[i, 2],
                              c=[model_colors[model_a]], s=size,
                              edgecolors='white', linewidths=1, zorder=5)
                
                ax.plot(points_b_3d[:, 0], points_b_3d[:, 1], points_b_3d[:, 2],
                       color=model_colors[model_b], linewidth=3, alpha=0.8,
                       label=model_b, linestyle='--')
                for i in range(7):
                    size = 150 if i == 6 else 50
                    ax.scatter(points_b_3d[i, 0], points_b_3d[i, 1], points_b_3d[i, 2],
                              c=[model_colors[model_b]], s=size,
                              edgecolors='white', linewidths=1, zorder=4)
                
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')
                ax.set_zlabel('Dim 3')
                ax.set_title('Original 3D Trajectory', fontsize=12, fontweight='bold')
                ax.legend(loc='upper left', fontsize=10)
                
                # 右图：以Final为中心
                ax = fig.add_subplot(122, projection='3d')
                
                rel_a_3d = points_a_3d - points_a_3d[-1]
                rel_b_3d = points_b_3d - points_b_3d[-1]
                
                ax.plot(rel_a_3d[:, 0], rel_a_3d[:, 1], rel_a_3d[:, 2],
                       color=model_colors[model_a], linewidth=3, alpha=0.8,
                       label=f'{model_a} (ours)')
                for i in range(7):
                    size = 150 if i == 6 else 50
                    ax.scatter(rel_a_3d[i, 0], rel_a_3d[i, 1], rel_a_3d[i, 2],
                              c=[model_colors[model_a]], s=size,
                              edgecolors='white', linewidths=1, zorder=5)
                
                ax.plot(rel_b_3d[:, 0], rel_b_3d[:, 1], rel_b_3d[:, 2],
                       color=model_colors[model_b], linewidth=3, alpha=0.8,
                       label=model_b, linestyle='--')
                for i in range(7):
                    size = 150 if i == 6 else 50
                    ax.scatter(rel_b_3d[i, 0], rel_b_3d[i, 1], rel_b_3d[i, 2],
                              c=[model_colors[model_b]], s=size,
                              edgecolors='white', linewidths=1, zorder=4)
                
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')
                ax.set_zlabel('Dim 3')
                ax.set_title('Centered at Final (3D)', fontsize=12, fontweight='bold')
                ax.legend(loc='upper left', fontsize=10)
                
                plt.suptitle(f'{title} - Question #{actual_idx} (3D View)',
                            fontsize=13, fontweight='bold')
                
                save_path = pair_dir / f'q{actual_idx}_3d.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"    Saved {len(sample_indices)} comparisons to {pair_dir}")
        
        # ==================== 生成解读指南 ====================
        self._generate_interpretation_guide()
    
    def _generate_interpretation_guide(self):
        """生成图片解读指南"""
        guide_text = """
================================================================================
                    LATENT TRAJECTORY VISUALIZATION INTERPRETATION GUIDE
                              图片解读指南
================================================================================

【图片类型说明】

1. trajectory_2d_qX.png / trajectory_3d_qX.png
   单题轨迹可视化，展示每个模型在第X题上的latent演化过程。

2. comparison_sircl_vs_simcon.png
   SIRCL vs SimCon 对比图，展示轨迹约束的效果。

3. comparison_codi_sircl_vs_codi.png  
   CODI+SIRCL vs CODI 对比图，展示组合方法的效果。

--------------------------------------------------------------------------------

【如何判断哪个模型更好？】

✅ 更好的模型（使用约束方法）应该具有：

1. 更紧密的聚集 (Tighter Clustering)
   - 在"以Final为中心"的图中，所有点更靠近中心
   - 参考圆越小，说明latent更集中
   - 指标：eucl_dist_to_final_mean 越小越好

2. 更短的路径长度 (Shorter Path Length)
   - 轨迹线更短，没有长距离的跳跃
   - 从iter1到iter7的总路程更短
   - 指标：path_length_mean 越小越好

3. 更平滑的轨迹 (Smoother Trajectory)
   - 轨迹线更直，没有大角度的折返
   - 每一步的方向更一致
   - 指标：trajectory_smoothness_mean 越大越好

4. 更快的收敛 (Faster Convergence)
   - 早期iteration就开始靠近Final
   - 收敛曲线下降更快
   - 指标：convergence_rate_mean 更负越好

5. 更高的余弦相似度 (Higher Cosine Similarity)
   - 中间token与Final token的方向更一致
   - 指标：cos_sim_to_final_mean 越大越好

--------------------------------------------------------------------------------

【图中符号说明】

- ★ (大星星): Final token (iteration 7)，这是用于生成答案的token
- ● ■ ▲ ◆ (小点): Intermediate tokens (iterations 1-6)，思考过程的中间状态
- 实线: SIRCL/CODI_SIRCL（我们的方法）
- 虚线: SimCon/CODI（基线方法）
- 虚线圆圈: 参考距离，帮助判断聚集程度

--------------------------------------------------------------------------------

【颜色编码】

- 🔴 红色 (codi): 基线CODI模型
- 🟢 绿色 (codi_sircl): CODI + SIRCL（我们的方法）
- 🟠 橙色 (simcon): SimCon基线
- 🔵 蓝色 (sircl): SIRCL（我们的方法）

颜色深浅表示iteration顺序：浅色=早期，深色=后期

--------------------------------------------------------------------------------

【预期结果】

基于约束方法的设计目标，我们预期：

1. sircl > simcon：
   - SIRCL的轨迹更紧凑
   - SIRCL的路径更短
   - SIRCL收敛更快

2. codi_sircl > codi：
   - CODI+SIRCL组合效果更好
   - 聚集度更高
   - 轨迹更平滑

--------------------------------------------------------------------------------

【统计指标对照表】

| 指标 | 含义 | 好的方向 | 我们的优势 |
|------|------|----------|------------|
| cos_sim_to_final_mean | 与Final的余弦相似度 | 越大越好 | 更高相似度=更一致的方向 |
| eucl_dist_to_final_mean | 到Final的欧氏距离 | 越小越好 | 更小距离=更紧密聚集 |
| cluster_compactness_mean | 聚类紧密度 | 越小越好 | 所有token更集中 |
| trajectory_smoothness_mean | 轨迹平滑度 | 越大越好 | 更平滑=更稳定的思考 |
| path_length_mean | 路径总长度 | 越小越好 | 更短路径=更高效的推理 |
| convergence_rate_mean | 收敛速率 | 越负越好 | 更快收敛到答案 |

================================================================================
"""
        
        guide_path = self.output_dir / 'INTERPRETATION_GUIDE.txt'
        with open(guide_path, 'w') as f:
            f.write(guide_text)
        print(f"    Saved interpretation guide to {guide_path}")

    def run_full_analysis(self, skip_existing: bool = True):
        """运行完整分析"""
        print("=" * 60)
        print("Starting Full Latent Token Analysis")
        print("=" * 60)
        
        # 1. 计算统计指标
        stats_path = self.output_dir / 'statistics.csv'
        if skip_existing and stats_path.exists():
            print("\n[1/8] Statistics already exists, loading...")
            stats_df = pd.read_csv(stats_path)
        else:
            print("\n[1/8] Computing statistics...")
            stats_df = self.compute_statistics()
            stats_df.to_csv(stats_path, index=False)
            print(f"  Statistics saved to {stats_path}")
        print("\nStatistics Summary:")
        print(stats_df.to_string())
        
        # 2. 2D可视化 (多种方法)
        print("\n[2/8] 2D Visualizations...")
        for method in ['pca', 'tsne', 'umap']:
            save_path = self.output_dir / f'latent_2d_{method}.png'
            if skip_existing and save_path.exists():
                print(f"  {method} already exists, skipping...")
            else:
                if method == 'umap' and not HAS_UMAP:
                    continue
                self.visualize_2d(method=method)
        
        # 3. 3D可视化
        print("\n[3/8] 3D Visualizations...")
        for method in ['pca', 'tsne']:
            save_path = self.output_dir / f'latent_3d_{method}.png'
            if skip_existing and save_path.exists():
                print(f"  {method} already exists, skipping...")
            else:
                self.visualize_3d(method=method)
        
        # 4. 相对分布可视化
        print("\n[4/8] Relative distribution visualization...")
        save_path = self.output_dir / 'relative_distribution.png'
        if skip_existing and save_path.exists():
            print("  Already exists, skipping...")
        else:
            self.visualize_relative_distribution()
        
        # 5. 相似度热力图
        print("\n[5/8] Similarity heatmap...")
        save_path = self.output_dir / 'similarity_heatmap.png'
        if skip_existing and save_path.exists():
            print("  Already exists, skipping...")
        else:
            self.visualize_heatmap_comparison()
        
        # 6. 统计指标对比
        print("\n[6/8] Statistics comparison...")
        save_path = self.output_dir / 'statistics_comparison.png'
        if skip_existing and save_path.exists():
            print("  Already exists, skipping...")
        else:
            self.visualize_statistics_comparison(stats_df)
        
        # 7. 每iteration统计
        print("\n[7/8] Per-iteration statistics...")
        save_path = self.output_dir / 'per_iteration_statistics.png'
        if skip_existing and save_path.exists():
            print("  Already exists, skipping...")
        else:
            self.visualize_per_iteration_statistics()
        
        # 8. 单题轨迹可视化 (新增)
        print("\n[8/8] Single question trajectory visualization...")
        self.visualize_single_question_trajectories(num_samples=10)
        
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print(f"All results saved to: {self.output_dir}")
        print("=" * 60)
        
        return stats_df
    
    def compute_statistics_for_indices(self, indices: np.ndarray, label: str = "") -> pd.DataFrame:
        """
        对指定的题目索引计算统计指标
        Args:
            indices: 题目索引数组
            label: 标签（用于标识是正确还是错误的题目）
        """
        stats = []
        
        for model_name in self.model_names:
            latents = self.latents[model_name][indices]  # (n, 7, dim)
            n_samples, n_iters, dim = latents.shape
            
            if n_samples == 0:
                continue
            
            model_stats = {'model': model_name, 'category': label, 'n_samples': n_samples}
            
            # 余弦相似度统计
            cos_sims_to_final = []
            cos_sims_consecutive = []
            
            for i in range(n_samples):
                sample_latents = latents[i]
                final_token = sample_latents[-1]
                
                for j in range(n_iters - 1):
                    sim = 1 - cosine(sample_latents[j], final_token)
                    cos_sims_to_final.append(sim)
                
                for j in range(n_iters - 1):
                    sim = 1 - cosine(sample_latents[j], sample_latents[j+1])
                    cos_sims_consecutive.append(sim)
            
            model_stats['cos_sim_to_final_mean'] = np.mean(cos_sims_to_final)
            model_stats['cos_sim_to_final_std'] = np.std(cos_sims_to_final)
            model_stats['cos_sim_consecutive_mean'] = np.mean(cos_sims_consecutive)
            
            # 欧氏距离统计
            eucl_dists_to_final = []
            eucl_dists_consecutive = []
            
            for i in range(n_samples):
                sample_latents = latents[i]
                final_token = sample_latents[-1]
                
                for j in range(n_iters - 1):
                    dist = euclidean(sample_latents[j], final_token)
                    eucl_dists_to_final.append(dist)
                    
                for j in range(n_iters - 1):
                    dist = euclidean(sample_latents[j], sample_latents[j+1])
                    eucl_dists_consecutive.append(dist)
            
            model_stats['eucl_dist_to_final_mean'] = np.mean(eucl_dists_to_final)
            model_stats['eucl_dist_to_final_std'] = np.std(eucl_dists_to_final)
            model_stats['eucl_dist_consecutive_mean'] = np.mean(eucl_dists_consecutive)
            
            # 聚类紧密度
            compactness_scores = []
            for i in range(n_samples):
                sample_latents = latents[i]
                centroid = sample_latents.mean(axis=0)
                dists = [euclidean(sample_latents[j], centroid) for j in range(n_iters)]
                compactness_scores.append(np.mean(dists))
            
            model_stats['cluster_compactness_mean'] = np.mean(compactness_scores)
            
            # 轨迹平滑度
            smoothness_scores = []
            for i in range(n_samples):
                sample_latents = latents[i]
                directions = []
                for j in range(n_iters - 1):
                    direction = sample_latents[j+1] - sample_latents[j]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-8:
                        directions.append(direction / norm)
                
                if len(directions) >= 2:
                    angle_consistencies = []
                    for j in range(len(directions) - 1):
                        cos_angle = np.dot(directions[j], directions[j+1])
                        angle_consistencies.append(cos_angle)
                    smoothness_scores.append(np.mean(angle_consistencies))
            
            model_stats['trajectory_smoothness_mean'] = np.mean(smoothness_scores) if smoothness_scores else 0
            
            # 路径长度
            path_lengths = []
            for i in range(n_samples):
                sample_latents = latents[i]
                total_length = sum(euclidean(sample_latents[j], sample_latents[j+1]) for j in range(n_iters - 1))
                path_lengths.append(total_length)
            
            model_stats['path_length_mean'] = np.mean(path_lengths)
            
            # 收敛速率
            convergence_rates = []
            for i in range(n_samples):
                sample_latents = latents[i]
                final_token = sample_latents[-1]
                dists = [euclidean(sample_latents[j], final_token) for j in range(n_iters - 1)]
                if len(dists) > 1:
                    x = np.arange(len(dists))
                    slope = np.polyfit(x, dists, 1)[0]
                    convergence_rates.append(slope)
            
            model_stats['convergence_rate_mean'] = np.mean(convergence_rates) if convergence_rates else 0
            
            # ========== 核心发散指标：Latent Token 分布熵 (向量化计算) ==========
            # 直接衡量前6个latent token在高维空间中分布的混乱程度
            
            # 批量计算前6个token的两两距离
            latents_first6 = latents[:, :-1, :]  # (n_samples, 6, dim)
            
            # 1. Latent Token 两两距离熵 - 使用scipy pdist向量化
            from scipy.spatial.distance import pdist
            pairwise_entropies = []
            for i in range(n_samples):
                dists = pdist(latents_first6[i])  # 快速计算两两距离
                if len(dists) > 0:
                    hist, _ = np.histogram(dists, bins=10, density=True)
                    hist = hist + 1e-10
                    hist = hist / hist.sum()
                    pairwise_entropies.append(-np.sum(hist * np.log2(hist)))
            model_stats['latent_pairwise_entropy'] = np.mean(pairwise_entropies) if pairwise_entropies else 0
            
            # 2. Latent Token 散度 (用方差代替PCA体积，更快)
            # 直接计算各维度方差的平均值
            latent_spreads = np.var(latents_first6, axis=1).mean(axis=1)  # (n_samples,)
            model_stats['latent_volume_mean'] = np.mean(latent_spreads)
            
            # 3. Latent Token 分布不均匀度 - 向量化
            centroids = latents_first6.mean(axis=1, keepdims=True)  # (n_samples, 1, dim)
            dists_to_centroid = np.linalg.norm(latents_first6 - centroids, axis=2)  # (n_samples, 6)
            mean_dists = dists_to_centroid.mean(axis=1)
            std_dists = dists_to_centroid.std(axis=1)
            cv = np.where(mean_dists > 1e-8, std_dists / mean_dists, 0)
            model_stats['latent_unevenness_mean'] = np.mean(cv)
            
            # 4. Latent Token 间距比 (max/min) - 向量化
            gap_ratios = []
            for i in range(n_samples):
                dists = pdist(latents_first6[i])
                if len(dists) > 0 and dists.min() > 1e-8:
                    gap_ratios.append(dists.max() / dists.min())
            model_stats['latent_gap_ratio_mean'] = np.mean(gap_ratios) if gap_ratios else 1.0
            
            # ========== 原有发散指标 (简化版) ==========
            
            # 步进向量
            steps = latents[:, 1:, :] - latents[:, :-1, :]  # (n_samples, 6, dim)
            step_norms = np.linalg.norm(steps, axis=2)  # (n_samples, 6)
            
            # 步长方差
            model_stats['step_size_variance_mean'] = np.mean(np.var(step_norms, axis=1))
            
            # 轨迹能量
            model_stats['trajectory_energy_mean'] = np.mean(np.sum(step_norms**2, axis=1))
            
            # 方向熵 - 简化：用相邻步向量夹角的方差代替
            step_normalized = steps / (step_norms[:, :, np.newaxis] + 1e-8)
            cos_angles = np.sum(step_normalized[:, :-1, :] * step_normalized[:, 1:, :], axis=2)  # (n_samples, 5)
            cos_angles = np.clip(cos_angles, -1, 1)
            angles = np.arccos(cos_angles)  # (n_samples, 5)
            model_stats['direction_entropy_mean'] = np.mean(np.var(angles, axis=1))
            
            # 回溯率 - 向量化
            final_tokens = latents[:, -1:, :]  # (n_samples, 1, dim)
            dists_to_final_all = np.linalg.norm(latents[:, :-1, :] - final_tokens, axis=2)  # (n_samples, 6)
            backtrack = (dists_to_final_all[:, 1:] > dists_to_final_all[:, :-1]).astype(float)  # (n_samples, 5)
            model_stats['backtracking_rate_mean'] = np.mean(backtrack)
            
            # Lyapunov估计 - 向量化
            lyapunov = np.log(step_norms[:, 1:] / (step_norms[:, :-1] + 1e-8) + 1e-8)  # (n_samples, 5)
            model_stats['lyapunov_estimate_mean'] = np.mean(lyapunov)
            
            # ========== 核心僵化指标：距离到Final的停滞检测 (向量化) ==========
            # dists_to_final_all: (n_samples, 6) - 每个token到final的距离
            
            # 1. 最后3步距离变化率
            last3_dists = dists_to_final_all[:, -3:]  # (n_samples, 3)
            last3_changes = np.abs(np.diff(last3_dists, axis=1))  # (n_samples, 2)
            initial_dists = dists_to_final_all[:, 0]  # (n_samples,)
            final_change_rate = np.where(initial_dists > 1e-8, 
                                         last3_changes.mean(axis=1) / initial_dists, 0)
            model_stats['final_dist_change_rate_mean'] = np.mean(final_change_rate)
            
            # 2. 距离停滞指数 (最后3步距离的标准差/初始距离)
            last3_std = np.std(last3_dists, axis=1)
            stagnation_index = np.where(initial_dists > 1e-8, last3_std / initial_dists, 0)
            model_stats['dist_stagnation_index_mean'] = np.mean(stagnation_index)
            
            # 3. 推理进展比 (最后2步进展 / 前4步进展)
            early_progress = dists_to_final_all[:, 0] - dists_to_final_all[:, 3]  # 前4步
            late_progress = dists_to_final_all[:, -2] - dists_to_final_all[:, -1]  # 最后2步
            progress_ratio = np.where(np.abs(early_progress) > 1e-8, 
                                      late_progress / early_progress, 0)
            model_stats['reasoning_progress_ratio_mean'] = np.mean(progress_ratio)
            
            # 4. 停滞检测 (最后3步CV < 5%)
            last3_mean = last3_dists.mean(axis=1)
            last3_cv = np.where(last3_mean > 1e-8, last3_std / last3_mean, 0)
            stagnation_detected = (last3_cv < 0.05).astype(float)
            model_stats['stagnation_detected_ratio'] = np.mean(stagnation_detected)
            
            # 5. 非单调比例 (距离增加的步数)
            dist_increases = (np.diff(dists_to_final_all, axis=1) >= 0).astype(float)  # (n_samples, 5)
            model_stats['non_monotonic_ratio_mean'] = np.mean(dist_increases)
            
            # ========== 原有僵化检测指标 (向量化) ==========
            
            # 轨迹跨度
            spans = np.linalg.norm(latents[:, -1, :] - latents[:, 0, :], axis=1)
            model_stats['trajectory_span_mean'] = np.mean(spans)
            
            # 有效移动比例
            path_lengths_arr = np.sum(step_norms, axis=1)
            effective_ratio = np.where(path_lengths_arr > 1e-8, spans / path_lengths_arr, 1.0)
            model_stats['effective_movement_ratio_mean'] = np.mean(effective_ratio)
            
            # 步长衰减率
            x = np.arange(n_iters - 1)
            decay_rates = []
            for i in range(n_samples):
                if step_norms[i].mean() > 1e-8:
                    slope = np.polyfit(x, step_norms[i], 1)[0]
                    decay_rates.append(-slope / step_norms[i].mean())
            model_stats['step_decay_rate_mean'] = np.mean(decay_rates) if decay_rates else 0
            
            # 最后3步停滞度
            final3_step_mean = step_norms[:, -3:].mean(axis=1)
            total_step_mean = step_norms.mean(axis=1)
            final_stag = np.where(total_step_mean > 1e-8, final3_step_mean / total_step_mean, 1.0)
            model_stats['final_stagnation_ratio_mean'] = np.mean(final_stag)
            
            # 停滞步数比例 (步长 < 平均的30%)
            threshold = step_norms.mean(axis=1, keepdims=True) * 0.3
            stagnant_steps = (step_norms < threshold).astype(float).mean(axis=1)
            model_stats['stagnant_steps_ratio_mean'] = np.mean(stagnant_steps)
            
            # 振荡检测
            dist_changes = np.diff(dists_to_final_all, axis=1)  # (n_samples, 5)
            direction_changes = (dist_changes[:, :-1] * dist_changes[:, 1:] < 0).astype(float)
            model_stats['oscillation_score_mean'] = np.mean(direction_changes)
            
            # 动态范围
            step_max = step_norms.max(axis=1)
            step_min = step_norms.min(axis=1)
            dynamic_range = np.where(step_min > 1e-8, step_max / step_min, 1.0)
            model_stats['dynamic_range_mean'] = np.mean(dynamic_range)
            
            # 首尾步长比
            first_last_ratio = np.where(step_norms[:, -1] > 1e-8, 
                                        step_norms[:, 0] / step_norms[:, -1], 1.0)
            model_stats['first_last_step_ratio_mean'] = np.mean(first_last_ratio)
            
            stats.append(model_stats)
        
        return pd.DataFrame(stats)
    
    def analyze_correct_vs_wrong(self):
        """
        分析每个模型在答对和答错题目上的表现差异
        """
        print("\n" + "=" * 60)
        print("CORRECT vs WRONG ANALYSIS")
        print("=" * 60)
        
        wrong_analysis_dir = self.output_dir / 'correct_vs_wrong'
        wrong_analysis_dir.mkdir(exist_ok=True)
        
        all_stats = []
        
        for model_name in self.model_names:
            correct_mask = self.correct_masks[model_name]
            correct_indices = np.where(correct_mask)[0]
            wrong_indices = np.where(~correct_mask)[0]
            
            print(f"\n{model_name}:")
            print(f"  Correct: {len(correct_indices)}, Wrong: {len(wrong_indices)}")
            
            # 计算正确题目的统计
            correct_stats = self.compute_statistics_for_indices(correct_indices, 'correct')
            correct_stats = correct_stats[correct_stats['model'] == model_name]
            
            # 计算错误题目的统计
            wrong_stats = self.compute_statistics_for_indices(wrong_indices, 'wrong')
            wrong_stats = wrong_stats[wrong_stats['model'] == model_name]
            
            all_stats.append(correct_stats)
            all_stats.append(wrong_stats)
        
        # 合并所有统计
        combined_stats = pd.concat(all_stats, ignore_index=True)
        stats_path = wrong_analysis_dir / 'correct_vs_wrong_statistics.csv'
        combined_stats.to_csv(stats_path, index=False)
        print(f"\nStatistics saved to {stats_path}")
        
        # 生成对比可视化
        self._visualize_correct_vs_wrong_comparison(combined_stats, wrong_analysis_dir)
        
        # 生成单题对比可视化（每个模型选几道对的和几道错的）
        self._visualize_correct_vs_wrong_trajectories(wrong_analysis_dir, num_samples=5)
        
        return combined_stats
    
    def _visualize_correct_vs_wrong_comparison(self, stats_df: pd.DataFrame, output_dir: Path):
        """
        可视化正确vs错误的统计对比
        """
        print("\nGenerating correct vs wrong comparison charts...")
        
        metrics = [
            ('cos_sim_to_final_mean', 'Cosine Similarity to Final', True),
            ('eucl_dist_to_final_mean', 'Euclidean Distance to Final', False),
            ('cluster_compactness_mean', 'Cluster Compactness', False),
            ('trajectory_smoothness_mean', 'Trajectory Smoothness', True),
            ('path_length_mean', 'Path Length', False),
            ('convergence_rate_mean', 'Convergence Rate', False),
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = {'correct': '#27ae60', 'wrong': '#e74c3c'}
        
        for idx, (col, label, higher_is_better) in enumerate(metrics):
            ax = axes[idx]
            
            x = np.arange(len(self.model_names))
            width = 0.35
            
            correct_vals = []
            wrong_vals = []
            
            for model_name in self.model_names:
                correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
                wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
                
                correct_vals.append(correct_row[col].values[0] if len(correct_row) > 0 else 0)
                wrong_vals.append(wrong_row[col].values[0] if len(wrong_row) > 0 else 0)
            
            bars1 = ax.bar(x - width/2, correct_vals, width, label='Correct ✓', color=colors['correct'], alpha=0.8)
            bars2 = ax.bar(x + width/2, wrong_vals, width, label='Wrong ✗', color=colors['wrong'], alpha=0.8)
            
            ax.set_ylabel(label)
            ax.set_title(f'{label}\n({"Higher" if higher_is_better else "Lower"} is better)')
            ax.set_xticks(x)
            ax.set_xticklabels(self.model_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Correct vs Wrong Questions: Latent Statistics Comparison\n'
                    'Green = Correct answers, Red = Wrong answers',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / 'correct_vs_wrong_statistics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")
        
        # 生成差异百分比图
        self._visualize_difference_percentage(stats_df, output_dir)
    
    def _visualize_difference_percentage(self, stats_df: pd.DataFrame, output_dir: Path):
        """
        可视化正确vs错误的差异百分比
        """
        metrics = [
            ('cos_sim_to_final_mean', 'Cosine Sim'),
            ('eucl_dist_to_final_mean', 'Eucl Dist'),
            ('cluster_compactness_mean', 'Compactness'),
            ('path_length_mean', 'Path Length'),
        ]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.model_names))
        width = 0.2
        
        for i, (col, label) in enumerate(metrics):
            diffs = []
            for model_name in self.model_names:
                correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
                wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
                
                if len(correct_row) > 0 and len(wrong_row) > 0:
                    c_val = correct_row[col].values[0]
                    w_val = wrong_row[col].values[0]
                    if c_val != 0:
                        diff_pct = (w_val - c_val) / abs(c_val) * 100
                    else:
                        diff_pct = 0
                else:
                    diff_pct = 0
                diffs.append(diff_pct)
            
            ax.bar(x + i * width, diffs, width, label=label)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Difference % (Wrong - Correct) / Correct')
        ax.set_title('How Different are Wrong Answers from Correct Ones?\n'
                    'Positive = Wrong answers have HIGHER value')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(self.model_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = output_dir / 'correct_vs_wrong_difference.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")
        
        # 添加发散指标可视化
        self._visualize_divergence_metrics(stats_df, output_dir)
    
    def _visualize_core_divergence(self, stats_df: pd.DataFrame, output_dir: Path, metrics: list):
        """
        可视化核心发散指标：Latent Token分布熵
        直接衡量前6个latent token在高维空间中的分布混乱程度
        """
        print("\n  Visualizing CORE divergence metrics (Latent Distribution)...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = {'correct': '#27ae60', 'wrong': '#e74c3c'}
        
        for idx, (col, label_en, label_cn, higher_is_chaotic) in enumerate(metrics):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            x = np.arange(len(self.model_names))
            width = 0.35
            
            correct_vals = []
            wrong_vals = []
            
            for model_name in self.model_names:
                correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
                wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
                
                c_val = correct_row[col].values[0] if len(correct_row) > 0 else 0
                w_val = wrong_row[col].values[0] if len(wrong_row) > 0 else 0
                correct_vals.append(c_val)
                wrong_vals.append(w_val)
            
            bars1 = ax.bar(x - width/2, correct_vals, width, label='Correct ✓', color=colors['correct'], alpha=0.8)
            bars2 = ax.bar(x + width/2, wrong_vals, width, label='Wrong ✗', color=colors['wrong'], alpha=0.8)
            
            ax.set_ylabel(label_en)
            indicator = "Higher = more chaotic" if higher_is_chaotic else "Lower = more chaotic"
            ax.set_title(f'🎯 {label_en}\n({label_cn})\n{indicator}', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.model_names, rotation=45, ha='right')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        # 隐藏多余axes
        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('🔥 CORE Divergence Metrics: Latent Token Distribution Entropy\n'
                    '(Measuring how chaotically the 6 latent tokens are distributed in space)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / 'core_divergence_latent_entropy.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved {save_path}")
        
        # 差异百分比图
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.model_names))
        width = 0.15
        colors_bar = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
        
        for i, (col, label_en, label_cn, _) in enumerate(metrics):
            diffs = []
            for model_name in self.model_names:
                correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
                wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
                
                if len(correct_row) > 0 and len(wrong_row) > 0:
                    c_val = correct_row[col].values[0]
                    w_val = wrong_row[col].values[0]
                    if abs(c_val) > 1e-8:
                        diff_pct = (w_val - c_val) / abs(c_val) * 100
                    else:
                        diff_pct = 0
                else:
                    diff_pct = 0
                diffs.append(diff_pct)
            
            offset = (i - len(metrics)/2) * width
            ax.bar(x + offset, diffs, width, label=label_en, color=colors_bar[i], alpha=0.85)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Difference % (Wrong - Correct)', fontsize=12)
        ax.set_title('🔥 Latent Token Entropy Change: Wrong vs Correct\n'
                    'Positive = Wrong answers have MORE chaotic latent distribution',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names, fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = output_dir / 'core_divergence_difference.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved {save_path}")
        
        # 打印关键发现
        print("\n    " + "=" * 50)
        print("    🔥 LATENT TOKEN ENTROPY KEY FINDINGS:")
        print("    " + "=" * 50)
        for model_name in self.model_names:
            correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
            wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
            
            if len(correct_row) > 0 and len(wrong_row) > 0:
                print(f"\n    {model_name}:")
                for col, label_en, _, _ in metrics[:3]:  # 只显示前3个核心指标
                    c_val = correct_row[col].values[0]
                    w_val = wrong_row[col].values[0]
                    diff = (w_val - c_val) / abs(c_val) * 100 if abs(c_val) > 1e-8 else 0
                    emoji = "⬆️" if diff > 5 else ("⬇️" if diff < -5 else "➡️")
                    print(f"      {emoji} {label_en}: Correct={c_val:.4f}, Wrong={w_val:.4f} ({diff:+.1f}%)")
    
    def _visualize_core_stagnation(self, stats_df: pd.DataFrame, output_dir: Path, metrics: list):
        """
        可视化核心僵化指标：距离到Final的停滞检测
        检测最后N个token到final的距离是否基本不变（失去推理能力）
        """
        print("\n  Visualizing CORE stagnation metrics (Distance to Final)...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = {'correct': '#27ae60', 'wrong': '#e74c3c'}
        
        for idx, (col, label_en, label_cn, lower_is_stagnant) in enumerate(metrics):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            x = np.arange(len(self.model_names))
            width = 0.35
            
            correct_vals = []
            wrong_vals = []
            
            for model_name in self.model_names:
                correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
                wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
                
                c_val = correct_row[col].values[0] if len(correct_row) > 0 else 0
                w_val = wrong_row[col].values[0] if len(wrong_row) > 0 else 0
                correct_vals.append(c_val)
                wrong_vals.append(w_val)
            
            bars1 = ax.bar(x - width/2, correct_vals, width, label='Correct ✓', color=colors['correct'], alpha=0.8)
            bars2 = ax.bar(x + width/2, wrong_vals, width, label='Wrong ✗', color=colors['wrong'], alpha=0.8)
            
            ax.set_ylabel(label_en)
            indicator = "Lower = stagnant" if lower_is_stagnant else "Higher = stagnant"
            ax.set_title(f'🧊 {label_en}\n({label_cn})\n{indicator}', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.model_names, rotation=45, ha='right')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('🧊 CORE Stagnation Metrics: Distance to Final Token\n'
                    '(Detecting if last N tokens\' distance to final stays unchanged = lost reasoning ability)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / 'core_stagnation_distance_to_final.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved {save_path}")
        
        # 差异百分比图
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.model_names))
        width = 0.12
        colors_bar = plt.cm.Set1(np.linspace(0, 1, len(metrics)))
        
        for i, (col, label_en, label_cn, lower_is_stagnant) in enumerate(metrics):
            diffs = []
            for model_name in self.model_names:
                correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
                wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
                
                if len(correct_row) > 0 and len(wrong_row) > 0:
                    c_val = correct_row[col].values[0]
                    w_val = wrong_row[col].values[0]
                    if abs(c_val) > 1e-8:
                        diff_pct = (w_val - c_val) / abs(c_val) * 100
                    else:
                        diff_pct = 0
                else:
                    diff_pct = 0
                diffs.append(diff_pct)
            
            offset = (i - len(metrics)/2) * width
            ax.bar(x + offset, diffs, width, label=label_en, color=colors_bar[i], alpha=0.85)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Difference % (Wrong - Correct)', fontsize=12)
        ax.set_title('🧊 Distance Stagnation Change: Wrong vs Correct\n'
                    'Negative (for "lower=stagnant" metrics) = Wrong answers show MORE stagnation',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names, fontsize=11)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = output_dir / 'core_stagnation_difference.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved {save_path}")
        
        # 打印关键发现
        print("\n    " + "=" * 50)
        print("    🧊 DISTANCE STAGNATION KEY FINDINGS:")
        print("    " + "=" * 50)
        for model_name in self.model_names:
            correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
            wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
            
            if len(correct_row) > 0 and len(wrong_row) > 0:
                print(f"\n    {model_name}:")
                
                # 显示最重要的僵化指标
                key_metrics = [
                    ('stagnation_detected_ratio', 'Stagnation Detected', False),  # higher = stagnant
                    ('final_dist_change_rate_mean', 'Final Dist Change', True),  # lower = stagnant
                    ('reasoning_progress_ratio_mean', 'Reasoning Progress', True),  # lower = stagnant
                ]
                
                for col, label, lower_is_stagnant in key_metrics:
                    if col in correct_row.columns:
                        c_val = correct_row[col].values[0]
                        w_val = wrong_row[col].values[0]
                        diff = (w_val - c_val) / abs(c_val) * 100 if abs(c_val) > 1e-8 else 0
                        
                        # 判断是否显示更僵化
                        if lower_is_stagnant:
                            is_more_stagnant = diff < -5
                        else:
                            is_more_stagnant = diff > 5
                        
                        emoji = "🧊" if is_more_stagnant else ("🔥" if abs(diff) > 5 else "➡️")
                        print(f"      {emoji} {label}: Correct={c_val:.4f}, Wrong={w_val:.4f} ({diff:+.1f}%)")
                        
                        if is_more_stagnant:
                            print(f"         → Wrong answers show MORE stagnation!")

    def _visualize_divergence_metrics(self, stats_df: pd.DataFrame, output_dir: Path):
        """
        可视化发散与失控指标 (Divergence & Chaos Metrics)
        这些指标用于衡量错误答案是否表现出更加"发散"和"失控"的轨迹
        """
        print("\nVisualizing divergence & chaos metrics...")
        
        # 核心发散指标（新增：直接衡量latent token分布）
        core_divergence_metrics = [
            ('latent_pairwise_entropy', 'Latent Pairwise Entropy', 'Latent两两距离熵', True),
            ('latent_volume_mean', 'Latent Volume', 'Latent体积/散度', True),
            ('latent_unevenness_mean', 'Latent Unevenness', 'Latent分布不均匀度', True),
            ('local_density_variation_mean', 'Local Density Var', 'Latent局部密度变化', True),
            ('latent_gap_ratio_mean', 'Latent Gap Ratio', 'Latent间距比', True),
        ]
        
        # 原有发散相关指标
        divergence_metrics = [
            ('direction_entropy_mean', 'Direction Entropy', '方向熵', True),  # (col, label_en, label_cn, higher_is_worse)
            ('step_size_variance_mean', 'Step Size Variance', '步长方差', True),
            ('curvature_variation_mean', 'Curvature Variation', '曲率变化', True),
            ('angular_acceleration_mean', 'Angular Acceleration', '角加速度', True),
            ('trajectory_dispersion_mean', 'Trajectory Dispersion', '轨迹分散度', True),
            ('lyapunov_estimate_mean', 'Lyapunov Estimate', '李雅普诺夫指数', True),
            ('backtracking_rate_mean', 'Backtracking Rate', '回溯率', True),
            ('trajectory_energy_mean', 'Trajectory Energy', '轨迹能量', True),
        ]
        
        # 合并所有发散指标
        all_divergence_metrics = core_divergence_metrics + divergence_metrics
        
        # 检查是否有这些指标
        available_metrics = [m for m in all_divergence_metrics if m[0] in stats_df.columns]
        
        if not available_metrics:
            print("  No divergence metrics available in statistics")
            return
        
        # 先单独可视化核心发散指标
        core_available = [m for m in core_divergence_metrics if m[0] in stats_df.columns]
        if core_available:
            self._visualize_core_divergence(stats_df, output_dir, core_available)
        
        # 可视化核心僵化指标（距离停滞检测）
        core_stagnation_metrics = [
            ('final_dist_change_rate_mean', 'Final Dist Change Rate', '末期距离变化率', True),  # 低=停滞
            ('dist_stagnation_index_mean', 'Dist Stagnation Index', '距离停滞指数', True),  # 低=停滞
            ('convergence_stop_point_mean', 'Convergence Stop Point', '收敛停止点', True),  # 低=早停
            ('reasoning_progress_ratio_mean', 'Reasoning Progress', '推理进展比', True),  # 低=后期无进展
            ('stagnation_detected_ratio', 'Stagnation Detected %', '检测到停滞比例', False),  # 高=更多停滞
            ('non_monotonic_ratio_mean', 'Non-monotonic Ratio', '非单调比例', False),  # 高=更多折返
        ]
        core_stag_available = [m for m in core_stagnation_metrics if m[0] in stats_df.columns]
        if core_stag_available:
            self._visualize_core_stagnation(stats_df, output_dir, core_stag_available)
        
        n_metrics = len(available_metrics)
        cols = 4
        rows = (n_metrics + cols - 1) // cols
        
        # 图1：正确vs错误的发散指标对比
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        colors = {'correct': '#27ae60', 'wrong': '#e74c3c'}
        
        for idx, (col, label_en, label_cn, higher_is_worse) in enumerate(available_metrics):
            ax = axes[idx]
            x = np.arange(len(self.model_names))
            width = 0.35
            
            correct_vals = []
            wrong_vals = []
            
            for model_name in self.model_names:
                correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
                wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
                
                c_val = correct_row[col].values[0] if len(correct_row) > 0 else 0
                w_val = wrong_row[col].values[0] if len(wrong_row) > 0 else 0
                correct_vals.append(c_val)
                wrong_vals.append(w_val)
            
            bars1 = ax.bar(x - width/2, correct_vals, width, label='Correct ✓', color=colors['correct'], alpha=0.8)
            bars2 = ax.bar(x + width/2, wrong_vals, width, label='Wrong ✗', color=colors['wrong'], alpha=0.8)
            
            ax.set_ylabel(label_en)
            worse_direction = "Higher" if higher_is_worse else "Lower"
            ax.set_title(f'{label_en}\n({label_cn})\n{worse_direction} = more chaotic', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(self.model_names, rotation=45, ha='right')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7)
        
        # 隐藏多余的subplot
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('🔥 Divergence & Chaos Metrics: Correct vs Wrong\n'
                    '(Higher values indicate more chaotic/unstable trajectories)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / 'divergence_metrics_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")
        
        # 图2：发散指标的差异百分比（Wrong - Correct）
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(self.model_names))
        width = 0.1
        
        colors_bar = plt.cm.Set3(np.linspace(0, 1, len(available_metrics)))
        
        for i, (col, label_en, label_cn, higher_is_worse) in enumerate(available_metrics):
            diffs = []
            for model_name in self.model_names:
                correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
                wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
                
                if len(correct_row) > 0 and len(wrong_row) > 0:
                    c_val = correct_row[col].values[0]
                    w_val = wrong_row[col].values[0]
                    if abs(c_val) > 1e-8:
                        diff_pct = (w_val - c_val) / abs(c_val) * 100
                    else:
                        diff_pct = 0
                else:
                    diff_pct = 0
                diffs.append(diff_pct)
            
            offset = (i - len(available_metrics)/2) * width
            bars = ax.bar(x + offset, diffs, width, label=f'{label_en}', color=colors_bar[i], alpha=0.85)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Difference % (Wrong - Correct) / Correct', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('🔥 Divergence Metrics: How Much MORE Chaotic are Wrong Answers?\n'
                    'Positive = Wrong answers show MORE divergence/chaos\n'
                    '(Hypothesis: Wrong answers should have higher divergence indicators)',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names, fontsize=11)
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加背景色区分区域
        ax.axhspan(0, ax.get_ylim()[1], alpha=0.1, color='red', label='More chaotic')
        ax.axhspan(ax.get_ylim()[0], 0, alpha=0.1, color='green', label='Less chaotic')
        
        plt.tight_layout()
        save_path = output_dir / 'divergence_metrics_difference.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")
        
        # 图3：发散指标雷达图
        self._visualize_divergence_radar(stats_df, output_dir, available_metrics)
        
        # 图4：发散指标综合得分
        self._visualize_divergence_composite_score(stats_df, output_dir, available_metrics)
    
    def _visualize_divergence_radar(self, stats_df: pd.DataFrame, output_dir: Path, metrics: list):
        """
        雷达图对比每个模型在正确vs错误时的发散指标
        """
        from math import pi
        
        n_metrics = len(metrics)
        angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
        angles += angles[:1]  # 闭合
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(polar=True))
        axes = axes.flatten()
        
        for idx, model_name in enumerate(self.model_names):
            ax = axes[idx]
            
            correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
            wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
            
            if len(correct_row) == 0 or len(wrong_row) == 0:
                continue
            
            # 获取指标值并归一化
            correct_vals = []
            wrong_vals = []
            
            for col, _, _, _ in metrics:
                c_val = correct_row[col].values[0]
                w_val = wrong_row[col].values[0]
                max_val = max(abs(c_val), abs(w_val), 1e-8)
                correct_vals.append(c_val / max_val)
                wrong_vals.append(w_val / max_val)
            
            correct_vals += correct_vals[:1]
            wrong_vals += wrong_vals[:1]
            
            # 绘制
            ax.plot(angles, correct_vals, 'o-', linewidth=2, color='#27ae60', label='Correct ✓')
            ax.fill(angles, correct_vals, alpha=0.25, color='#27ae60')
            ax.plot(angles, wrong_vals, 'o-', linewidth=2, color='#e74c3c', label='Wrong ✗')
            ax.fill(angles, wrong_vals, alpha=0.25, color='#e74c3c')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m[1][:10] for m in metrics], fontsize=8)
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold', pad=20)
            ax.legend(loc='upper right', fontsize=8)
        
        plt.suptitle('🕸️ Divergence Metrics Radar: Correct vs Wrong\n'
                    '(Larger area = more chaotic behavior)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / 'divergence_radar.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")
    
    def _visualize_divergence_composite_score(self, stats_df: pd.DataFrame, output_dir: Path, metrics: list):
        """
        计算并可视化发散综合得分
        综合得分 = 所有发散指标归一化后的平均值
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 计算综合得分
        composite_scores = {'correct': {}, 'wrong': {}}
        
        for model_name in self.model_names:
            for category in ['correct', 'wrong']:
                row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == category)]
                if len(row) == 0:
                    continue
                
                # 对每个指标，计算相对于所有模型的归一化值
                scores = []
                for col, _, _, _ in metrics:
                    val = row[col].values[0]
                    # 取所有模型该指标的最大值进行归一化
                    max_val = stats_df[col].max()
                    if max_val > 1e-8:
                        scores.append(val / max_val)
                
                composite_scores[category][model_name] = np.mean(scores) if scores else 0
        
        # 图1：综合得分柱状图
        ax = axes[0]
        x = np.arange(len(self.model_names))
        width = 0.35
        
        correct_scores = [composite_scores['correct'].get(m, 0) for m in self.model_names]
        wrong_scores = [composite_scores['wrong'].get(m, 0) for m in self.model_names]
        
        bars1 = ax.bar(x - width/2, correct_scores, width, label='Correct ✓', color='#27ae60', alpha=0.8)
        bars2 = ax.bar(x + width/2, wrong_scores, width, label='Wrong ✗', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Composite Divergence Score', fontsize=12)
        ax.set_title('📊 Composite Chaos Score\n(Higher = more chaotic)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in list(bars1) + list(bars2):
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        # 图2：得分增加百分比
        ax = axes[1]
        
        diffs = []
        for model_name in self.model_names:
            c_score = composite_scores['correct'].get(model_name, 0)
            w_score = composite_scores['wrong'].get(model_name, 0)
            if c_score > 1e-8:
                diff_pct = (w_score - c_score) / c_score * 100
            else:
                diff_pct = 0
            diffs.append(diff_pct)
        
        colors_bar = ['#e74c3c' if d > 0 else '#27ae60' for d in diffs]
        bars = ax.bar(self.model_names, diffs, color=colors_bar, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Score Increase % (Wrong - Correct)', fontsize=12)
        ax.set_title('📈 Chaos Score Increase When Wrong\n'
                    '(Positive = Wrong is MORE chaotic than Correct)',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, diff in zip(bars, diffs):
            height = bar.get_height()
            ax.annotate(f'{diff:+.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.suptitle('🎯 Divergence Composite Analysis\n'
                    'Hypothesis: Wrong answers should show higher divergence scores',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / 'divergence_composite_score.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")
        
        # 保存发散指标详细统计
        divergence_stats = stats_df[['model', 'category'] + [m[0] for m in metrics]].copy()
        divergence_stats['composite_score'] = divergence_stats.apply(
            lambda row: composite_scores[row['category']].get(row['model'], 0), axis=1
        )
        save_path = output_dir / 'divergence_statistics.csv'
        divergence_stats.to_csv(save_path, index=False)
        print(f"  Saved {save_path}")
        
        # 打印关键发现
        print("\n" + "=" * 50)
        print("🔥 DIVERGENCE ANALYSIS KEY FINDINGS:")
        print("=" * 50)
        
        for model_name in self.model_names:
            c_score = composite_scores['correct'].get(model_name, 0)
            w_score = composite_scores['wrong'].get(model_name, 0)
            diff = (w_score - c_score) / c_score * 100 if c_score > 1e-8 else 0
            
            emoji = "⚠️" if diff > 10 else ("✅" if diff < 0 else "➡️")
            print(f"\n{emoji} {model_name}:")
            print(f"   Correct chaos score: {c_score:.4f}")
            print(f"   Wrong chaos score:   {w_score:.4f}")
            print(f"   Change: {diff:+.1f}%")
            
            if diff > 10:
                print(f"   → Wrong answers ARE significantly more chaotic!")
            elif diff < -5:
                print(f"   → Unexpected: Wrong answers are LESS chaotic")
            else:
                print(f"   → Similar chaos levels between correct/wrong")
        
        # 添加僵化指标可视化
        self._visualize_stagnation_metrics(stats_df, output_dir)
    
    def _visualize_stagnation_metrics(self, stats_df: pd.DataFrame, output_dir: Path):
        """
        可视化僵化/停滞指标 (Stagnation/Rigidity Metrics)
        验证假设：错误时模型"卡住"，轨迹变短、变平、停滞
        """
        print("\nVisualizing stagnation/rigidity metrics...")
        
        # 僵化相关指标及其解释
        # lower_is_stagnant: True表示值越低越僵化
        stagnation_metrics = [
            ('trajectory_span_mean', 'Trajectory Span', '轨迹跨度', True),  # 低=僵化
            ('effective_movement_ratio_mean', 'Effective Movement', '有效移动比', False),  # 高=直线僵化
            ('step_decay_rate_mean', 'Step Decay Rate', '步长衰减率', False),  # 高=后期减速
            ('final_stagnation_ratio_mean', 'Final Stagnation', '末期停滞度', True),  # 低=末期卡住
            ('stagnant_steps_ratio_mean', 'Stagnant Steps %', '停滞步比例', False),  # 高=更多停滞
            ('oscillation_score_mean', 'Oscillation', '振荡度', False),  # 高=来回振荡
            ('position_clustering_std_mean', 'Position Spread', '位置分散度', True),  # 低=聚集/僵化
            ('dynamic_range_mean', 'Dynamic Range', '动态范围', True),  # 低=僵硬
            ('first_last_step_ratio_mean', 'First/Last Step', '首尾步长比', False),  # 高=后期减速
            ('progress_stagnation_mean', 'Progress Stagnation', '进展停滞', True),  # 低=接近后卡住
        ]
        
        # 检查哪些指标可用
        available_metrics = [m for m in stagnation_metrics if m[0] in stats_df.columns]
        
        if not available_metrics:
            print("  No stagnation metrics available")
            return
        
        n_metrics = len(available_metrics)
        cols = 5
        rows = (n_metrics + cols - 1) // cols
        
        # 图1：僵化指标对比
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        colors = {'correct': '#27ae60', 'wrong': '#e74c3c'}
        
        for idx, (col, label_en, label_cn, lower_is_stagnant) in enumerate(available_metrics):
            ax = axes[idx]
            x = np.arange(len(self.model_names))
            width = 0.35
            
            correct_vals = []
            wrong_vals = []
            
            for model_name in self.model_names:
                correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
                wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
                
                c_val = correct_row[col].values[0] if len(correct_row) > 0 else 0
                w_val = wrong_row[col].values[0] if len(wrong_row) > 0 else 0
                correct_vals.append(c_val)
                wrong_vals.append(w_val)
            
            bars1 = ax.bar(x - width/2, correct_vals, width, label='Correct ✓', color=colors['correct'], alpha=0.8)
            bars2 = ax.bar(x + width/2, wrong_vals, width, label='Wrong ✗', color=colors['wrong'], alpha=0.8)
            
            ax.set_ylabel(label_en, fontsize=9)
            stagnant_indicator = "Lower" if lower_is_stagnant else "Higher"
            ax.set_title(f'{label_en}\n({label_cn})\n{stagnant_indicator} = stagnation', fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(self.model_names, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 隐藏多余的subplot
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('🧊 Stagnation/Rigidity Metrics: Correct vs Wrong\n'
                    '(Testing hypothesis: Wrong answers show "frozen" trajectories)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / 'stagnation_metrics_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")
        
        # 图2：僵化指标差异百分比
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(len(self.model_names))
        width = 0.08
        
        colors_bar = plt.cm.tab10(np.linspace(0, 1, len(available_metrics)))
        
        for i, (col, label_en, label_cn, lower_is_stagnant) in enumerate(available_metrics):
            diffs = []
            for model_name in self.model_names:
                correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
                wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
                
                if len(correct_row) > 0 and len(wrong_row) > 0:
                    c_val = correct_row[col].values[0]
                    w_val = wrong_row[col].values[0]
                    if abs(c_val) > 1e-8:
                        diff_pct = (w_val - c_val) / abs(c_val) * 100
                    else:
                        diff_pct = 0
                else:
                    diff_pct = 0
                diffs.append(diff_pct)
            
            offset = (i - len(available_metrics)/2) * width
            ax.bar(x + offset, diffs, width, label=f'{label_en}', color=colors_bar[i], alpha=0.85)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Difference % (Wrong - Correct) / Correct', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('🧊 Stagnation Metrics: Are Wrong Answers More "Frozen"?\n'
                    'Negative span/spread = more stagnant | Positive decay/stagnant_steps = more stagnant',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names, fontsize=11)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = output_dir / 'stagnation_metrics_difference.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")
        
        # 图3：综合僵化得分
        self._visualize_stagnation_composite(stats_df, output_dir, available_metrics)
        
        # 保存僵化指标统计
        stagnation_stats = stats_df[['model', 'category'] + [m[0] for m in available_metrics]].copy()
        save_path = output_dir / 'stagnation_statistics.csv'
        stagnation_stats.to_csv(save_path, index=False)
        print(f"  Saved {save_path}")
    
    def _visualize_stagnation_composite(self, stats_df: pd.DataFrame, output_dir: Path, metrics: list):
        """
        计算并可视化综合僵化得分
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 计算综合僵化得分（需要对指标方向进行标准化）
        stagnation_scores = {'correct': {}, 'wrong': {}}
        
        for model_name in self.model_names:
            for category in ['correct', 'wrong']:
                row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == category)]
                if len(row) == 0:
                    continue
                
                # 对每个指标计算僵化得分（归一化到0-1，越高越僵化）
                scores = []
                for col, _, _, lower_is_stagnant in metrics:
                    val = row[col].values[0]
                    # 获取所有值的范围
                    all_vals = stats_df[col].values
                    min_val, max_val = all_vals.min(), all_vals.max()
                    
                    if max_val - min_val > 1e-8:
                        # 归一化到0-1
                        normalized = (val - min_val) / (max_val - min_val)
                        # 如果低值表示僵化，则翻转
                        if lower_is_stagnant:
                            normalized = 1 - normalized
                        scores.append(normalized)
                
                stagnation_scores[category][model_name] = np.mean(scores) if scores else 0
        
        # 图1：综合僵化得分
        ax = axes[0]
        x = np.arange(len(self.model_names))
        width = 0.35
        
        correct_scores = [stagnation_scores['correct'].get(m, 0) for m in self.model_names]
        wrong_scores = [stagnation_scores['wrong'].get(m, 0) for m in self.model_names]
        
        bars1 = ax.bar(x - width/2, correct_scores, width, label='Correct ✓', color='#27ae60', alpha=0.8)
        bars2 = ax.bar(x + width/2, wrong_scores, width, label='Wrong ✗', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Composite Stagnation Score', fontsize=12)
        ax.set_title('🧊 Composite Stagnation Score\n(Higher = more frozen/stuck)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in list(bars1) + list(bars2):
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        # 图2：僵化得分变化百分比
        ax = axes[1]
        
        diffs = []
        for model_name in self.model_names:
            c_score = stagnation_scores['correct'].get(model_name, 0)
            w_score = stagnation_scores['wrong'].get(model_name, 0)
            if c_score > 1e-8:
                diff_pct = (w_score - c_score) / c_score * 100
            else:
                diff_pct = 0
            diffs.append(diff_pct)
        
        colors_bar = ['#3498db' if d > 0 else '#e67e22' for d in diffs]
        bars = ax.bar(self.model_names, diffs, color=colors_bar, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Stagnation Score Change %', fontsize=12)
        ax.set_title('📈 Stagnation Change: Wrong vs Correct\n'
                    '(Positive = Wrong is MORE frozen)',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, diff in zip(bars, diffs):
            height = bar.get_height()
            ax.annotate(f'{diff:+.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 图3：关键指标的详细对比 - Trajectory Span vs Path Length
        ax = axes[2]
        
        if 'trajectory_span_mean' in stats_df.columns and 'path_length_mean' in stats_df.columns:
            for idx, model_name in enumerate(self.model_names):
                correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
                wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
                
                if len(correct_row) > 0 and len(wrong_row) > 0:
                    # 正确点
                    ax.scatter(correct_row['trajectory_span_mean'].values[0], 
                              correct_row['path_length_mean'].values[0],
                              s=200, marker='o', label=f'{model_name} ✓' if idx == 0 else None,
                              c='#27ae60', alpha=0.8, edgecolors='white', linewidths=2)
                    ax.annotate(f'{model_name}✓', 
                               (correct_row['trajectory_span_mean'].values[0], 
                                correct_row['path_length_mean'].values[0]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                    
                    # 错误点
                    ax.scatter(wrong_row['trajectory_span_mean'].values[0], 
                              wrong_row['path_length_mean'].values[0],
                              s=200, marker='x', label=f'{model_name} ✗' if idx == 0 else None,
                              c='#e74c3c', alpha=0.8, linewidths=3)
                    ax.annotate(f'{model_name}✗', 
                               (wrong_row['trajectory_span_mean'].values[0], 
                                wrong_row['path_length_mean'].values[0]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                    
                    # 连线
                    ax.plot([correct_row['trajectory_span_mean'].values[0], wrong_row['trajectory_span_mean'].values[0]],
                           [correct_row['path_length_mean'].values[0], wrong_row['path_length_mean'].values[0]],
                           'k--', alpha=0.3)
            
            ax.set_xlabel('Trajectory Span (起点到终点直线距离)', fontsize=11)
            ax.set_ylabel('Path Length (实际走过的路径长度)', fontsize=11)
            ax.set_title('🎯 Span vs Path Length\n'
                        '(Low span + High path = spinning in place\n'
                        'Low span + Low path = frozen/stuck)',
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 添加参考区域
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.axvspan(xlim[0], xlim[0] + (xlim[1]-xlim[0])*0.3, alpha=0.1, color='blue', label='Low span zone')
        
        plt.suptitle('🧊 Stagnation Analysis: Is Model "Frozen" When Wrong?\n'
                    'Hypothesis: Wrong answers show frozen/stuck trajectories',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / 'stagnation_composite_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")
        
        # 打印关键发现
        print("\n" + "=" * 50)
        print("🧊 STAGNATION ANALYSIS KEY FINDINGS:")
        print("=" * 50)
        
        for model_name in self.model_names:
            c_score = stagnation_scores['correct'].get(model_name, 0)
            w_score = stagnation_scores['wrong'].get(model_name, 0)
            diff = (w_score - c_score) / c_score * 100 if c_score > 1e-8 else 0
            
            correct_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'correct')]
            wrong_row = stats_df[(stats_df['model'] == model_name) & (stats_df['category'] == 'wrong')]
            
            emoji = "🧊" if diff > 5 else ("🔥" if diff < -5 else "➡️")
            print(f"\n{emoji} {model_name}:")
            print(f"   Stagnation score: Correct={c_score:.4f}, Wrong={w_score:.4f} ({diff:+.1f}%)")
            
            if len(correct_row) > 0 and len(wrong_row) > 0:
                # 关键指标对比
                if 'trajectory_span_mean' in correct_row.columns:
                    c_span = correct_row['trajectory_span_mean'].values[0]
                    w_span = wrong_row['trajectory_span_mean'].values[0]
                    span_diff = (w_span - c_span) / c_span * 100 if c_span > 1e-8 else 0
                    print(f"   Trajectory span: Correct={c_span:.2f}, Wrong={w_span:.2f} ({span_diff:+.1f}%)")
                
                if 'stagnant_steps_ratio_mean' in correct_row.columns:
                    c_stag = correct_row['stagnant_steps_ratio_mean'].values[0]
                    w_stag = wrong_row['stagnant_steps_ratio_mean'].values[0]
                    stag_diff = (w_stag - c_stag) / max(c_stag, 1e-8) * 100
                    print(f"   Stagnant steps: Correct={c_stag:.1%}, Wrong={w_stag:.1%} ({stag_diff:+.1f}%)")
                
                if 'final_stagnation_ratio_mean' in correct_row.columns:
                    c_final = correct_row['final_stagnation_ratio_mean'].values[0]
                    w_final = wrong_row['final_stagnation_ratio_mean'].values[0]
                    final_diff = (w_final - c_final) / c_final * 100 if c_final > 1e-8 else 0
                    print(f"   Final stagnation: Correct={c_final:.3f}, Wrong={w_final:.3f} ({final_diff:+.1f}%)")
            
            if diff > 5:
                print(f"   → ✅ CONFIRMED: Wrong answers show MORE stagnation/freezing!")
            elif diff < -5:
                print(f"   → ❌ REJECTED: Wrong answers are LESS stagnant (more active)")
            else:
                print(f"   → ➡️ INCONCLUSIVE: Similar stagnation levels")

    def _visualize_correct_vs_wrong_trajectories(self, output_dir: Path, num_samples: int = 5):
        """
        为每个模型可视化正确和错误题目的轨迹对比
        """
        print("\nGenerating correct vs wrong trajectory visualizations...")
        
        trajectory_dir = output_dir / 'trajectories'
        trajectory_dir.mkdir(exist_ok=True)
        
        model_colors = {'codi': '#e74c3c', 'codi_sircl': '#27ae60', 
                       'simcon': '#f39c12', 'sircl': '#3498db'}
        
        np.random.seed(42)
        
        for model_name in self.model_names:
            correct_mask = self.correct_masks[model_name]
            correct_indices = np.where(correct_mask)[0]
            wrong_indices = np.where(~correct_mask)[0]
            
            if len(wrong_indices) == 0:
                continue
            
            # 随机选择样本
            n_correct = min(num_samples, len(correct_indices))
            n_wrong = min(num_samples, len(wrong_indices))
            
            selected_correct = np.random.choice(correct_indices, n_correct, replace=False)
            selected_wrong = np.random.choice(wrong_indices, n_wrong, replace=False)
            
            model_dir = trajectory_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # 为每道题生成对比图
            all_selected = list(zip(selected_correct, ['correct'] * n_correct)) + \
                          list(zip(selected_wrong, ['wrong'] * n_wrong))
            
            for idx, category in all_selected:
                latents = self.latents[model_name][idx]  # (7, dim)
                
                # PCA降维
                pca = PCA(n_components=2)
                embedded_2d = pca.fit_transform(latents)
                
                pca_3d = PCA(n_components=3)
                embedded_3d = pca_3d.fit_transform(latents)
                
                # 计算统计
                dist_to_final = np.mean([euclidean(latents[i], latents[-1]) for i in range(6)])
                path_length = sum(euclidean(latents[i], latents[i+1]) for i in range(6))
                cos_sim = np.mean([1 - cosine(latents[i], latents[-1]) for i in range(6)])
                
                # 2D图
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                color = '#27ae60' if category == 'correct' else '#e74c3c'
                marker_color = model_colors.get(model_name, 'blue')
                
                # 左图：原始轨迹
                ax = axes[0]
                ax.plot(embedded_2d[:, 0], embedded_2d[:, 1], color=marker_color, 
                       linewidth=3, alpha=0.8)
                for i in range(7):
                    size = 200 if i == 6 else 80
                    marker = '*' if i == 6 else 'o'
                    ax.scatter(embedded_2d[i, 0], embedded_2d[i, 1], 
                              c=[marker_color], s=size, marker=marker,
                              edgecolors='white', linewidths=2, zorder=5)
                    ax.annotate(f'{i+1}', (embedded_2d[i, 0], embedded_2d[i, 1]),
                               xytext=(5, 5), textcoords='offset points', fontsize=10)
                
                ax.set_xlabel('PCA Dimension 1')
                ax.set_ylabel('PCA Dimension 2')
                ax.set_title('Trajectory')
                ax.grid(True, alpha=0.3)
                
                # 右图：以Final为中心
                ax = axes[1]
                rel = embedded_2d - embedded_2d[-1]
                
                ax.plot(rel[:, 0], rel[:, 1], color=marker_color, linewidth=3, alpha=0.8)
                for i in range(7):
                    size = 200 if i == 6 else 80
                    marker = '*' if i == 6 else 'o'
                    ax.scatter(rel[i, 0], rel[i, 1], c=[marker_color], s=size, marker=marker,
                              edgecolors='white', linewidths=2, zorder=5)
                    ax.annotate(f'{i+1}', (rel[i, 0], rel[i, 1]),
                               xytext=(5, 5), textcoords='offset points', fontsize=10)
                
                # 参考圆
                max_dist = np.max(np.abs(rel)) * 1.2
                for r in np.linspace(max_dist/4, max_dist, 4):
                    circle = plt.Circle((0, 0), r, fill=False, color='gray', linestyle=':', alpha=0.4)
                    ax.add_patch(circle)
                
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
                ax.set_aspect('equal')
                ax.set_xlabel('Relative Dimension 1')
                ax.set_ylabel('Relative Dimension 2')
                ax.set_title('Centered at Final Token')
                ax.grid(True, alpha=0.3)
                
                # 统计信息
                status_emoji = '✓' if category == 'correct' else '✗'
                stats_text = (
                    f"Statistics:\n"
                    f"• Avg dist to final: {dist_to_final:.2f}\n"
                    f"• Path length: {path_length:.2f}\n"
                    f"• Cosine sim: {cos_sim:.3f}"
                )
                fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
                
                plt.suptitle(f'{model_name} - Question #{idx} [{category.upper()} {status_emoji}]',
                            fontsize=14, fontweight='bold', color=color)
                plt.tight_layout(rect=[0.1, 0.1, 1, 0.95])
                
                save_path = model_dir / f'q{idx}_{category}_2d.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"  Saved {n_correct + n_wrong} trajectories for {model_name}")
        
        # 生成综合对比图：同一模型的正确vs错误
        self._visualize_correct_vs_wrong_side_by_side(output_dir, num_samples=3)
    
    def _visualize_correct_vs_wrong_side_by_side(self, output_dir: Path, num_samples: int = 3):
        """
        每个模型生成正确vs错误的并排对比图
        """
        print("\nGenerating side-by-side correct vs wrong comparison...")
        
        model_colors = {'codi': '#e74c3c', 'codi_sircl': '#27ae60', 
                       'simcon': '#f39c12', 'sircl': '#3498db'}
        
        np.random.seed(42)
        
        for model_name in self.model_names:
            correct_mask = self.correct_masks[model_name]
            correct_indices = np.where(correct_mask)[0]
            wrong_indices = np.where(~correct_mask)[0]
            
            if len(wrong_indices) == 0 or len(correct_indices) == 0:
                continue
            
            n_samples_actual = min(num_samples, len(correct_indices), len(wrong_indices))
            
            selected_correct = np.random.choice(correct_indices, n_samples_actual, replace=False)
            selected_wrong = np.random.choice(wrong_indices, n_samples_actual, replace=False)
            
            # 创建大图：2行 x n_samples列
            fig, axes = plt.subplots(2, n_samples_actual, figsize=(5*n_samples_actual, 10))
            
            if n_samples_actual == 1:
                axes = axes.reshape(2, 1)
            
            for col, (c_idx, w_idx) in enumerate(zip(selected_correct, selected_wrong)):
                # 正确题目（上行）
                latents_c = self.latents[model_name][c_idx]
                pca = PCA(n_components=2)
                emb_c = pca.fit_transform(latents_c)
                rel_c = emb_c - emb_c[-1]
                
                ax = axes[0, col]
                ax.plot(rel_c[:, 0], rel_c[:, 1], color='#27ae60', linewidth=2.5, alpha=0.8)
                for i in range(7):
                    size = 150 if i == 6 else 60
                    marker = '*' if i == 6 else 'o'
                    ax.scatter(rel_c[i, 0], rel_c[i, 1], c=['#27ae60'], s=size, marker=marker,
                              edgecolors='white', linewidths=1.5, zorder=5)
                
                dist_c = np.mean([euclidean(latents_c[i], latents_c[-1]) for i in range(6)])
                path_c = sum(euclidean(latents_c[i], latents_c[i+1]) for i in range(6))
                
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
                ax.set_aspect('equal')
                ax.set_title(f'Q#{c_idx} ✓ CORRECT\nd={dist_c:.1f}, p={path_c:.1f}', 
                            fontsize=11, color='#27ae60', fontweight='bold')
                ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.set_ylabel('CORRECT ✓', fontsize=14, color='#27ae60', fontweight='bold')
                
                # 错误题目（下行）
                latents_w = self.latents[model_name][w_idx]
                emb_w = pca.fit_transform(latents_w)
                rel_w = emb_w - emb_w[-1]
                
                ax = axes[1, col]
                ax.plot(rel_w[:, 0], rel_w[:, 1], color='#e74c3c', linewidth=2.5, alpha=0.8)
                for i in range(7):
                    size = 150 if i == 6 else 60
                    marker = '*' if i == 6 else 'o'
                    ax.scatter(rel_w[i, 0], rel_w[i, 1], c=['#e74c3c'], s=size, marker=marker,
                              edgecolors='white', linewidths=1.5, zorder=5)
                
                dist_w = np.mean([euclidean(latents_w[i], latents_w[-1]) for i in range(6)])
                path_w = sum(euclidean(latents_w[i], latents_w[i+1]) for i in range(6))
                
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
                ax.set_aspect('equal')
                ax.set_title(f'Q#{w_idx} ✗ WRONG\nd={dist_w:.1f}, p={path_w:.1f}',
                            fontsize=11, color='#e74c3c', fontweight='bold')
                ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.set_ylabel('WRONG ✗', fontsize=14, color='#e74c3c', fontweight='bold')
            
            plt.suptitle(f'{model_name.upper()}: Correct vs Wrong Trajectories\n'
                        f'(Centered at Final Token, d=avg distance, p=path length)',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = output_dir / f'{model_name}_correct_vs_wrong.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {save_path}")
        
        # 生成所有模型的汇总对比
        self._visualize_all_models_correct_vs_wrong_summary(output_dir)
    
    def _visualize_all_models_correct_vs_wrong_summary(self, output_dir: Path):
        """
        生成所有模型正确vs错误的汇总对比图
        """
        print("\nGenerating all models correct vs wrong summary...")
        
        fig, axes = plt.subplots(2, len(self.model_names), figsize=(5*len(self.model_names), 10))
        
        model_colors = {'codi': '#e74c3c', 'codi_sircl': '#27ae60', 
                       'simcon': '#f39c12', 'sircl': '#3498db'}
        
        np.random.seed(123)  # 不同的种子以选择不同的样本
        
        for col, model_name in enumerate(self.model_names):
            correct_mask = self.correct_masks[model_name]
            correct_indices = np.where(correct_mask)[0]
            wrong_indices = np.where(~correct_mask)[0]
            
            if len(wrong_indices) == 0 or len(correct_indices) == 0:
                continue
            
            # 选择一个代表性样本
            c_idx = np.random.choice(correct_indices)
            w_idx = np.random.choice(wrong_indices)
            
            # 正确题目
            latents_c = self.latents[model_name][c_idx]
            pca = PCA(n_components=2)
            emb_c = pca.fit_transform(latents_c)
            rel_c = emb_c - emb_c[-1]
            
            ax = axes[0, col]
            ax.plot(rel_c[:, 0], rel_c[:, 1], color=model_colors.get(model_name, 'blue'),
                   linewidth=2.5, alpha=0.8)
            for i in range(7):
                size = 150 if i == 6 else 60
                marker = '*' if i == 6 else 'o'
                ax.scatter(rel_c[i, 0], rel_c[i, 1], 
                          c=[model_colors.get(model_name, 'blue')], s=size, marker=marker,
                          edgecolors='white', linewidths=1.5, zorder=5)
            
            dist_c = np.mean([euclidean(latents_c[i], latents_c[-1]) for i in range(6)])
            path_c = sum(euclidean(latents_c[i], latents_c[i+1]) for i in range(6))
            
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            ax.set_aspect('equal')
            ax.set_title(f'{model_name}\n✓ Q#{c_idx}\nd={dist_c:.1f}, p={path_c:.1f}',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel('CORRECT ✓', fontsize=14, color='#27ae60', fontweight='bold')
            
            # 错误题目
            latents_w = self.latents[model_name][w_idx]
            emb_w = pca.fit_transform(latents_w)
            rel_w = emb_w - emb_w[-1]
            
            ax = axes[1, col]
            ax.plot(rel_w[:, 0], rel_w[:, 1], color=model_colors.get(model_name, 'blue'),
                   linewidth=2.5, alpha=0.8, linestyle='--')
            for i in range(7):
                size = 150 if i == 6 else 60
                marker = '*' if i == 6 else 's'
                ax.scatter(rel_w[i, 0], rel_w[i, 1],
                          c=[model_colors.get(model_name, 'blue')], s=size, marker=marker,
                          edgecolors='white', linewidths=1.5, zorder=5)
            
            dist_w = np.mean([euclidean(latents_w[i], latents_w[-1]) for i in range(6)])
            path_w = sum(euclidean(latents_w[i], latents_w[i+1]) for i in range(6))
            
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            ax.set_aspect('equal')
            ax.set_title(f'✗ Q#{w_idx}\nd={dist_w:.1f}, p={path_w:.1f}',
                        fontsize=11)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel('WRONG ✗', fontsize=14, color='#e74c3c', fontweight='bold')
        
        plt.suptitle('All Models: Correct vs Wrong Trajectory Comparison\n'
                    '(Centered at Final Token, ★=Final, ●=Intermediate)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / 'all_models_correct_vs_wrong_summary.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Latent Token Analysis')
    parser.add_argument('--results_dir', type=str, 
                       default='/data/yhao/baseline/CODI/results/route',
                       help='Results directory path')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                       help='Dataset name')
    parser.add_argument('--run', type=int, default=0,
                       help='Run index')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                       help='Skip existing visualizations')
    args = parser.parse_args()
    
    analyzer = LatentAnalyzer(args.results_dir)
    analyzer.load_data(dataset=args.dataset, run=args.run)
    
    # 运行主分析（所有模型都答对的题目）
    stats_df = analyzer.run_full_analysis(skip_existing=args.skip_existing)
    
    # 运行正确vs错误分析
    correct_wrong_stats = analyzer.analyze_correct_vs_wrong()
    
    # 打印关键发现
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    # 找出在关键指标上表现最好的模型
    print("\nBest performing model by metric (all correct questions):")
    print(f"  - Highest Cosine Similarity to Final: {stats_df.loc[stats_df['cos_sim_to_final_mean'].idxmax(), 'model']}")
    print(f"  - Lowest Euclidean Distance to Final: {stats_df.loc[stats_df['eucl_dist_to_final_mean'].idxmin(), 'model']}")
    print(f"  - Best Cluster Compactness: {stats_df.loc[stats_df['cluster_compactness_mean'].idxmin(), 'model']}")
    print(f"  - Smoothest Trajectory: {stats_df.loc[stats_df['trajectory_smoothness_mean'].idxmax(), 'model']}")
    print(f"  - Shortest Path Length: {stats_df.loc[stats_df['path_length_mean'].idxmin(), 'model']}")
    print(f"  - Best Convergence Rate: {stats_df.loc[stats_df['convergence_rate_mean'].idxmin(), 'model']}")
    
    # 打印正确vs错误的关键发现
    print("\n" + "-" * 40)
    print("CORRECT vs WRONG KEY INSIGHTS:")
    print("-" * 40)
    
    for model_name in analyzer.model_names:
        correct_row = correct_wrong_stats[(correct_wrong_stats['model'] == model_name) & 
                                          (correct_wrong_stats['category'] == 'correct')]
        wrong_row = correct_wrong_stats[(correct_wrong_stats['model'] == model_name) & 
                                        (correct_wrong_stats['category'] == 'wrong')]
        
        if len(correct_row) > 0 and len(wrong_row) > 0:
            c_dist = correct_row['eucl_dist_to_final_mean'].values[0]
            w_dist = wrong_row['eucl_dist_to_final_mean'].values[0]
            c_path = correct_row['path_length_mean'].values[0]
            w_path = wrong_row['path_length_mean'].values[0]
            
            print(f"\n{model_name}:")
            print(f"  Distance to final: Correct={c_dist:.2f}, Wrong={w_dist:.2f} ({(w_dist-c_dist)/c_dist*100:+.1f}%)")
            print(f"  Path length: Correct={c_path:.2f}, Wrong={w_path:.2f} ({(w_path-c_path)/c_path*100:+.1f}%)")
            
            # 发散指标
            if 'direction_entropy_mean' in correct_row.columns:
                c_entropy = correct_row['direction_entropy_mean'].values[0]
                w_entropy = wrong_row['direction_entropy_mean'].values[0]
                print(f"  Direction entropy: Correct={c_entropy:.3f}, Wrong={w_entropy:.3f} ({(w_entropy-c_entropy)/max(c_entropy,1e-8)*100:+.1f}%)")
            
            if 'lyapunov_estimate_mean' in correct_row.columns:
                c_lyap = correct_row['lyapunov_estimate_mean'].values[0]
                w_lyap = wrong_row['lyapunov_estimate_mean'].values[0]
                print(f"  Lyapunov estimate: Correct={c_lyap:.3f}, Wrong={w_lyap:.3f} ({(w_lyap-c_lyap)/max(abs(c_lyap),1e-8)*100:+.1f}%)")


if __name__ == '__main__':
    main()
