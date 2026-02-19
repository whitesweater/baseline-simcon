"""
消融实验可视化脚本
用于绘制 GSM8K 上的参数消融实验结果
"""
import matplotlib.pyplot as plt
import numpy as np
from color_config import COLOR_LIST, LINE_COLOR, BAR_EDGE_COLOR, GRID_ALPHA, BAR_ALPHA


def plot_gsm8k_ablation(x_values, acc_values, x_label, title, save_path):
	"""
	通用画图函数：横坐标为参数值，纵坐标为GSM8K准确率。
	x_values: list[str|float|int] 横坐标参数值
	acc_values: list[float] GSM8K准确率
	x_label: str 横坐标标签
	title: str 图标题
	save_path: str 图片保存路径
	"""
	plt.style.use('seaborn-v0_8-colorblind')
	fig, ax = plt.subplots(figsize=(7, 5))
	# 自动扩展颜色
	bar_colors = COLOR_LIST * ((len(x_values) + len(COLOR_LIST) - 1) // len(COLOR_LIST))
	bars = ax.bar([str(x) for x in x_values], acc_values, color=bar_colors[:len(x_values)], 
	              width=0.6, edgecolor=BAR_EDGE_COLOR, linewidth=1.2)
	for bar in bars:
		height = bar.get_height()
		ax.annotate(f'{height:.2f}%',
					xy=(bar.get_x() + bar.get_width() / 2, height),
					xytext=(0, 5),
					textcoords="offset points",
					ha='center', va='bottom', fontsize=12, fontweight='bold')
	ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')
	ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
	ax.set_ylim(50, max(acc_values) + 5)
	ax.yaxis.grid(True, linestyle='--', alpha=GRID_ALPHA)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.set_xticklabels([str(x) for x in x_values], fontsize=13, fontweight='bold')
	plt.tight_layout()
	plt.savefig(save_path, dpi=200)
	plt.close()


def plot_gsm8k_ablation_line(x_values, acc_values, x_label, title, save_path):
	"""
	折线图版本：横坐标为参数值，纵坐标为GSM8K准确率。
	"""
	plt.style.use('seaborn-v0_8-colorblind')
	fig, ax = plt.subplots(figsize=(7, 5))
	ax.plot(x_values, acc_values, color=LINE_COLOR, marker='o', markersize=7, linewidth=2.5)
	for x, y in zip(x_values, acc_values):
		ax.annotate(f'{y:.2f}%',
					xy=(x, y),
					xytext=(0, 6),
					textcoords="offset points",
					ha='center', va='bottom', fontsize=12, fontweight='bold')
	ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')
	ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
	ax.set_ylim(50, max(acc_values) + 10)
	ax.yaxis.grid(True, linestyle='--', alpha=GRID_ALPHA)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.set_xticks(x_values)
	ax.set_xticklabels([str(x) for x in x_values], fontsize=13, fontweight='bold')
	plt.tight_layout()
	plt.savefig(save_path, dpi=200)
	plt.close()


def plot_gsm8k_ablation_combo(x_values, acc_values, x_label, title, save_path):
	"""
	柱状图 + 折线图叠加版本：横坐标为参数值，纵坐标为GSM8K准确率。
	"""
	plt.style.use('seaborn-v0_8-colorblind')
	fig, ax = plt.subplots(figsize=(7, 5))
	bar_colors = COLOR_LIST * ((len(x_values) + len(COLOR_LIST) - 1) // len(COLOR_LIST))
	bar_labels = [str(x) for x in x_values]
	# 柱状图
	bar_container = ax.bar(
		bar_labels,
		acc_values,
		color=bar_colors[:len(x_values)],
		width=0.6,
		edgecolor=BAR_EDGE_COLOR,
		linewidth=1.2,
		alpha=BAR_ALPHA
	)
	# 折线图（叠加）
	ax.plot(bar_labels, acc_values, color=LINE_COLOR, marker='o', markersize=6, linewidth=2.0)
	# 数值标注
	for bar in bar_container:
		height = bar.get_height()
		ax.annotate(f'{height:.2f}%',
					xy=(bar.get_x() + bar.get_width() / 2, height),
					xytext=(0, 5),
					textcoords="offset points",
					ha='center', va='bottom', fontsize=12, fontweight='bold')
	ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')
	ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
	distance = 0.5 * (max(acc_values) - min(acc_values))
	ax.set_ylim(min(acc_values)-distance, max(acc_values) + distance)
	ax.yaxis.grid(True, linestyle='--', alpha=GRID_ALPHA)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.set_xticklabels(bar_labels, fontsize=13, fontweight='bold')
	plt.tight_layout()
	plt.savefig(save_path, dpi=200)
	plt.close()


if __name__ == "__main__":
	# 消融实验1：trajectory_loss_factor消融
	loss_factors = [0.01, 0.05, 0.075, 0.1, 0.2]
	gsm8k_acc1 = [48.07, 52.99, 54.06, 56.10, 50.95]
	
	# plot_gsm8k_ablation(
	# 	loss_factors,
	# 	gsm8k_acc1,
	# 	x_label=r"Loss Weight $\lambda$",
	# 	title="GSM8K Accuracy vs Trajectory Loss Factor",
	# 	save_path="results/gsm8k_vs_loss_factor.png"
	# )
	# plot_gsm8k_ablation_line(
	# 	loss_factors,
	# 	gsm8k_acc1,
	# 	x_label=r"Loss Weight $\lambda$",
	# 	title="GSM8K Accuracy vs Trajectory Loss Factor",
	# 	save_path="results/gsm8k_vs_loss_factor_line.png"
	# )
	plot_gsm8k_ablation_combo(
		loss_factors,
		gsm8k_acc1,
		x_label=r"Loss Weight $\lambda$",
		title="GSM8K Accuracy vs Trajectory Loss Factor",
		save_path="results/gsm8k_vs_loss_factor_combo.png"
	)

	# 消融实验2：trajectory_radius_threshold消融
	thresholds = [1, 2, 8]
	gsm8k_acc2 = [53.83, 56.10, 54.06]
	
	# plot_gsm8k_ablation(
	# 	thresholds,
	# 	gsm8k_acc2,
	# 	x_label="Radius",
	# 	title="GSM8K Accuracy vs Trajectory Radius Threshold",
	# 	save_path="results/gsm8k_vs_radius_threshold.png"
	# )
	# plot_gsm8k_ablation_line(
	# 	thresholds,
	# 	gsm8k_acc2,
	# 	x_label="Radius",
	# 	title="GSM8K Accuracy vs Trajectory Radius Threshold",
	# 	save_path="results/gsm8k_vs_radius_threshold_line.png"
	# )
	plot_gsm8k_ablation_combo(
		thresholds,
		gsm8k_acc2,
		x_label="Radius",
		title="GSM8K Accuracy vs Trajectory Radius Threshold",
		save_path="results/gsm8k_vs_radius_threshold_combo.png"
	)
	
	# 消融实验3：Latent Tokens Number
	latent_tokens = ["6", "16", "32"]
	gsm8k_acc3 = [56.1, 58, 57]
	
	# plot_gsm8k_ablation(
	# 	latent_tokens,
	# 	gsm8k_acc3,
	# 	x_label="Latent Tokens Number",
	# 	title="GSM8K Accuracy vs Latent Tokens Number",
	# 	save_path="results/accuracy_vs_latent_tokens.png"
	# )
	plot_gsm8k_ablation_combo(
		latent_tokens,
		gsm8k_acc3,
		x_label="Latent Tokens Number",
		title="GSM8K Accuracy vs Latent Tokens Number",
		save_path="results/accuracy_vs_latent_tokens_combo.png"
	)
	
	print("✅ 所有消融实验图表已生成！")
