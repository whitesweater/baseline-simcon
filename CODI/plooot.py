import matplotlib.pyplot as plt
import numpy as np


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
	# 颜色分别为：(128,216,207), (255,159,159), (255,221,147), (153,200,254), (152,154,202)
	color_list = [
		(128/255, 216/255, 207/255),
		(255/255, 159/255, 159/255),
		(255/255, 221/255, 147/255),
		(153/255, 200/255, 254/255),
		(152/255, 154/255, 202/255)
	]
	# 自动扩展颜色
	bar_colors = color_list * ((len(x_values) + 2) // 3)
	bars = ax.bar([str(x) for x in x_values], acc_values, color=bar_colors[:len(x_values)], width=0.6, edgecolor='black', linewidth=1.2)
	for bar in bars:
		height = bar.get_height()
		ax.annotate(f'{height:.2f}%',
					xy=(bar.get_x() + bar.get_width() / 2, height),
					xytext=(0, 5),
					textcoords="offset points",
					ha='center', va='bottom', fontsize=12, fontweight='bold')
	ax.set_ylabel("GSM8K Accuracy (%)", fontsize=14, fontweight='bold')
	ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
	ax.set_ylim(0, max(acc_values) + 10)
	ax.yaxis.grid(True, linestyle='--', alpha=0.7)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.set_xticklabels([str(x) for x in x_values], fontsize=13, fontweight='bold')
	ax.set_title(title, fontsize=15, fontweight='bold')
	plt.tight_layout()
	plt.savefig(save_path, dpi=200)
	plt.show()


def plot_gsm8k_ablation_line(x_values, acc_values, x_label, title, save_path):
	"""
	折线图版本：横坐标为参数值，纵坐标为GSM8K准确率。
	"""
	plt.style.use('seaborn-v0_8-colorblind')
	fig, ax = plt.subplots(figsize=(7, 5))
	line_color = (152/255, 154/255, 202/255)  # 深紫
	ax.plot(x_values, acc_values, color=line_color, marker='o', markersize=7, linewidth=2.5)
	for x, y in zip(x_values, acc_values):
		ax.annotate(f'{y:.2f}%',
					xy=(x, y),
					xytext=(0, 6),
					textcoords="offset points",
					ha='center', va='bottom', fontsize=12, fontweight='bold')
	ax.set_ylabel("GSM8K Accuracy (%)", fontsize=14, fontweight='bold')
	ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
	ax.set_ylim(0, max(acc_values) + 10)
	ax.yaxis.grid(True, linestyle='--', alpha=0.7)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.set_xticks(x_values)
	ax.set_xticklabels([str(x) for x in x_values], fontsize=13, fontweight='bold')
	ax.set_title(title, fontsize=15, fontweight='bold')
	plt.tight_layout()
	plt.savefig(save_path, dpi=200)
	plt.show()


def plot_gsm8k_ablation_combo(x_values, acc_values, x_label, title, save_path):
	"""
	柱状图 + 折线图叠加版本：横坐标为参数值，纵坐标为GSM8K准确率。
	"""
	plt.style.use('seaborn-v0_8-colorblind')
	fig, ax = plt.subplots(figsize=(7, 5))
	# 颜色分别为：(128,216,207), (255,159,159), (255,221,147), (153,200,254), (152,154,202)
	color_list = [
		(128/255, 216/255, 207/255),
		(255/255, 159/255, 159/255),
		(255/255, 221/255, 147/255),
		(153/255, 200/255, 254/255),
		(152/255, 154/255, 202/255)
	]
	bar_colors = color_list * ((len(x_values) + 2) // 3)
	bar_labels = [str(x) for x in x_values]
	# 柱状图
	bar_container = ax.bar(
		bar_labels,
		acc_values,
		color=bar_colors[:len(x_values)],
		width=0.6,
		edgecolor='black',
		linewidth=1.2,
		alpha=0.75
	)
	# 折线图（叠加）
	line_color = (152/255, 154/255, 202/255)  # 深紫
	ax.plot(bar_labels, acc_values, color=line_color, marker='o', markersize=6, linewidth=2.0)
	# 数值标注
	for bar in bar_container:
		height = bar.get_height()
		ax.annotate(f'{height:.2f}%',
					xy=(bar.get_x() + bar.get_width() / 2, height),
					xytext=(0, 5),
					textcoords="offset points",
					ha='center', va='bottom', fontsize=12, fontweight='bold')
	ax.set_ylabel("GSM8K Accuracy (%)", fontsize=14, fontweight='bold')
	ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
	ax.set_ylim(0, max(acc_values) + 10)
	ax.yaxis.grid(True, linestyle='--', alpha=0.7)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.set_xticklabels(bar_labels, fontsize=13, fontweight='bold')
	ax.set_title(title, fontsize=15, fontweight='bold')
	plt.tight_layout()
	plt.savefig(save_path, dpi=200)
	plt.show()

# 示例1：trajectory_loss_factor消融
loss_factors = [0.01, 0.05, 0.075, 0.1, 0.2]
gsm8k_acc1 = [48.07, 52.99, 54.06, 56.10, 50.95]
plot_gsm8k_ablation(
	loss_factors,
	gsm8k_acc1,
	x_label="Loss Factor",
	title="GSM8K Accuracy vs Trajectory Loss Factor",
	save_path="gsm8k_vs_loss_factor.png"
)
plot_gsm8k_ablation_line(
	loss_factors,
	gsm8k_acc1,
	x_label="Loss Factor",
	title="GSM8K Accuracy vs Trajectory Loss Factor (Line)",
	save_path="gsm8k_vs_loss_factor_line.png"
)
plot_gsm8k_ablation_combo(
	loss_factors,
	gsm8k_acc1,
	x_label="Loss Factor",
	title="GSM8K Accuracy vs Trajectory Loss Factor (Bar+Line)",
	save_path="gsm8k_vs_loss_factor_combo.png"
)

# 示例2：trajectory_radius_threshold消融
thresholds = [1, 2, 8]
gsm8k_acc2 = [53.83, 56.10, 54.06]
plot_gsm8k_ablation(
	thresholds,
	gsm8k_acc2,
	x_label="Trajectory Radius Threshold",
	title="GSM8K Accuracy vs Trajectory Radius Threshold",
	save_path="gsm8k_vs_radius_threshold.png"
)
plot_gsm8k_ablation_line(
	thresholds,
	gsm8k_acc2,
	x_label="Trajectory Radius Threshold",
	title="GSM8K Accuracy vs Trajectory Radius Threshold (Line)",
	save_path="gsm8k_vs_radius_threshold_line.png"
)
plot_gsm8k_ablation_combo(
	thresholds,
	gsm8k_acc2,
	x_label="Trajectory Radius Threshold",
	title="GSM8K Accuracy vs Trajectory Radius Threshold (Bar+Line)",
	save_path="gsm8k_vs_radius_threshold_combo.png"
)
