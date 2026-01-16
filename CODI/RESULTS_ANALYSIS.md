# 测试结果分析工具集

这个工具集用于统计、追踪和可视化 `test.py` 的运行结果。

## 📊 数据结构

```
每次运行 test.py 生成 N=11 个 batch（代表 11 个不同配置）
运行 M 次后，总共有 M × 11 个 batch

batch 0-10   → Run 0, Config 0-10
batch 11-21  → Run 1, Config 0-10
batch 22-32  → Run 2, Config 0-10
...
```

**两种维度的分析**：
1. **同一次运行内**：11 个配置的对比（纵向对比）
2. **跨 M 次运行**：相同配置的横向对比（检验稳定性/变化趋势）

## 🚀 快速开始

### 1. 快速查看（推荐日常使用）
```bash
# 查看所有运行结果
python quick_compare.py

# 只查看第 0 次运行
python quick_compare.py --run 0

# 显示跨运行对比表
python quick_compare.py --cross

# 导出 CSV
python quick_compare.py --csv
```

### 2. 完整分析和可视化
```bash
# 生成所有 CSV 和图表
python visualize_results.py

# 指定每次运行的 batch 数（如果不是 11）
python visualize_results.py -n 8
```

## 📊 输出说明

### CSV 文件（在 `results/` 目录下）

| 文件 | 说明 |
|------|------|
| `*_detailed.csv` | 每个指标的完整数据（包含 run_id, config_id） |
| `*_cross_run.csv` | 跨运行对比表（行=config, 列=run） |
| `summary_all.csv` | 所有配置和运行的汇总表 |

**跨运行对比 CSV 示例**（`accel_cross_run.csv`）：
```csv
config_id,run_0,run_1,run_2,mean_across_runs,std_across_runs
0,28.0,27.5,28.2,27.9,0.35
1,27.8,27.9,27.7,27.8,0.10
...
```

### 图表文件

| 文件 | 说明 |
|------|------|
| `within_run_*.png` | 每次运行内的 11 个配置对比 |
| `cross_run_comparison.png` | 跨运行折线图（每条线=一次运行） |
| `heatmap_config_run.png` | 热力图（行=config, 列=run） |
| `boxplot_config_stability.png` | 箱线图（每个配置在多次运行中的分布） |

## 📈 数据文件

脚本分析以下结果文件：
- `results/accel_gsm8k.jsonl`：加速度指标
- `results/action_gsm8k.jsonl`：作用量指标
- `results/geodesic_gsm8k.jsonl`：测地线偏差指标
- `results/radius_gsm8k.jsonl`：半径指标

> 注：`gsm8k.json` 不被分析

## 🔧 使用建议

### 日常工作流
```bash
# 1. 运行测试（生成 11 个配置的结果）
python test.py

# 2. 快速查看结果
python quick_compare.py

# 3. 生成完整报告
python visualize_results.py
```

### 多次运行工作流
```bash
# 运行 3 次测试
python test.py  # batch 0-10
python test.py  # batch 11-21
python test.py  # batch 22-32

# 查看跨运行对比
python quick_compare.py --cross

# 生成完整分析（包含热力图、箱线图等）
python visualize_results.py
```

### 参数说明
```bash
# 如果每次运行的 batch 数不是 11
python visualize_results.py -n 8
python quick_compare.py -n 8
```

## 📦 依赖

基础依赖（quick_compare.py）：
- pandas
- tabulate

完整功能：
- numpy
- matplotlib
- seaborn

安装：
```bash
pip install pandas tabulate numpy matplotlib seaborn
```

## 💡 自定义

所有脚本支持指定结果目录：
```bash
python quick_compare.py --results_dir /path/to/results
python track_runs.py --results_dir /path/to/results --save
python visualize_results.py --results_dir /path/to/results
```

## 🎯 核心指标解释

| 指标 | 含义 |
|------|------|
| **accel_mean** | 平均加速度（越低越好） |
| **action_mean** | 平均作用量（越低越好） |
| **geodesic_mean** | 平均测地线偏差（越低越好） |
| **radius_mean** | 平均半径（越低越好） |
| **violation_rate** | 违反约束的比例 |
