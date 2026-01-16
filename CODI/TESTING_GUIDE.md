# CODI 多模型多数据集测试指南

## 核心改进

### 1. 模型只加载一次
使用 `test_multi_dataset.py`，每个模型只加载一次，然后在所有数据集上测试：

```
加载 euclidean 模型
  → 测试 gsm8k (run 0, 1, 2...)
  → 测试 svamp (run 0, 1, 2...)
  → 测试 gsm-hard (run 0, 1, 2...)
卸载模型

加载 geodesic 模型
  → ...
```

### 2. 清晰的结果目录结构
```
results/
├── models/                          # 按模型组织
│   ├── euclidean/
│   │   ├── gsm8k/
│   │   │   ├── run_0/
│   │   │   │   ├── predictions.json # 预测结果
│   │   │   │   ├── metrics.json     # 准确率等指标
│   │   │   │   └── trajectory_stats.json
│   │   │   └── run_1/
│   │   ├── svamp/
│   │   └── model_summary.csv        # 该模型在所有数据集上的汇总
│   └── geodesic/
│
├── datasets/                        # 按数据集组织
│   ├── gsm8k/
│   │   └── all_models.csv           # 所有模型在该数据集上的对比
│   └── svamp/
│
└── summary/                         # 全局汇总
    ├── all_results.csv              # 所有结果
    └── comparison_matrix.csv        # 模型×数据集矩阵
```

### 3. 自动生成汇总报告
运行测试后自动生成：
- `comparison_matrix.csv`: 模型×数据集准确率矩阵
- `all_results.csv`: 所有测试结果的完整记录
- 每个模型/数据集的详细汇总

## 快速开始

### 1. 配置环境

确保 `config.env` 中配置正确：

```bash
# 结果目录
export CODI_RESULT_DIR="/data/yhao/baseline/CODI/results"

# 模型路径
export CODI_LLAMA1B_PATH="/path/to/Llama-3.2-1B-Instruct"

# 训练好的模型目录
export CODI_SAVE_DIR="/data/yhao/baseline/CODI/outputs"
# 模型应放在 ${CODI_SAVE_DIR}/trained/ 下
```

### 2. 运行测试

#### 方法1：使用批量脚本（推荐）

```bash
cd CODI/scripts

# 测试所有模型在所有数据集上
./batch_test_multi.sh

# 测试指定模型
./batch_test_multi.sh -m "euclidean geodesic"

# 测试指定数据集
./batch_test_multi.sh -d "gsm8k svamp"

# 每个数据集运行3次
./batch_test_multi.sh -r 3

# 预览命令（不实际运行）
./batch_test_multi.sh --dry-run
```

#### 方法2：直接使用 Python

```bash
cd CODI

# 单个模型，多个数据集
python test_multi_dataset.py \
    --model_name_or_path /path/to/base/model \
    --ckpt_dir /path/to/trained/euclidean \
    --datasets "gsm8k svamp gsm-hard" \
    --num_runs 3 \
    --lora_init --use_lora True \
    --lora_r 128 --lora_alpha 32 \
    --batch_size 128 \
    --greedy True
```

### 3. 查看结果

#### 使用分析脚本

```bash
cd CODI

# 打印结果摘要
python analyze_results.py

# 只分析指定模型
python analyze_results.py --model euclidean

# 只分析指定数据集
python analyze_results.py --dataset gsm8k

# 生成可视化图表
python analyze_results.py --plot
```

#### 直接查看 CSV

```bash
# 查看对比矩阵
column -t -s',' results/summary/comparison_matrix.csv

# 查看所有结果
cat results/summary/all_results.csv

# 查看特定模型在所有数据集上的表现
cat results/models/euclidean/model_summary.csv

# 查看特定数据集上所有模型的排名
cat results/datasets/gsm8k/all_models.csv
```

## 支持的数据集

| 数据集 | HuggingFace ID | 答案类型 |
|--------|----------------|----------|
| gsm8k | zen-E/GSM8k-Aug | 数字 |
| gsm-hard | juyoung-trl/gsm-hard | 数字 |
| svamp | ChilleD/SVAMP | 数字 |
| multi-arith | ChilleD/MultiArith | 数字 |
| commonsense | zen-E/CommonsenseQA-GPT4omini | 选择题 |

**注意**：数据集会在首次使用时自动下载。

## 文件说明

| 文件 | 说明 |
|------|------|
| `test_multi_dataset.py` | 新的多数据集测试脚本（模型只加载一次） |
| `scripts/batch_test_multi.sh` | 批量测试脚本 |
| `analyze_results.py` | 结果分析和可视化 |
| `test.py` | 原始单数据集测试（保留兼容性） |

## 示例输出

### 对比矩阵
```
model         gsm8k_mean  svamp_mean  gsm-hard_mean  AVG
euclidean     0.7234      0.6821      0.4532         0.6196
geodesic      0.7156      0.6745      0.4423         0.6108
hyperbolic    0.7089      0.6698      0.4356         0.6048
```

### 数据集排名
```
【GSM8K】
  🥇 1. euclidean    : 72.34%
  🥈 2. geodesic     : 71.56%
  🥉 3. hyperbolic   : 70.89%
```

### 可视化图表
- `plots/heatmap.png`: 准确率热力图
- `plots/model_comparison.png`: 模型整体对比
- `plots/per_dataset_comparison.png`: 各数据集模型对比

## 网络配置

如需配置 HuggingFace 镜像或代理，在 `config.env` 中添加：

```bash
# HuggingFace 镜像（国内推荐）
export HF_ENDPOINT="https://hf-mirror.com"

# 代理（可选）
export HTTP_PROXY="http://127.0.0.1:3128"
export HTTPS_PROXY="http://127.0.0.1:3128"
```

## 常见问题

### Q: 如何添加新模型进行测试？
将训练好的 checkpoint 放到 `${CODI_SAVE_DIR}/trained/` 目录下，例如：
```
outputs/trained/my_new_model/
├── model.safetensors
└── config.json
```

### Q: 如何只测试部分数据集？
```bash
./batch_test_multi.sh -d "gsm8k svamp"
```

### Q: 如何指定结果保存目录？
```bash
./batch_test_multi.sh -o /custom/results/dir
```

### Q: 测试失败如何排查？
1. 查看日志：`cat results/batch_test_*.log`
2. 检查单个模型的错误：`cat results/models/*/run_*/`

## 与旧脚本的区别

| 特性 | 旧脚本 (test.py + batch_test.sh) | 新脚本 (test_multi_dataset.py) |
|------|----------------------------------|-------------------------------|
| 模型加载 | 每次测试都重新加载 | 每个模型只加载一次 |
| 循环顺序 | bash 控制，低效 | Python 控制，高效 |
| 结果组织 | 扁平结构 | 清晰的层级结构 |
| 汇总报告 | 手动生成 | 自动生成 |
| 可视化 | 需要单独脚本 | 集成在 analyze_results.py |

**建议**：新项目使用 `test_multi_dataset.py` 和 `batch_test_multi.sh`。
