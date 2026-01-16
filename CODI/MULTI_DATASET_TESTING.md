# 多数据集测试指南

本文档说明如何在多个数据集上进行测试，并将结果按数据集分开保存。

## 重要说明

- **自动下载**：数据集会在首次使用时自动通过 HuggingFace 下载，无需手动下载
- **循环顺序**：模型 → 数据集 → 运行次数（每个模型只加载一次，避免重复加载）
- **结果分离**：结果按数据集分目录保存，便于管理和对比

## 目录结构

测试结果将按照以下结构保存：

```
CODI/results/
├── gsm8k/
│   ├── gsm8k.json                      # 预测结果
│   ├── radius_gsm8k.jsonl              # 半径统计
│   ├── accel_gsm8k.jsonl               # 加速度统计
│   ├── action_gsm8k.jsonl              # 作用量统计
│   ├── geodesic_gsm8k.jsonl            # 测地线偏差统计
│   └── model_name_run0_output.txt      # 详细输出
├── svamp/
│   └── ...
├── gsm-hard/
│   └── ...
└── batch_summary_TIMESTAMP.csv          # 批量测试汇总
```

## 支持的数据集

- **gsm8k**: GSM8K 数学问题数据集（增强版）
- **gsm-hard**: GSM8K 困难版本
- **multi-arith**: MultiArith 多步算术推理
- **svamp**: SVAMP 数学推理
- **commonsense**: CommonsenseQA 常识推理

## 1. 网络配置（可选）

数据集会在首次使用时自动下载。如需配置镜像或代理，在 `config.env` 中设置：

```bash
# HuggingFace 镜像（推荐国内使用）
export HF_ENDPOINT="https://hf-mirror.com"

# 如需代理（可选）
export HTTP_PROXY="http://127.0.0.1:3128"
export HTTPS_PROXY="http://127.0.0.1:3128"

# 缓存目录（可选）
export CODI_CACHE_DIR="/data/yhao/sim-con/CODI/cache"
```

## 2. 单次测试（testcopy.sh）

### 基本用法

```bash
cd CODI/scripts

# 使用默认数据集 (gsm8k)
./testcopy.sh

# 指定数据集
./testcopy.sh svamp

# 指定数据集和 checkpoint
./testcopy.sh gsm-hard /path/to/checkpoint
```

### 参数说明

```bash
./testcopy.sh [dataset_name] [checkpoint_dir]
```

- `dataset_name`: 数据集名称（默认: gsm8k）
- `checkpoint_dir`: checkpoint 路径（可选，默认从 config.env 读取）

### 示例

```bash
# 在 SVAMP 数据集上测试
./testcopy.sh svamp

# 在 GSM-Hard 上测试，使用特定 checkpoint
./testcopy.sh gsm-hard "${CODI_SAVE_DIR}/trained/euclidean"
```

## 3. 批量测试（batch_test.sh）

### 基本用法

```bash
cd CODI/scripts

# 测试所有模型在 gsm8k 上（默认）
./batch_test.sh

# 在多个数据集上测试
./batch_test.sh -d "gsm8k svamp gsm-hard"

# 每个模型运行3次
./batch_test.sh -d "gsm8k svamp" -r 3

# 测试指定模型
./batch_test.sh -m "euclidean hyperbolic" -d "gsm8k svamp"
```

### 完整选项

```bash
./batch_test.sh [选项]

选项:
  -r, --runs N          每个模型运行的次数（默认: 1）
  -m, --models LIST     要测试的模型列表，空格分隔（默认: 全部）
  -d, --datasets NAMES  数据集名称列表，空格分隔（默认: gsm8k）
  -o, --output DIR      结果输出目录（默认: $CODI_RESULT_DIR）
  --dry-run             只显示命令，不实际运行
  -h, --help            显示帮助

注意：
  - 数据集会在首次使用时自动下载
  - 循环顺序：模型 → 数据集 → 运行次数（避免重复加载模型）
# 每个模型运行3次
./batch_test.sh --download -d "gsm8k svamp gsm-hard multi-arith"
```

#### 2. 测试特定模型在多个数据集上的表现

```bash
# 测试 euclidean 和 geodesic 模型
./batch_test.sh \
  -m "euclidean geodesic" \
  -d "gsm8k svamp gsm-hard" \
  -r 3
```

#### 3. Dry-run 模式：预览将要执行的命令
# 执行顺序：euclidean(gsm8k→svamp→gsm-hard) → geodesic(gsm8k→svamp→gsm-hard)
./batch_test.sh \
  -m "euclidean geodesic" \
  -d "gsm8k svamp gsm-hard" \
  -r 3
```

#### 2
```bash
./batch_test.sh \
  -d "gsm8k svamp" \
  -o "/custom/output/path" \
  -r 3
```

## 4. 查看结果

### 汇总表格

批量测试会生成 CSV 汇总文件：

```bash
# 查看最新的汇总
cd CODI/results
cat batch_summary_*.csv | column -t -s','
```

汇总包含：
- timestamp: 时间戳
- model: 模型名称
- dataset: 数据集名称
- run_id: 运行编号
- accuracy: 准确率
- total_samples: 总样本数
- correct: 正确数
- elapsed_sec: 耗时（秒）

### 单个数据集结果

```bash
# 查看 GSM8K 结果
cat results/gsm8k/gsm8k.json

# 查看 SVAMP 的详细输出
cat results/svamp/*_output.txt
```

### 统计信息

```bash
# 半径统计
cat results/gsm8k/radius_gsm8k.jsonl | jq '.'

# 加速度统计
cat results/gsm8k/accel_gsm8k.jsonl | jq '.'
```

## 5. 跨数据集对比

### 生成对比报告

```bash
# 查看所有数据集的准确率
for dataset in gsm8k svamp gsm-hard multi-arith; do
    echo "=== $dataset ==="
    cat results/${dataset}/${dataset}.json | jq -r '.ans | length'
done
```

### 使用 Python 分析

```python
import pandas as pd

# 读取汇总 CSV
df = pd.read_csv("results/batch_summary_TIMESTAMP.csv")

# 按数据集和模型分组
summary = df.groupby(['dataset', 'model'])['accuracy'].agg(['mean', 'std', 'count'])
print(summary)

# 可视化
import matplotlib.pyplot as plt
summary['mean'].unstack().plot(kind='bar', figsize=(12, 6))
plt.title("Model Performance Across Datasets")
plt.ylabel("Accuracy")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("results/performance_comparison.png")
```

## 6. 故障排查

### 数据集下载失败
（在 `config.env` 中）：
   ```bash
   echo $HF_ENDPOINT
   echo $HTTP_PROXY
   ```

2. **切换镜像站**：
   ```bash
   export HF_ENDPOINT="https://hf-mirror.com"
   ```

3. **启用代理**：
   ```bash
   export HTTP_PROXY="http://127.0.0.1:3128"
   export HTTPS_PROXY="http://127.0.0.1:3128"
   ```

4. **检查缓存目录**：
   ```bash
   echo $CODI_CACHE_DIR
   ls -la $CODI_CACHE_DIR
   python download_datasets.py gsm8k  # 单独下载出问题的数据集
   ```

### 测试失败

1. **查看错误日志**：
   ```bash
   cat results/batch_test_TIMESTAMP.log
   cat results/gsm8k/model_name_run0_error.txt
   ```

2. **检查 checkpoint 路径**：
   ```bash
   ls ${CODI_SAVE_DIR}/trained/
   ```

3. **调试单个测试**：
   ```bash
   # 使用 dry-run 查看命令
   ./batch_test.sh --dry-run -d gsm8k
   
   # 手动运行单个测试
   ./testcopy.sh gsm8k /path/to/checkpoint
   ```

## 7. 最佳实践

### 完整测试流程

```bash和网络（可选）
cd CODI
source config.env
# 如需配置镜像：export HF_ENDPOINT="https://hf-mirror.com"

# Step 2: 直接运行批量测试（数据集会自动下载）
cd scripts
./batch_test.sh \
  -d "gsm8k svamp gsm-hard multi-arith" \
  -r 3

# Step 3
# Step 4: 查看结果
cat ../results/batch_summary_*.csv | tail -20 | column -t -s','
```

### 增量测试

如果已经测试了部分数据集，可以只测试新的：

```bash
# 已测试 gsm8k，现在添加 svamp 和 multi-arith
./batch_test.sh -d "svamp multi-arith" -r 3
```

### 保存实验配置

```bash
# 创建实验记录
cat > experiment_config.txt << EOF
Date: $(date)
Models: euclidean geodesic hyperbolic
Datasets: gsm8k svamp gsm-hard
Runs per config: 3
Checkpoint: ${CODI_CKPT_DIR}
EOF
```

## 8. 环境变量配置

确保在 `config.env` 中正确配置：

```bash
# 结果目录
export CODI_RESULT_DIR="/data/yhao/baseline/CODI/results"

# 缓存目录（数据集下载）
export CODI_CACHE_DIR="/data/yhao/sim-con/CODI/cache"

# HuggingFace 配置
export HF_ENDPOINT="https://hf-mirror.com"

# 代理（如需要）
# export HTTP_PROXY="http://127.0.0.1:3128"
# export HTTPS_PROXY="http://127.0.0.1:3128"
```

## 总结

- 使用 `download_datasets.py` 提前下载数据集，支持镜像和代理
- `testcopy.sh` 用于单次测试，支持指定数据集
- `batch_test.sh` 用于批量测试，支持多模型、多数据集、多次运行
- 结果按数据集分目录保存，便于管理和对比
- 使用 CSV 汇总文件进行跨数据集、跨模型的性能对比
