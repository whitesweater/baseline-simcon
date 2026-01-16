# 多数据集测试改进总结

## 核心改进

### 1. 自动下载机制 ✅
- HuggingFace 的 `load_dataset` 会自动下载缺失的数据集
- 无需手动预下载，首次使用时自动完成
- 支持在 `config.env` 配置镜像和代理

### 2. 优化循环顺序 ✅
**改进前**：
```python
for model in models:
    for dataset in datasets:
        for run in runs:
            load_model()  # ❌ 重复加载
            test()
```

**改进后**：
```python
for model in models:
    load_model()  # ✅ 只加载一次
    for dataset in datasets:
        for run in runs:
            test()
```

### 3. 结果分离保存 ✅
```
results/
├── gsm8k/
│   ├── gsm8k.json
│   ├── model1_run0_output.txt
│   └── ...
├── svamp/
│   └── ...
└── batch_summary_TIMESTAMP.csv
```

## 修改的文件

### 1. `scripts/testcopy.sh`
- 支持命令行参数指定数据集：`./testcopy.sh [dataset] [ckpt]`
- 结果按数据集分目录保存

### 2. `scripts/batch_test.sh`
- 调整循环顺序：模型 → 数据集 → 运行次数
- 移除 `--download` 选项（自动下载）
- 更新帮助信息和日志输出
- CSV 汇总包含 `dataset` 列

### 3. `test.py`
- 结果路径改为 `${CODI_RESULT_DIR}/${data_name}/`
- 所有统计文件按数据集分目录

### 4. 新增文档
- `download_datasets.py`：可选的手动下载工具
- `MULTI_DATASET_QUICK_START.md`：快速开始指南
- `MULTI_DATASET_TESTING.md`：完整使用文档

## 使用示例

### 单数据集测试
```bash
./testcopy.sh svamp
```

### 多数据集批量测试
```bash
# 所有模型在多个数据集上测试
./batch_test.sh -d "gsm8k svamp gsm-hard"

# 指定模型，每个运行3次
./batch_test.sh -m "euclidean geodesic" -d "gsm8k svamp" -r 3
```

### 配置镜像（可选）
在 `config.env` 中：
```bash
export HF_ENDPOINT="https://hf-mirror.com"
export HTTP_PROXY="http://127.0.0.1:3128"
```

## 执行流程示例

命令：`./batch_test.sh -m "euclidean geodesic" -d "gsm8k svamp" -r 2`

执行顺序：
1. **euclidean 模型**（加载一次）
   - gsm8k run 0
   - gsm8k run 1
   - svamp run 0
   - svamp run 1
2. **geodesic 模型**（加载一次）
   - gsm8k run 0
   - gsm8k run 1
   - svamp run 0
   - svamp run 1

## 结果查看

```bash
# 汇总表
cat results/batch_summary_*.csv | column -t -s','

# 特定数据集
cat results/gsm8k/gsm8k.json
cat results/svamp/*_output.txt

# 跨数据集对比
for d in gsm8k svamp gsm-hard; do
    echo "=== $d ==="
    grep "accuracy" results/$d/*_output.txt
done
```

## 优势

1. **简化使用**：无需手动下载数据集
2. **提高效率**：避免模型重复加载，节省时间和显存
3. **结果清晰**：按数据集分目录，便于管理和对比
4. **灵活配置**：支持镜像站和代理，适应不同网络环境
5. **批量处理**：一条命令完成多模型、多数据集、多次运行

## 支持的数据集

- gsm8k (zen-E/GSM8k-Aug)
- gsm-hard (juyoung-trl/gsm-hard)
- multi-arith (ChilleD/MultiArith)
- svamp (ChilleD/SVAMP)
- commonsense (zen-E/CommonsenseQA-GPT4omini)

所有数据集在首次使用时自动下载！
