# 多数据集测试 - 快速开始

## 核心改进

✅ **自动下载**：数据集首次使用时自动通过 HuggingFace 下载  
✅ **优化循环**：模型 → 数据集 → 运行次数（避免重复加载模型）  
✅ **结果分离**：按数据集分目录保存结果  

## 快速使用

### 1. 配置网络（可选）

编辑 `config.env`：

```bash
# 国内推荐使用镜像
export HF_ENDPOINT="https://hf-mirror.com"

# 如需代理
export HTTP_PROXY="http://127.0.0.1:3128"
export HTTPS_PROXY="http://127.0.0.1:3128"
```

### 2. 单次测试

```bash
cd CODI/scripts

# 默认 gsm8k
./testcopy.sh

# 指定数据集
./testcopy.sh svamp

# 指定数据集和 checkpoint
./testcopy.sh gsm-hard /path/to/checkpoint
```

### 3. 批量测试

```bash
# 在多个数据集上测试所有模型
./batch_test.sh -d "gsm8k svamp gsm-hard"

# 测试指定模型，每个运行3次
./batch_test.sh -m "euclidean geodesic" -d "gsm8k svamp" -r 3

# 预览执行计划
./batch_test.sh -d "gsm8k svamp" --dry-run
```

## 支持的数据集

| 数据集 | 说明 | HF ID |
|--------|------|-------|
| gsm8k | GSM8K 数学问题（增强） | zen-E/GSM8k-Aug |
| gsm-hard | GSM8K 困难版本 | juyoung-trl/gsm-hard |
| multi-arith | 多步算术推理 | ChilleD/MultiArith |
| svamp | SVAMP 数学推理 | ChilleD/SVAMP |
| commonsense | 常识推理 | zen-E/CommonsenseQA-GPT4omini |

## 结果目录结构

```
CODI/results/
├── gsm8k/                              # GSM8K 数据集结果
│   ├── gsm8k.json                      # 预测结果
│   ├── model_name_run0_output.txt      # 详细输出
│   ├── radius_gsm8k.jsonl              # 统计数据
│   └── ...
├── svamp/                              # SVAMP 数据集结果
│   └── ...
└── batch_summary_20260116_123456.csv   # 汇总表
```

## 执行逻辑

批量测试按以下顺序执行（**避免模型重复加载**）：

```
for 模型 in [euclidean, geodesic, hyperbolic]:
    加载模型一次
    for 数据集 in [gsm8k, svamp, gsm-hard]:
        for 运行 in [0, 1, 2]:
            测试并保存结果
    卸载模型
```

示例：`./batch_test.sh -m "euclidean geodesic" -d "gsm8k svamp" -r 2`

```
1. euclidean 模型：
   - gsm8k run 0
   - gsm8k run 1
   - svamp run 0
   - svamp run 1
   
2. geodesic 模型：
   - gsm8k run 0
   - gsm8k run 1
   - svamp run 0
   - svamp run 1
```

## 查看结果

```bash
# 查看汇总
cat results/batch_summary_*.csv | column -t -s','

# 查看特定数据集
cat results/gsm8k/gsm8k.json | jq '.'

# 统计准确率
for d in gsm8k svamp gsm-hard; do
    echo "=== $d ==="
    grep "accuracy" results/$d/*_output.txt | tail -1
done
```

## 常见命令

```bash
# 测试所有模型在 gsm8k 和 svamp 上
./batch_test.sh -d "gsm8k svamp"

# 测试 euclidean 模型在所有数据集上，各运行5次
./batch_test.sh -m euclidean -d "gsm8k svamp gsm-hard multi-arith commonsense" -r 5

# 只测试一个模型一个数据集
./testcopy.sh gsm8k "${CODI_SAVE_DIR}/trained/euclidean"

# 预览测试计划
./batch_test.sh -d "gsm8k svamp gsm-hard" --dry-run
```

## 详细文档

更多高级功能和故障排查，请参考 [MULTI_DATASET_TESTING.md](MULTI_DATASET_TESTING.md)
