#!/bin/bash
# 批量处理多个数据集
# 使用方法: bash preprocessing/process_all_datasets.sh

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "批量处理数据集: GSM8K, GSM-Hard, MultiArith, SVAMP"
echo "======================================================================"
echo ""

# 创建数据目录
mkdir -p data

# 1. GSM8K (原始版本)
# echo ">>> 处理 GSM8K..."
# python preprocessing/multi_dataset.py --dataset gsm8k --split train --format both
# python preprocessing/multi_dataset.py --dataset gsm8k --split test --format both
# echo ""

# # 2. GSM8K-Aug (增强版本)
# echo ">>> 处理 GSM8K-Aug..."
# python preprocessing/multi_dataset.py --dataset gsm8k-aug --split train --format both
# python preprocessing/multi_dataset.py --dataset gsm8k-aug --split test --format both
# echo ""

# 3. GSM-Hard
echo ">>> 处理 GSM-Hard..."
python preprocessing/multi_dataset.py --dataset gsm-hard --split train --format both
echo ""

# 4. MultiArith
echo ">>> 处理 MultiArith..."
python preprocessing/multi_dataset.py --dataset multiarith --split train --format both
python preprocessing/multi_dataset.py --dataset multiarith --split test --format both
echo ""

# 5. SVAMP
echo ">>> 处理 SVAMP..."
python preprocessing/multi_dataset.py --dataset svamp --split train --format both
python preprocessing/multi_dataset.py --dataset svamp --split test --format both
python preprocessing/multi_dataset.py --dataset svamp --split all --format both
echo ""

echo "======================================================================"
echo "所有数据集处理完成！"
echo "======================================================================"
echo ""
echo "生成的文件:"
ls -lh data/*.json | tail -20
echo ""
echo "生成的 iCoT 文件:"
ls -lh data/*.txt | tail -20
