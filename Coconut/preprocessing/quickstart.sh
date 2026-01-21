#!/bin/bash
# 快速开始：数据预处理和测试
# 用法: bash preprocessing/quickstart.sh

set -e

echo "======================================================================"
echo "Coconut 多数据集支持 - 快速开始"
echo "======================================================================"
echo ""

# 步骤 1: 处理一个数据集进行测试
echo "步骤 1: 处理 MultiArith 数据集（用于快速测试）"
echo "----------------------------------------------------------------------"
python preprocessing/multi_dataset.py --dataset multiarith --split train --format both
python preprocessing/multi_dataset.py --dataset multiarith --split test --format both
echo ""

# 步骤 2: 测试数据加载
echo "步骤 2: 测试数据加载功能"
echo "----------------------------------------------------------------------"
python preprocessing/test_dataset_loading.py
echo ""

# 步骤 3: 显示生成的文件
echo "步骤 3: 查看生成的文件"
echo "----------------------------------------------------------------------"
echo "JSON 文件:"
ls -lh data/*.json 2>/dev/null || echo "  (暂无)"
echo ""
echo "iCoT 文件:"
ls -lh data/*.txt 2>/dev/null || echo "  (暂无)"
echo ""

# 步骤 4: 提示后续操作
echo "======================================================================"
echo "快速开始完成！"
echo "======================================================================"
echo ""
echo "后续步骤:"
echo "  1. 处理所有数据集:"
echo "     bash preprocessing/process_all_datasets.sh"
echo ""
echo "  2. 在 GSM-Hard 上训练:"
echo "     python coconut.py --config args/gsm_hard_coconut.yaml"
echo ""
echo "  3. 在 MultiArith 上训练:"
echo "     python coconut.py --config args/multiarith_coconut.yaml"
echo ""
echo "  4. 查看完整文档:"
echo "     cat MULTI_DATASET.md"
echo ""
