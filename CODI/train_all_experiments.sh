#!/bin/bash
# =============================================================================
# 训练全部模型 - 2个数据集 x 6种模型 = 12个实验
# 每个数据集训练完成后自动测试
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================================"
echo "训练全部模型配置"
echo "============================================================================"
echo ""
echo "数据集: MultiArith, SVAMP"
echo "模型 (6种):"
echo "  1. sim-con+sircl:       decoder=True,  trajectory=True,  factor=0.1"
echo "  2. sim-con+sircl_f005:  decoder=True,  trajectory=True,  factor=0.05"
echo "  3. codi+sircl:          decoder=False, trajectory=True,  factor=0.1"
echo "  4. codi+sircl_f005:     decoder=False, trajectory=True,  factor=0.05"
echo "  5. sim-con:             decoder=True,  trajectory=False"
echo "  6. codi:                decoder=False, trajectory=False"
echo ""
echo "总计: 12个训练实验 + 12个测试"
echo "============================================================================"
echo ""

TOTAL_START=$(date +%s)

# MultiArith (训练6个模型 + 测试)
echo ">>> 开始 MultiArith 训练+测试..."
bash "${SCRIPT_DIR}/multiarith/train_all_models.sh"
echo ""

# SVAMP (训练6个模型 + 测试)
echo ">>> 开始 SVAMP 训练+测试..."
bash "${SCRIPT_DIR}/svamp/train_all_models.sh"
echo ""

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo "============================================================================"
echo "✅ 全部12个实验完成!"
echo "总耗时: ${TOTAL_TIME}s ($(echo "scale=2; ${TOTAL_TIME}/3600" | bc)小时)"
echo "============================================================================"
echo ""
echo "📁 结果文件:"
echo "  - MultiArith: results/multiarith/all_models_summary.csv"
echo "  - SVAMP: results/svamp/all_models_summary.csv"
echo "============================================================================"
