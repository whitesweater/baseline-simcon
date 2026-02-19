#!/bin/bash
# =============================================================================
# MultiArith + SVAMP 并行训练 + 测试脚本
# =============================================================================
# 在 4 张 GPU 上并行训练两个数据集：
#   - GPU 0,1: MultiArith
#   - GPU 2,3: SVAMP
#
# 用法:
#   ./train_and_test_both.sh           # 并行训练两个数据集
#   ./train_and_test_both.sh --seq     # 顺序训练两个数据集 (适用于只有2张GPU的情况)
# =============================================================================

set -e

# Load environment config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found."; exit 1; }
source /data/yhao/baseline/.venv/bin/activate

# 解析参数
PARALLEL=true
if [[ "$1" == "--seq" ]] || [[ "$1" == "--sequential" ]]; then
    PARALLEL=false
fi

echo "============================================================================"
echo "MultiArith + SVAMP 训练 + 测试"
echo "============================================================================"
echo "模式: $([ "$PARALLEL" = true ] && echo '并行 (需要4张GPU)' || echo '顺序 (需要2张GPU)')"
echo "============================================================================"

OVERALL_START=$(date +%s)

# 创建日志目录
mkdir -p "${CODI_SAVE_DIR}/logs"

if $PARALLEL; then
    # =============================================================================
    # 并行模式：同时训练两个数据集
    # =============================================================================
    echo ""
    echo "🚀 启动并行训练..."
    echo "  - GPU 0,1: MultiArith"
    echo "  - GPU 2,3: SVAMP"
    echo ""
    
    # 启动 MultiArith 训练 (后台)
    CUDA_VISIBLE_DEVICES=0,1 bash "${SCRIPT_DIR}/multiarith/train_and_test_multiarith.sh" \
        > "${CODI_SAVE_DIR}/logs/multiarith_train.log" 2>&1 &
    PID_MULTIARITH=$!
    echo "  MultiArith 启动 (PID: ${PID_MULTIARITH})"
    
    # 启动 SVAMP 训练 (后台)
    CUDA_VISIBLE_DEVICES=2,3 bash "${SCRIPT_DIR}/svamp/train_and_test_svamp.sh" \
        > "${CODI_SAVE_DIR}/logs/svamp_train.log" 2>&1 &
    PID_SVAMP=$!
    echo "  SVAMP 启动 (PID: ${PID_SVAMP})"
    
    echo ""
    echo "等待训练完成..."
    echo "  日志文件:"
    echo "    - ${CODI_SAVE_DIR}/logs/multiarith_train.log"
    echo "    - ${CODI_SAVE_DIR}/logs/svamp_train.log"
    echo ""
    
    # 等待两个任务完成
    wait $PID_MULTIARITH
    MULTIARITH_EXIT=$?
    wait $PID_SVAMP
    SVAMP_EXIT=$?
    
    echo ""
    if [[ $MULTIARITH_EXIT -eq 0 ]]; then
        echo "✅ MultiArith 训练成功"
    else
        echo "❌ MultiArith 训练失败 (退出码: $MULTIARITH_EXIT)"
    fi
    
    if [[ $SVAMP_EXIT -eq 0 ]]; then
        echo "✅ SVAMP 训练成功"
    else
        echo "❌ SVAMP 训练失败 (退出码: $SVAMP_EXIT)"
    fi
    
else
    # =============================================================================
    # 顺序模式：依次训练两个数据集
    # =============================================================================
    echo ""
    echo "📝 顺序训练模式"
    echo ""
    
    echo "============================================================================"
    echo "[1/2] 训练 MultiArith..."
    echo "============================================================================"
    bash "${SCRIPT_DIR}/multiarith/train_and_test_multiarith.sh"
    
    echo ""
    echo "============================================================================"
    echo "[2/2] 训练 SVAMP..."
    echo "============================================================================"
    bash "${SCRIPT_DIR}/svamp/train_and_test_svamp.sh"
fi

OVERALL_END=$(date +%s)
OVERALL_TIME=$((OVERALL_END - OVERALL_START))

# =============================================================================
# 汇总结果
# =============================================================================
echo ""
echo "============================================================================"
echo "🎉 全部完成!"
echo "============================================================================"
echo "总耗时: ${OVERALL_TIME}s ($(echo "scale=1; ${OVERALL_TIME}/60" | bc)分钟)"
echo ""
echo "📁 结果目录:"
echo "  - MultiArith: ${CODI_RESULT_DIR}/multiarith/summary/"
echo "  - SVAMP: ${CODI_RESULT_DIR}/svamp/summary/"
echo ""

# 合并结果
echo "📊 合并测试结果..."
COMBINED_DIR="${CODI_RESULT_DIR}/combined_results"
mkdir -p "${COMBINED_DIR}"

# 创建汇总文件
echo "model,dataset,accuracy,correct,total_samples" > "${COMBINED_DIR}/all_results.csv"

for dataset in multiarith svamp; do
    result_file="${CODI_RESULT_DIR}/${dataset}/summary/all_results.csv"
    if [[ -f "$result_file" ]]; then
        tail -n +2 "$result_file" >> "${COMBINED_DIR}/all_results.csv"
    fi
done

echo ""
echo "📊 合并结果:"
if [[ -f "${COMBINED_DIR}/all_results.csv" ]]; then
    cat "${COMBINED_DIR}/all_results.csv"
fi

echo ""
echo "============================================================================"
