#!/bin/bash
# =============================================================================
# 批量运行多个模型的测试脚本
# =============================================================================
# 自动遍历多个模型目录，每个模型调用 batch_test_multi.sh 进行测试
#
# 用法:
#   ./run_all_models.sh                    # 顺序执行所有模型
#   ./run_all_models.sh --parallel         # 并行执行所有模型（后台运行）
#   ./run_all_models.sh --dry-run          # 预览命令，不实际执行
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found"; exit 1; }
source /data/yhao/baseline/.venv/bin/activate
# 参数
PARALLEL=false
DRY_RUN=false
MAX_JOBS=2  # 并发数，1 表示顺序执行
DATASETS="coin_flip commonsense"  # 默认数据集，可通过 -d 修改

# =============================================================================
# 模型配置列表 (格式: "模型名:训练目录路径")
# =============================================================================
MODEL_CONFIGS=(
    "commen_simcon_sircl_factor005:/data/yhao/baseline/CODI/outputs/commen_simcon_sircl_factor005/Llama-3.2-1B-Instruct/ep_30/lr_0.0008/seed_11"
)

# 结果根目录
RESULT_BASE="${CODI_RESULT_DIR}/manual_newfactors"

# =============================================================================
# 帮助信息
# =============================================================================
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -j, --jobs N          并发数 (默认: 1，顺序执行)"
    echo "  -p, --parallel        全部并行执行（等同于 -j ${#MODEL_CONFIGS[@]}）"
    echo "  -d, --datasets NAMES  数据集列表 (默认: \"$DATASETS\")"
    echo "  -o, --output DIR      结果根目录 (默认: $RESULT_BASE)"
    echo "  --dry-run             预览命令，不实际执行"
    echo "  -h, --help            显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 -j 2               # 同时运行 2 个模型"
    echo "  $0 -j 4               # 同时运行 4 个模型"
    echo "  $0 -p                 # 全部并行"
    echo ""
    echo "模型列表:"
    for config in "${MODEL_CONFIGS[@]}"; do
        model_name="${config%%:*}"
        echo "  - $model_name"
    done
    exit 0
}

# =============================================================================
# 解析参数
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        -p|--parallel)
            MAX_JOBS=999  # 足够大，表示全部并行
            shift
            ;;
        -d|--datasets)
            DATASETS="$2"
            shift 2
            ;;
        -o|--output)
            RESULT_BASE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "未知选项: $1"
            usage
            ;;
    esac
done

# =============================================================================
# 主流程
# =============================================================================
echo "============================================================================"
echo "批量模型测试"
echo "============================================================================"
echo "模型数量: ${#MODEL_CONFIGS[@]}"
echo "并发数: $MAX_JOBS"
echo "数据集: $DATASETS"
echo "结果目录: $RESULT_BASE"
echo "============================================================================"
echo ""

PIDS=()
RUNNING=0
LOG_DIR="${RESULT_BASE}/logs"
mkdir -p "$LOG_DIR"

# 等待函数：当运行中的任务数达到上限时等待
wait_for_slot() {
    while [ $RUNNING -ge $MAX_JOBS ]; do
        # 检查哪些任务已完成
        for i in "${!PIDS[@]}"; do
            if [ -n "${PIDS[$i]}" ] && ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                wait "${PIDS[$i]}" 2>/dev/null || true
                unset 'PIDS[$i]'
                ((RUNNING--))
            fi
        done
        if [ $RUNNING -ge $MAX_JOBS ]; then
            sleep 2
        fi
    done
}

for config in "${MODEL_CONFIGS[@]}"; do
    model_name="${config%%:*}"
    train_dir="${config##*:}"
    result_dir="${RESULT_BASE}/${model_name}"
    log_file="${LOG_DIR}/${model_name}.log"
    
    # 根据模型名前缀决定测试数据集
    if [[ "$model_name" == flip_* ]] || [[ "$model_name" == filp_* ]]; then
        test_datasets="coin_flip"
    else
        # commen_* 开头的模型
        test_datasets="commonsense"
    fi
    
    echo "----------------------------------------"
    echo "模型: $model_name"
    echo "  训练目录: $train_dir"
    echo "  结果目录: $result_dir"
    echo "  数据集: $test_datasets"
    echo "----------------------------------------"
    
    # 构建命令 - 使用环境变量传递参数
    cmd="TRAINED_DIR=\"$train_dir\" RESULTS_DIR=\"$result_dir\" DATASETS=\"$test_datasets\" bash ${SCRIPT_DIR}/batch_test_multi.sh"
    
    if $DRY_RUN; then
        echo "[DRY-RUN] $cmd"
        echo ""
        continue
    fi
    
    if [ $MAX_JOBS -gt 1 ]; then
        # 并发模式：等待有空闲槽位
        wait_for_slot
        echo "启动任务: $model_name (当前运行: $RUNNING/$MAX_JOBS)"
        nohup bash -c "$cmd" > "$log_file" 2>&1 &
        pid=$!
        PIDS+=($pid)
        ((RUNNING++))
        echo "  PID: $pid"
        echo "  日志: $log_file"
    else
        # 顺序模式
        echo "执行: $model_name"
        if bash -c "$cmd" 2>&1 | tee "$log_file"; then
            echo "✅ $model_name 完成"
        else
            echo "❌ $model_name 失败"
        fi
    fi
    echo ""
done

# =============================================================================
# 等待所有任务完成（如果是并发模式）
# =============================================================================
if [ $MAX_JOBS -gt 1 ] && [ ${#PIDS[@]} -gt 0 ]; then
    echo "============================================================================"
    echo "等待所有任务完成..."
    echo "  运行中: $RUNNING"
    echo "  日志目录: $LOG_DIR"
    echo ""
    echo "查看进度:"
    echo "  tail -f ${LOG_DIR}/*.log"
    echo "============================================================================"
    
    # 等待所有后台任务
    for pid in "${PIDS[@]}"; do
        if [ -n "$pid" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo ""
    echo "✅ 所有并发任务已完成"
fi

# =============================================================================
# 生成汇总图表
# =============================================================================
if ! $DRY_RUN; then
    echo ""
    echo "============================================================================"
    echo "生成汇总图表..."
    echo "============================================================================"
    python "${SCRIPT_DIR}/../analyze_results.py" \
        --results-dir "$RESULT_BASE" \
        --plot \
        --output "${RESULT_BASE}/plots"
    echo ""
    echo "✅ 全部完成！"
    echo "📊 图表: ${RESULT_BASE}/plots/"
    echo "📁 结果: ${RESULT_BASE}/summary/"
fi
