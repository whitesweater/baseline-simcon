#!/bin/bash
# =============================================================================
# 批量测试脚本 - 支持多模型、多次运行、结果记录
# =============================================================================
# 用法:
#   ./batch_test.sh                    # 测试所有模型，每个运行1次
#   ./batch_test.sh -r 3               # 每个模型运行3次
#   ./batch_test.sh -m euclidean       # 只测试 euclidean 模型
#   ./batch_test.sh -m "euclidean hyperbolic"  # 测试指定的多个模型
#   ./batch_test.sh -d svamp           # 使用 svamp 数据集
#   ./batch_test.sh --dry-run          # 只显示将要执行的命令，不实际运行
# =============================================================================

set -e

# Load environment config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found"; exit 1; }

# =============================================================================
# 默认配置
# =============================================================================
TRAINED_DIR="${CODI_SAVE_DIR}/trained"
RESULTS_DIR="${CODI_RESULT_DIR}"
DATA_NAMES="gsm8k svamp gsm-hard multi-arith commonsense"  # 默认测试 gsm8k，支持多个数据集用空格分隔
NUM_RUNS=1
MODELS=""  # 空表示测试所有
DRY_RUN=false
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# 解析命令行参数
# =============================================================================
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -r, --runs N          每个模型运行的次数 (默认: 1)"
    echo "  -m, --models LIST     要测试的模型列表，空格分隔 (默认: 全部)"
    echo "  -d, --datasets NAMES  数据集名称列表，空格分隔 (默认: gsm8k)"
    echo "                        可用: gsm8k, svamp, gsm-hard, multi-arith, commonsense"
    echo "  -o, --output DIR      结果输出目录 (默认: ${CODI_RESULT_DIR})"
    echo "  --dry-run             只显示命令，不实际运行"
    echo "  -h, --help            显示帮助"
    echo ""
    echo "注意:"
    echo "  - 数据集会在首次使用时自动下载（通过 HuggingFace）"
    echo "  - 循环顺序: 模型 -> 数据集 -> 运行次数（避免重复加载模型）"
    echo ""
    echo "示例:"
    echo "  $0 -d \"gsm8k svamp\" -r 3                      # 在两个数据集上各运行3次"
    echo "  $0 -m euclidean -d \"gsm8k gsm-hard\"           # 指定模型和数据集"
    echo ""
    echo "可用模型 (${TRAINED_DIR}):"
    ls -1 "${TRAINED_DIR}" 2>/dev/null | sed 's/^/  - /'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -d|--datasets|--data)
            DATA_NAMES="$2"
            shift 2
            ;;
        -o|--output)
            RESULTS_DIR="$2"
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
# 获取模型列表
# =============================================================================
if [[ -z "$MODELS" ]]; then
    MODELS=$(ls -1 "${TRAINED_DIR}" 2>/dev/null | tr '\n' ' ')
fi

# =============================================================================
# 创建结果目录和日志文件
# =============================================================================
mkdir -p "${RESULTS_DIR}"
LOG_FILE="${RESULTS_DIR}/batch_test_${TIMESTAMP}.log"
SUMMARY_FILE="${RESULTS_DIR}/batch_summary_${TIMESTAMP}.csv"

# 为每个数据集创建子目录
for data_name in $DATA_NAMES; do
    mkdir -p "${RESULTS_DIR}/${data_name}"
done

# 初始化 summary CSV
echo "timestamp,model,dataset,run_id,accuracy,total_samples,correct,elapsed_sec" > "${SUMMARY_FILE}"

# =============================================================================
# 日志函数
# =============================================================================
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

# =============================================================================
# 运行单次测试
# =============================================================================
run_single_test() {
    local model_name=$1
    local data_name=$2
    local run_id=$3
    local ckpt_dir="${TRAINED_DIR}/${model_name}"
    
    # 检查模型目录是否存在
    if [[ ! -d "$ckpt_dir" ]]; then
        log "⚠ 跳过: 模型目录不存在 - ${ckpt_dir}"
        return 1
    fi
    
    local output_dir="${CODI_SAVE_DIR}/testoutput/${data_name}/${model_name}_run${run_id}_${TIMESTAMP}"
    local result_subdir="${RESULTS_DIR}/${data_name}"
    mkdir -p "${result_subdir}"
    
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "📊 测试: model=${model_name}, dataset=${data_name}, run=${run_id}/${NUM_RUNS}"
    log "   ckpt: ${ckpt_dir}"
    log "   结果: ${result_subdir}/"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    local cmd="python ${SCRIPT_DIR}/../test.py \
        --data_name \"${data_name}\" \
        --output_dir \"${output_dir}\" \
        --model_name_or_path \"${CODI_LLAMA1B_PATH}\" \
        --seed $((11 + run_id)) \
        --model_max_length 512 \
        --bf16 \
        --lora_r 128 --lora_alpha 32 --lora_init \
        --batch_size 128 \
        --greedy True \
        --num_latent 6 \
        --use_prj True \
        --prj_dim 2048 \
        --prj_no_ln False \
        --prj_dropout 0.0 \
        --inf_latent_iterations 6 \
        --inf_num_iterations 1 \
        --remove_eos True \
        --trajectory_radius_threshold 2 \
        --trajectory_max_acceleration 1.0 \
        --trajectory_action_lambda_energy 1.0 \
        --trajectory_action_lambda_length 0.1 \
        --trajectory_curvature -1.0 \
        --use_lora True \
        --ckpt_dir \"${ckpt_dir}\""
    
    if $DRY_RUN; then
        log "[DRY-RUN] 将执行:"
        echo "$cmd" | sed 's/^/    /'
        return 0
    fi
    
    local start_time=$(date +%s)
    
    # 运行测试并捕获输出
    local test_output
    if test_output=$(eval "$cmd" 2>&1); then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        
        # 尝试从输出中提取准确率
        local accuracy=$(echo "$test_output" | grep -oP 'accuracy[:\s]+\K[\d.]+' | tail -1 || echo "N/A")
        local correct=$(echo "$test_output" | grep -oP 'correct[:\s]+\K\d+' | tail -1 || echo "N/A")
        local total=$(echo "$test_output" | grep -oP 'total[:\s]+\K\d+' | tail -1 || echo "N/A")
        
        log "✅ 完成: accuracy=${accuracy}, time=${elapsed}s"
        
        # 记录到 summary
        echo "${TIMESTAMP},${model_name},${data_name},${run_id},${accuracy},${total},${correct},${elapsed}" >> "${SUMMARY_FILE}"
        
        # 保存详细输出（按数据集分类）
        echo "$test_output" >> "${result_subdir}/${model_name}_run${run_id}_output.txt"
        
    else
        log "❌ 失败: ${model_name} (${data_name}) run ${run_id}"
        echo "$test_output" >> "${result_subdir}/${model_name}_run${run_id}_error.txt"
        echo "${TIMESTAMP},${model_name},${data_name},${run_id},FAILED,,," >> "${SUMMARY_FILE}"
        return 1
    fi
}

# =============================================================================
# 主流程
# =============================================================================
log "============================================================================"
log "批量测试开始"
log "============================================================================"
log "配置:"
log "  - 模型目录: ${TRAINED_DIR}"
log "  - 模型列表: ${MODELS}"
log "  - 数据集: ${DATA_NAMES}"
log "  - 运行次数: ${NUM_RUNS}"
log "  - 结果目录: ${RESULTS_DIR}"
log "  - 日志文件: ${LOG_FILE}"
log "  - 汇总文件: ${SUMMARY_FILE}"
log "  - 循环顺序: 模型 -> 数据集 -> 运行次数 (避免重复加载)"
log "  - 数据集下载: 自动（首次使用时通过 HuggingFace）"
log "============================================================================"

total_tests=0
passed_tests=0
failed_tests=0

for model in $MODELS; do
    log ""
    log "═══════════════════════════════════════════════════════════════════════════════"
    log "🔧 模型: ${model}"
    log "═══════════════════════════════════════════════════════════════════════════════"
    
    for data_name in $DATA_NAMES; do
        log ""
        log "📊 数据集: ${data_name}"
        
        for ((run=0; run<NUM_RUNS; run++)); do
            total_tests=$((total_tests + 1))
            if run_single_test "$model" "$data_name" "$run"; then
                passed_tests=$((passed_tests + 1))
            else
                failed_tests=$((failed_tests + 1))
            fi
        done
    done
done

# =============================================================================
# 汇总报告
# =============================================================================
log ""
log "============================================================================"
log "测试完成汇总"
log "============================================================================"
log "  总测试数: ${total_tests}"
log "  成功: ${passed_tests}"
log "  失败: ${failed_tests}"
log "============================================================================"

if [[ -f "${SUMMARY_FILE}" ]]; then
    log ""
    log "📊 结果汇总表 (${SUMMARY_FILE}):"
    column -t -s',' "${SUMMARY_FILE}" | head -20
fi

log ""
log "📁 结果文件保存在: ${RESULTS_DIR}/"
log "   - batch_test_${TIMESTAMP}.log (运行日志)"
log "   - batch_summary_${TIMESTAMP}.csv (汇总表)"
log ""

# 如果有失败的测试，返回非零退出码
if [[ $failed_tests -gt 0 ]]; then
    exit 1
fi
