#!/bin/bash
# =============================================================================
# 多模型多数据集批量测试脚本
# =============================================================================
# 核心优化：
# - 每个模型只加载一次，然后在所有数据集上测试
# - 清晰的结果目录结构
# - 自动生成汇总报告
#
# 用法:
#   ./batch_test_multi.sh                              # 测试所有模型和数据集
#   ./batch_test_multi.sh -m euclidean                 # 只测试 euclidean 模型
#   ./batch_test_multi.sh -d "gsm8k svamp"             # 只测试指定数据集
#   ./batch_test_multi.sh -r 3                         # 每个数据集运行3次
#   ./batch_test_multi.sh --dry-run                    # 预览命令
# =============================================================================

set -e

# Load environment config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found"; exit 1; }

# =============================================================================
# 默认配置
# =============================================================================
TRAINED_DIR="${CODI_SAVE_DIR}/codi-base/Llama-3.2-1B-Instruct/ep_10/lr_0.0008/seed_11"
RESULTS_DIR="${CODI_RESULT_DIR}/codi-base"
DATASETS="gsm8k svamp gsm-hard multi-arith"  # 默认数据集
NUM_RUNS=1
MODELS=""  # 空表示测试所有
DRY_RUN=false
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# 帮助信息
# =============================================================================
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -r, --runs N          每个数据集运行的次数 (默认: 1)"
    echo "  -m, --models LIST     要测试的模型列表，空格分隔 (默认: 全部)"
    echo "  -d, --datasets NAMES  数据集列表，空格分隔 (默认: gsm8k svamp gsm-hard multi-arith)"
    echo "                        可用: gsm8k, svamp, gsm-hard, multi-arith, commonsense"
    echo "  -o, --output DIR      结果输出目录 (默认: ${CODI_RESULT_DIR})"
    echo "  --dry-run             只显示命令，不实际运行"
    echo "  -h, --help            显示帮助"
    echo ""
    echo "核心优化:"
    echo "  - 每个模型只加载一次，然后在所有数据集上测试"
    echo "  - 清晰的结果目录结构：results/models/{model}/{dataset}/"
    echo "  - 自动生成汇总报告"
    echo ""
    echo "示例:"
    echo "  $0 -m euclidean -d \"gsm8k svamp\" -r 3"
    echo ""
    echo "可用模型 (${TRAINED_DIR}):"
    ls -1 "${TRAINED_DIR}" 2>/dev/null | sed 's/^/  - /' || echo "  (无)"
    exit 0
}

# =============================================================================
# 解析命令行参数
# =============================================================================
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
            DATASETS="$2"
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

if [[ -z "$MODELS" ]]; then
    echo "错误: 没有找到可用的模型"
    echo "请确认目录 ${TRAINED_DIR} 中有训练好的模型"
    exit 1
fi

# =============================================================================
# 日志设置
# =============================================================================
mkdir -p "${RESULTS_DIR}"
LOG_FILE="${RESULTS_DIR}/batch_test_${TIMESTAMP}.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

# =============================================================================
# 主流程
# =============================================================================
log "============================================================================"
log "多模型多数据集批量测试"
log "============================================================================"
log "配置:"
log "  - 模型目录: ${TRAINED_DIR}"
log "  - 模型列表: ${MODELS}"
log "  - 数据集: ${DATASETS}"
log "  - 每个数据集运行次数: ${NUM_RUNS}"
log "  - 结果目录: ${RESULTS_DIR}"
log "  - 日志文件: ${LOG_FILE}"
log "============================================================================"
log ""
log "执行策略: 每个模型加载一次，然后在所有数据集上测试"
log "============================================================================"

total_models=0
passed_models=0
failed_models=0

for model in $MODELS; do
    total_models=$((total_models + 1))
    ckpt_dir="${TRAINED_DIR}/${model}"
    
    if [[ ! -d "$ckpt_dir" ]]; then
        log "⚠ 跳过: 模型目录不存在 - ${ckpt_dir}"
        failed_models=$((failed_models + 1))
        continue
    fi
    
    log ""
    log "═══════════════════════════════════════════════════════════════════════════════"
    log "🔧 模型: ${model}"
    log "   Checkpoint: ${ckpt_dir}"
    log "   数据集: ${DATASETS}"
    log "   运行次数: ${NUM_RUNS}"
    log "═══════════════════════════════════════════════════════════════════════════════"
    
    cmd="python ${SCRIPT_DIR}/../test_multi_dataset.py \
        --model_name_or_path \"${CODI_LLAMA1B_PATH}\" \
        --ckpt_dir \"${ckpt_dir}\" \
        --datasets \"${DATASETS}\" \
        --num_runs ${NUM_RUNS} \
        --result_dir \"${RESULTS_DIR}\" \
        --seed 11 \
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
        --remove_eos True \
        --use_lora True"
    
    if $DRY_RUN; then
        log "[DRY-RUN] 将执行:"
        echo "$cmd" | sed 's/^/    /'
        passed_models=$((passed_models + 1))
        continue
    fi
    
    start_time=$(date +%s)
    
    if eval "$cmd" 2>&1 | tee -a "${LOG_FILE}"; then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        log "✅ 模型 ${model} 测试完成 (耗时: ${elapsed}s)"
        passed_models=$((passed_models + 1))
    else
        log "❌ 模型 ${model} 测试失败"
        failed_models=$((failed_models + 1))
    fi
done

# =============================================================================
# 汇总报告
# =============================================================================
log ""
log "============================================================================"
log "批量测试完成"
log "============================================================================"
log "  总模型数: ${total_models}"
log "  成功: ${passed_models}"
log "  失败: ${failed_models}"
log "============================================================================"
log ""
log "📁 结果目录结构:"
log "   ${RESULTS_DIR}/"
log "   ├── models/          # 按模型组织的结果"
log "   ├── datasets/        # 按数据集组织的模型对比"
log "   └── summary/         # 汇总报告"
log ""

# 显示汇总
if [[ -f "${RESULTS_DIR}/summary/comparison_matrix.csv" ]]; then
    log "📊 模型×数据集对比矩阵:"
    column -t -s',' "${RESULTS_DIR}/summary/comparison_matrix.csv" 2>/dev/null | head -20
fi

log ""
log "查看详细结果:"
log "  cat ${RESULTS_DIR}/summary/all_results.csv"
log "  cat ${RESULTS_DIR}/summary/comparison_matrix.csv"
log ""

if [[ $failed_models -gt 0 ]]; then
    exit 1
fi
