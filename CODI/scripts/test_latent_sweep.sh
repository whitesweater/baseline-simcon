#!/bin/bash
# =============================================================================
# 遍历 num_latent 和 inf_latent_iterations (1-18) 测试脚本
# =============================================================================
# 测试 final_use_model_codi_sim_sircl 目录下的所有模型
# 只在 gsm8k 数据集上测试
# num_latent 和 inf_latent_iterations 设置相同，从 1 到 18 遍历
#
# 用法:
#   ./test_latent_sweep.sh                     # 测试所有模型，遍历 1-18
#   ./test_latent_sweep.sh -m "codi simcon"    # 只测试指定模型
#   ./test_latent_sweep.sh --start 1 --end 6   # 只遍历 1-6
#   ./test_latent_sweep.sh --dry-run           # 预览命令
# =============================================================================

set -e

# Load environment config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found"; exit 1; }
source /data/yhao/baseline/.venv/bin/activate

# =============================================================================
# 默认配置
# =============================================================================
TRAINED_DIR="/data/yhao/baseline/CODI/final_use_model_codi_sim_sircl"
RESULTS_DIR="${CODI_RESULT_DIR}/latent_sweep_gsm8k"
DATASET="gsm8k"

START_LATENT=1
END_LATENT=18
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
    echo "  -m, --models LIST     要测试的模型列表，空格分隔 (默认: 全部)"
    echo "  --start N             起始 latent 数 (默认: 1)"
    echo "  --end N               结束 latent 数 (默认: 18)"
    echo "  -o, --output DIR      结果输出目录"
    echo "  --dry-run             只显示命令，不实际运行"
    echo "  -h, --help            显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 --start 1 --end 18"
    echo "  $0 -m \"codi codi_sircl\" --start 1 --end 10"
    echo ""
    echo "可用模型:"
    ls -1 "${TRAINED_DIR}" 2>/dev/null | sed 's/^/  - /' || echo "  (无)"
    exit 0
}

# =============================================================================
# 解析命令行参数
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        --start)
            START_LATENT="$2"
            shift 2
            ;;
        --end)
            END_LATENT="$2"
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
    exit 1
fi

# =============================================================================
# 日志设置
# =============================================================================
mkdir -p "${RESULTS_DIR}"
LOG_FILE="${RESULTS_DIR}/latent_sweep_${TIMESTAMP}.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

# =============================================================================
# 主流程
# =============================================================================
log "============================================================================"
log "Latent Sweep 测试 (num_latent = inf_latent_iterations: ${START_LATENT} -> ${END_LATENT})"
log "============================================================================"
log "配置:"
log "  - 模型目录: ${TRAINED_DIR}"
log "  - 模型列表: ${MODELS}"
log "  - 数据集: ${DATASET}"
log "  - Latent 范围: ${START_LATENT} - ${END_LATENT}"
log "  - 结果目录: ${RESULTS_DIR}"
log "============================================================================"

total_runs=0
passed_runs=0
failed_runs=0

for model in $MODELS; do
    ckpt_dir="${TRAINED_DIR}/${model}"
    
    if [[ ! -d "$ckpt_dir" ]]; then
        log "⚠ 跳过: 模型目录不存在 - ${ckpt_dir}"
        continue
    fi
    
    log ""
    log "═══════════════════════════════════════════════════════════════════════════════"
    log "🔧 模型: ${model}"
    log "═══════════════════════════════════════════════════════════════════════════════"
    
    for latent in $(seq ${START_LATENT} ${END_LATENT}); do
        total_runs=$((total_runs + 1))
        
        # 每个 latent 值创建单独的结果子目录
        run_result_dir="${RESULTS_DIR}/latent_${latent}"
        mkdir -p "${run_result_dir}"
        
        log ""
        log "--- 模型: ${model} | num_latent=${latent} | inf_latent_iterations=${latent} ---"
        
        cmd="python ${SCRIPT_DIR}/../test_multi_dataset.py \
            --model_name_or_path \"${CODI_LLAMA1B_PATH}\" \
            --ckpt_dir \"${ckpt_dir}\" \
            --datasets \"${DATASET}\" \
            --num_runs 1 \
            --result_dir \"${run_result_dir}\" \
            --seed 11 \
            --model_max_length 512 \
            --bf16 \
            --lora_r 128 --lora_alpha 32 --lora_init \
            --batch_size 128 \
            --greedy True \
            --num_latent ${latent} \
            --use_prj True \
            --prj_dim 2048 \
            --prj_no_ln False \
            --prj_dropout 0.0 \
            --inf_latent_iterations ${latent} \
            --remove_eos True \
            --use_lora True"
        
        if $DRY_RUN; then
            log "[DRY-RUN] 将执行:"
            echo "$cmd" | sed 's/  */ /g'
            passed_runs=$((passed_runs + 1))
            continue
        fi
        
        start_time=$(date +%s)
        
        if eval "$cmd" 2>&1 | tee -a "${LOG_FILE}"; then
            end_time=$(date +%s)
            elapsed=$((end_time - start_time))
            log "✅ ${model} latent=${latent} 完成 (耗时: ${elapsed}s)"
            passed_runs=$((passed_runs + 1))
        else
            log "❌ ${model} latent=${latent} 失败"
            failed_runs=$((failed_runs + 1))
        fi
    done
done

# =============================================================================
# 汇总报告
# =============================================================================
log ""
log "============================================================================"
log "Latent Sweep 测试完成"
log "============================================================================"
log "  总运行数: ${total_runs}"
log "  成功: ${passed_runs}"
log "  失败: ${failed_runs}"
log "============================================================================"
log ""
log "结果目录: ${RESULTS_DIR}"
log "  每个 latent 值的结果在: ${RESULTS_DIR}/latent_N/"
log ""

if [[ $failed_runs -gt 0 ]]; then
    exit 1
fi
