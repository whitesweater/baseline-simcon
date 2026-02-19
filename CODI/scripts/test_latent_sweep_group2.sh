#!/bin/bash
# =============================================================================
# Latent Sweep 测试脚本 - 第2组: simcon, sircl
# =============================================================================
# 两个模型分别在 GPU 0 和 GPU 1 上并行运行
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found"; exit 1; }
source /data/yhao/baseline/.venv/bin/activate

# =============================================================================
# 配置
# =============================================================================
TRAINED_DIR="/data/yhao/baseline/CODI/final_use_model_codi_sim_sircl"
RESULTS_DIR="${CODI_RESULT_DIR}/latent_sweep_gsm8k"
DATASET="gsm8k"

# 第2组模型 - 分配到不同 GPU
MODEL_GPU0="simcon"
MODEL_GPU1="simcon_sircl"

START_LATENT=1
END_LATENT=18
DRY_RUN=false
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --start) START_LATENT="$2"; shift 2 ;;
        --end) END_LATENT="$2"; shift 2 ;;
        -o|--output) RESULTS_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) shift ;;
    esac
done

mkdir -p "${RESULTS_DIR}"
LOG_FILE="${RESULTS_DIR}/latent_sweep_group2_${TIMESTAMP}.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [Group2] $1"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

log "============================================================================"
log "Latent Sweep 测试 - 第2组 (并行模式)"
log "  GPU 0: ${MODEL_GPU0}"
log "  GPU 1: ${MODEL_GPU1}"
log "Latent 范围: ${START_LATENT} - ${END_LATENT}"
log "============================================================================"

# =============================================================================
# 单个模型测试函数
# =============================================================================
run_model_tests() {
    local model=$1
    local gpu_id=$2
    local log_suffix=$3
    
    local model_log="${RESULTS_DIR}/latent_sweep_${model}_${TIMESTAMP}.log"
    local ckpt_dir="${TRAINED_DIR}/${model}"
    
    if [[ ! -d "$ckpt_dir" ]]; then
        echo "[GPU${gpu_id}] ⚠ 跳过: 模型目录不存在 - ${ckpt_dir}" | tee -a "${model_log}"
        return 1
    fi
    
    echo "[GPU${gpu_id}] 开始测试模型: ${model}" | tee -a "${model_log}"
    
    local passed=0
    local failed=0
    
    for latent in $(seq ${START_LATENT} ${END_LATENT}); do
        run_result_dir="${RESULTS_DIR}/latent_${latent}"
        mkdir -p "${run_result_dir}"
        
        echo "[GPU${gpu_id}] ${model} | latent=${latent}" | tee -a "${model_log}"
        
        if $DRY_RUN; then
            echo "[GPU${gpu_id}] [DRY-RUN] CUDA_VISIBLE_DEVICES=${gpu_id} python test_multi_dataset.py ... --num_latent ${latent}" | tee -a "${model_log}"
            passed=$((passed + 1))
            continue
        fi
        
        start_time=$(date +%s)
        
        if CUDA_VISIBLE_DEVICES=${gpu_id} python ${SCRIPT_DIR}/../test_multi_dataset.py \
            --model_name_or_path "${CODI_LLAMA1B_PATH}" \
            --ckpt_dir "${ckpt_dir}" \
            --datasets "${DATASET}" \
            --num_runs 1 \
            --result_dir "${run_result_dir}" \
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
            --use_lora True 2>&1 | tee -a "${model_log}"; then
            end_time=$(date +%s)
            elapsed=$((end_time - start_time))
            echo "[GPU${gpu_id}] ✅ ${model} latent=${latent} 完成 (${elapsed}s)" | tee -a "${model_log}"
            passed=$((passed + 1))
        else
            echo "[GPU${gpu_id}] ❌ ${model} latent=${latent} 失败" | tee -a "${model_log}"
            failed=$((failed + 1))
        fi
    done
    
    echo "[GPU${gpu_id}] ${model} 完成: 成功=${passed} 失败=${failed}" | tee -a "${model_log}"
    return $failed
}

# =============================================================================
# 并行运行两个模型
# =============================================================================
log "🚀 启动并行测试..."

# GPU 0 运行第一个模型
run_model_tests "${MODEL_GPU0}" 0 "gpu0" &
PID_GPU0=$!

# GPU 1 运行第二个模型  
run_model_tests "${MODEL_GPU1}" 1 "gpu1" &
PID_GPU1=$!

log "  ${MODEL_GPU0} PID: ${PID_GPU0} (GPU 0)"
log "  ${MODEL_GPU1} PID: ${PID_GPU1} (GPU 1)"
log "⏳ 等待两个模型测试完成..."

# 等待两个进程完成
wait $PID_GPU0
STATUS_GPU0=$?

wait $PID_GPU1
STATUS_GPU1=$?

log ""
log "============================================================================"
log "第2组测试完成"
log "  ${MODEL_GPU0} (GPU 0): $([ $STATUS_GPU0 -eq 0 ] && echo '✅ 成功' || echo '❌ 有失败')"
log "  ${MODEL_GPU1} (GPU 1): $([ $STATUS_GPU1 -eq 0 ] && echo '✅ 成功' || echo '❌ 有失败')"
log "============================================================================"

[[ $STATUS_GPU0 -ne 0 || $STATUS_GPU1 -ne 0 ]] && exit 1 || exit 0
