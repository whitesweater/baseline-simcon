#!/bin/bash
# =============================================================================
# MultiArith - 训练全部6种模型 + 统一测试
# 每个模型使用2张GPU，顺序执行
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found."; exit 1; }
source /data/yhao/baseline/.venv/bin/activate

DATASET_NAME="multiarith"
RESULT_DIR="${CODI_RESULT_DIR}/${DATASET_NAME}"
mkdir -p "${RESULT_DIR}"

echo "============================================================================"
echo "MultiArith - 训练全部6种模型 + 统一测试"
echo "============================================================================"
echo ""
echo "模型配置:"
echo "  1. sim-con+sircl:       decoder=True,  trajectory=True,  factor=0.1"
echo "  2. sim-con+sircl_f005:  decoder=True,  trajectory=True,  factor=0.05"
echo "  3. codi+sircl:          decoder=False, trajectory=True,  factor=0.1"
echo "  4. codi+sircl_f005:     decoder=False, trajectory=True,  factor=0.05"
echo "  5. sim-con:             decoder=True,  trajectory=False"
echo "  6. codi:                decoder=False, trajectory=False"
echo "============================================================================"
echo ""

TOTAL_START=$(date +%s)

# 存储所有 checkpoint 路径
declare -A CHECKPOINTS

# =============================================================================
# 训练阶段
# =============================================================================

train_model() {
    local model_name=$1
    local script_name=$2
    local idx=$3
    local total=$4
    
    echo ""
    echo "[${idx}/${total}] 训练 ${model_name}..."
    echo "------------------------------------------------------------"
    bash "${SCRIPT_DIR}/${script_name}"
    
    # 获取 checkpoint 路径
    local expt_name="${DATASET_NAME}_${model_name}"
    local ckpt_base="${CODI_SAVE_DIR}/${expt_name}/$(basename ${CODI_LLAMA1B_PATH})/ep_12/lr_0.0008/seed_11"
    local ckpt=$(ls -dt ${ckpt_base}/checkpoint-* 2>/dev/null | head -1)
    [[ -z "${ckpt}" ]] && ckpt="${ckpt_base}"
    CHECKPOINTS["${model_name}"]="${ckpt}"
    echo "✅ ${model_name} checkpoint: ${ckpt}"
}

# 训练6个模型
train_model "simcon_sircl" "train_simcon_sircl.sh" 1 6
train_model "simcon_sircl_f005" "train_simcon_sircl_f005.sh" 2 6
train_model "codi_sircl" "train_codi_sircl.sh" 3 6
train_model "codi_sircl_f005" "train_codi_sircl_f005.sh" 4 6
train_model "simcon" "train_simcon.sh" 5 6
train_model "codi" "train_codi.sh" 6 6

TRAIN_END=$(date +%s)
TRAIN_TIME=$((TRAIN_END - TOTAL_START))
echo ""
echo "============================================================================"
echo "✅ 全部6个模型训练完成! 训练总耗时: ${TRAIN_TIME}s"
echo "============================================================================"

# =============================================================================
# 测试阶段
# =============================================================================
echo ""
echo "============================================================================"
echo "开始测试阶段"
echo "============================================================================"

TEST_START=$(date +%s)

test_model() {
    local model_name=$1
    local ckpt_dir=$2
    local idx=$3
    local total=$4
    local port=$5
    
    local result_subdir="${RESULT_DIR}/${model_name}"
    mkdir -p "${result_subdir}"
    
    echo ""
    echo "[${idx}/${total}] 测试 ${model_name}..."
    echo "  Checkpoint: ${ckpt_dir}"
    echo "  结果目录: ${result_subdir}"
    echo "------------------------------------------------------------"
    
    torchrun --nnodes 1 --master_port ${port} --nproc_per_node 2 test_multi_dataset.py \
        --model_name_or_path "${CODI_LLAMA1B_PATH}" \
        --ckpt_dir "${ckpt_dir}" \
        --datasets "multi-arith svamp gsm8k" \
        --num_runs 1 \
        --result_dir "${result_subdir}" \
        --seed 11 \
        --model_max_length 512 \
        --bf16 \
        --lora_r 128 --lora_alpha 32 --lora_init \
        --batch_size 32 \
        --greedy True \
        --num_latent 6 \
        --use_prj True \
        --prj_dim 2048 \
        --prj_no_ln False \
        --prj_dropout 0.0 \
        --inf_latent_iterations 6 \
        --remove_eos True \
        --use_lora True
    
    echo "✅ ${model_name} 测试完成"
}

# 测试6个模型
test_model "simcon_sircl" "${CHECKPOINTS[simcon_sircl]}" 1 6 23521
test_model "simcon_sircl_f005" "${CHECKPOINTS[simcon_sircl_f005]}" 2 6 23522
test_model "codi_sircl" "${CHECKPOINTS[codi_sircl]}" 3 6 23523
test_model "codi_sircl_f005" "${CHECKPOINTS[codi_sircl_f005]}" 4 6 23524
test_model "simcon" "${CHECKPOINTS[simcon]}" 5 6 23525
test_model "codi" "${CHECKPOINTS[codi]}" 6 6 23526

TEST_END=$(date +%s)
TEST_TIME=$((TEST_END - TEST_START))

# =============================================================================
# 汇总结果
# =============================================================================
echo ""
echo "============================================================================"
echo "汇总所有模型结果..."
echo "============================================================================"

# 合并所有结果到一个文件
SUMMARY_FILE="${RESULT_DIR}/all_models_summary.csv"
echo "model,dataset,accuracy,run" > "${SUMMARY_FILE}"

for model_name in simcon_sircl simcon_sircl_f005 codi_sircl codi_sircl_f005 simcon codi; do
    result_file="${RESULT_DIR}/${model_name}/summary/all_results.csv"
    if [[ -f "${result_file}" ]]; then
        # 跳过header，添加模型名前缀
        tail -n +2 "${result_file}" | while read line; do
            echo "${model_name},${line}" >> "${SUMMARY_FILE}"
        done
    fi
done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================================================"
echo "✅ MultiArith 全部完成!"
echo "============================================================================"
echo "  训练耗时: ${TRAIN_TIME}s ($(echo "scale=2; ${TRAIN_TIME}/3600" | bc)小时)"
echo "  测试耗时: ${TEST_TIME}s ($(echo "scale=2; ${TEST_TIME}/3600" | bc)小时)"
echo "  总耗时: ${TOTAL_TIME}s ($(echo "scale=2; ${TOTAL_TIME}/3600" | bc)小时)"
echo ""
echo "📁 结果文件: ${SUMMARY_FILE}"
echo "============================================================================"

# 显示汇总结果
if [[ -f "${SUMMARY_FILE}" ]]; then
    echo ""
    echo "📊 所有模型测试结果:"
    cat "${SUMMARY_FILE}"
fi
