#!/bin/bash
# =============================================================================
# MultiArith 数据集训练 + 测试一体化脚本
# 使用 2 张 GPU 进行训练
# =============================================================================

set -e

# Load environment config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found."; exit 1; }
source /data/yhao/baseline/.venv/bin/activate

# =============================================================================
# 配置
# =============================================================================
DATASET_NAME="multiarith"
EXPT_NAME="${DATASET_NAME}_euclidean"
NUM_GPUS=2
MASTER_PORT=22521
NUM_EPOCHS=12
BATCH_SIZE=32  # 每卡 batch size

SAVE_DIR="${CODI_SAVE_DIR}"
RESULT_DIR="${CODI_RESULT_DIR}/${DATASET_NAME}"
mkdir -p "${SAVE_DIR}" "${RESULT_DIR}"

echo "============================================================================"
echo "MultiArith 训练 + 测试流程"
echo "============================================================================"
echo "配置:"
echo "  - 数据集: ${DATASET_NAME}"
echo "  - 实验名称: ${EXPT_NAME}"
echo "  - GPU 数量: ${NUM_GPUS}"
echo "  - 训练轮数: ${NUM_EPOCHS}"
echo "  - 每卡 Batch Size: ${BATCH_SIZE}"
echo "  - 保存目录: ${SAVE_DIR}"
echo "  - 结果目录: ${RESULT_DIR}"
echo "============================================================================"

# =============================================================================
# 阶段1: 训练
# =============================================================================
echo ""
echo "[阶段1] 开始训练..."
echo "============================================================================"

TRAIN_START=$(date +%s)

torchrun --nnodes 1 --master_port ${MASTER_PORT} --nproc_per_node ${NUM_GPUS} train.py \
    --output_dir "${SAVE_DIR}" \
    --expt_name "${EXPT_NAME}" \
    --logging_dir "${SAVE_DIR}/logs/${EXPT_NAME}-logs" \
    --logging_steps 10 \
    --model_name_or_path "${CODI_LLAMA1B_PATH}" \
    --data_name "${DATASET_NAME}" \
    --seed 11 \
    --model_max_length 512 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 2 \
    --num_train_epochs ${NUM_EPOCHS} \
    --learning_rate 8e-4 \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 128 \
    --lora_alpha 32 \
    --lora_init \
    --save_strategy epoch \
    --save_total_limit 5 \
    --save_safetensors False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --do_train \
    --report_to tensorboard \
    --num_latent 6 \
    --logging_strategy steps \
    --use_prj True \
    --prj_dim 2048 \
    --prj_dropout 0.0 \
    --distill_loss_div_std True \
    --exp_mode False \
    --remove_eos True \
    --distill_loss_factor 20 \
    --print_ref_model_stats False \
    --max_token_num 300 \
    --use_decoder True \
    --use_trajectory_consistency True \
    --trajectory_space_type euclidean \
    --trajectory_radius_threshold 2 \
    --trajectory_loss_factor 0.1

TRAIN_END=$(date +%s)
TRAIN_TIME=$((TRAIN_END - TRAIN_START))
echo ""
echo "✅ 训练完成! 耗时: ${TRAIN_TIME}s"

# =============================================================================
# 阶段2: 找到最新的 checkpoint
# =============================================================================
echo ""
echo "[阶段2] 查找最新 checkpoint..."
echo "============================================================================"

# 构建 checkpoint 路径模式
CKPT_BASE="${SAVE_DIR}/${EXPT_NAME}/$(basename ${CODI_LLAMA1B_PATH})/ep_${NUM_EPOCHS}/lr_0.0008/seed_11"

# 找到最新的 checkpoint
LATEST_CKPT=$(ls -dt ${CKPT_BASE}/checkpoint-* 2>/dev/null | head -1)

if [[ -z "${LATEST_CKPT}" ]]; then
    echo "❌ 未找到 checkpoint，尝试使用基础目录..."
    LATEST_CKPT="${CKPT_BASE}"
fi

echo "使用 checkpoint: ${LATEST_CKPT}"

# =============================================================================
# 阶段3: 测试
# =============================================================================
echo ""
echo "[阶段3] 开始测试..."
echo "============================================================================"

TEST_START=$(date +%s)

# 在多个数据集上测试
python test_multi_dataset.py \
    --model_name_or_path "${CODI_LLAMA1B_PATH}" \
    --ckpt_dir "${LATEST_CKPT}" \
    --datasets "multi-arith svamp gsm8k" \
    --num_runs 1 \
    --result_dir "${RESULT_DIR}" \
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

TEST_END=$(date +%s)
TEST_TIME=$((TEST_END - TEST_START))
echo ""
echo "✅ 测试完成! 耗时: ${TEST_TIME}s"

# =============================================================================
# 汇总
# =============================================================================
echo ""
echo "============================================================================"
echo "全部完成!"
echo "============================================================================"
echo "  训练耗时: ${TRAIN_TIME}s"
echo "  测试耗时: ${TEST_TIME}s"
echo "  总耗时: $((TRAIN_TIME + TEST_TIME))s"
echo ""
echo "📁 输出文件:"
echo "  - 模型: ${LATEST_CKPT}"
echo "  - 结果: ${RESULT_DIR}/summary/"
echo "============================================================================"

# 显示测试结果
if [[ -f "${RESULT_DIR}/summary/all_results.csv" ]]; then
    echo ""
    echo "📊 测试结果:"
    cat "${RESULT_DIR}/summary/all_results.csv"
fi
