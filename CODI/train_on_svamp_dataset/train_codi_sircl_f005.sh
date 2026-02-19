#!/bin/bash
# =============================================================================
# SVAMP - codi+sircl 模型 (factor=0.05)
# use_decoder=False, use_trajectory=True, trajectory_loss_factor=0.05
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found."; exit 1; }
source /data/yhao/baseline/.venv/bin/activate

DATASET_NAME="svamp"
MODEL_TYPE="codi_sircl_f005"
EXPT_NAME="${DATASET_NAME}_${MODEL_TYPE}"
NUM_GPUS=2
MASTER_PORT=22536
NUM_EPOCHS=12
BATCH_SIZE=32

SAVE_DIR="${CODI_SAVE_DIR}"
RESULT_DIR="${CODI_RESULT_DIR}/${DATASET_NAME}/${MODEL_TYPE}"
mkdir -p "${SAVE_DIR}" "${RESULT_DIR}"

echo "============================================================================"
echo "SVAMP - codi+sircl (decoder=False, trajectory=True, factor=0.05)"
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
    --use_decoder False \
    --use_trajectory_consistency True \
    --trajectory_space_type euclidean \
    --trajectory_radius_threshold 2 \
    --trajectory_loss_factor 0.05

TRAIN_END=$(date +%s)
TRAIN_TIME=$((TRAIN_END - TRAIN_START))
echo "✅ 训练完成! 耗时: ${TRAIN_TIME}s"
