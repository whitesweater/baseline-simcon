#!/bin/bash
set -euo pipefail
# Qwen3-0.6B training with decoder + Euclidean trajectory consistency

# Load environment config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found. Copy config.env.example to config.env and configure."; exit 1; }
source "${CODI_VENV_PATH}"
cd "${SCRIPT_DIR}/.."

EXPT_NAME="qwen3_0p6b_4radius"
SAVE_DIR="${CODI_SAVE_DIR}/qwen3_0p6b"
LOG_DIR="${CODI_RUN_ROOT}/logs/${EXPT_NAME}"
RESULT_DIR="${CODI_RESULT_DIR}/${EXPT_NAME}"
MODEL_PATH="${CODI_QWEN3_0P6B_PATH}"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}" "${RESULT_DIR}"

torchrun --nnodes 1 --master_port 22520 --nproc_per_node 4 train.py \
	--output_dir "${SAVE_DIR}" \
	--expt_name "${EXPT_NAME}" \
	--logging_dir "${LOG_DIR}" \
	--logging_steps 10 \
	--model_name_or_path "${MODEL_PATH}" \
	--data_name icot \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 32 \
	--gradient_accumulation_steps 1 \
	--bf16 \
	--dataloader_num_workers 4 \
	--dataloader_pin_memory True \
	--dataloader_persistent_workers True \
	--dataloader_prefetch_factor 2 \
	--num_train_epochs 10 \
	--learning_rate 8e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 \
	--lora_alpha 32 \
	--lora_init \
	--save_strategy epoch \
	--save_total_limit 20 \
	--save_safetensors False \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type cosine \
	--do_train \
	--report_to tensorboard \
	--num_latent 16 \
	--logging_strategy steps \
	--use_prj True \
	--prj_dim 1024 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--exp_data_num 200 \
	--remove_eos True \
	--distill_loss_factor 20 \
	--print_ref_model_stats False \
	--max_token_num 200 \
	--use_decoder True \
	--use_trajectory_consistency False \
	--trajectory_space_type euclidean \
	--trajectory_radius_threshold 4 \
	--trajectory_loss_factor 0.1
