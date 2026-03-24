#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${CODI_DIR}/config.env" || { echo "Error: config.env not found. Copy config.env.example to config.env and configure."; exit 1; }
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
# shellcheck disable=SC1091
source "${CODI_VENV_PATH}" || { echo "Error: CODI_VENV_PATH is invalid: ${CODI_VENV_PATH}"; exit 1; }

bash "${SCRIPT_DIR}/prepare_assets.sh" --models llama3b --skip-datasets

export CODI_SAVE_DIR="${CODI_MULTIMODEL_SAVE_DIR}"
export CODI_RESULT_DIR="${CODI_MULTIMODEL_RESULT_DIR}"
export CODI_CACHE_DIR="${CODI_MULTIMODEL_CACHE_DIR}"

NNODES="${CODI_TRAIN_NNODES:-1}"
NPROC_PER_NODE="${CODI_TRAIN_NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-22503}"
MODEL_PATH="${CODI_MM_LLAMA3B_PATH}"
EXPT_NAME="${CODI_MULTIMODEL_TAG}_gsm8k_llama3b_decoder"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model path is not ready: ${MODEL_PATH}"
  echo "Run: bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama3b --skip-datasets"
  exit 1
fi

if [[ ! -f "${CODI_MULTIMODEL_ICOT_CACHE_DIR}/dataset_icot_0a5b3650760a22ea.pt" ]]; then
  echo "Required icot cache is missing under: ${CODI_MULTIMODEL_ICOT_CACHE_DIR}"
  echo "Run: bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama3b --skip-datasets"
  exit 1
fi

echo "=================================================================="
echo "Stage root : ${CODI_MULTIMODEL_ROOT}"
echo "Model path : ${MODEL_PATH}"
echo "Output dir : ${CODI_SAVE_DIR}"
echo "Result dir : ${CODI_RESULT_DIR}"
echo "Cache dir  : ${CODI_CACHE_DIR}"
echo "Expt name  : ${EXPT_NAME}"
echo "GPUs/node  : ${NPROC_PER_NODE}"
echo "Master port: ${MASTER_PORT}"
echo "=================================================================="

torchrun --nnodes "${NNODES}" --master_port "${MASTER_PORT}" --nproc_per_node "${NPROC_PER_NODE}" \
  "${CODI_DIR}/train.py" \
  --output_dir "${CODI_SAVE_DIR}" \
  --expt_name "${EXPT_NAME}" \
  --logging_dir "${CODI_MULTIMODEL_LOG_DIR}/${EXPT_NAME}" \
  --logging_steps 10 \
  --model_name_or_path "${MODEL_PATH}" \
  --data_name icot \
  --seed 11 \
  --model_max_length 512 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --bf16 \
  --num_train_epochs 8 \
  --learning_rate 3e-4 \
  --max_grad_norm 2.0 \
  --use_lora True \
  --lora_r 128 \
  --lora_alpha 32 \
  --lora_init \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 2 \
  --save_safetensors False \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --do_train \
  --report_to tensorboard \
  --num_latent 6 \
  --logging_strategy steps \
  --use_prj True \
  --prj_dim 3072 \
  --prj_dropout 0.0 \
  --distill_loss_div_std True \
  --exp_mode False \
  --exp_data_num 200 \
  --remove_eos True \
  --distill_loss_factor 20 \
  --print_ref_model_stats True \
  --max_token_num 200 \
  --use_decoder True \
  --ddp_find_unused_parameters False
