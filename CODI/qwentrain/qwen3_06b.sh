#!/bin/bash
set -euo pipefail

# Load environment config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found. Copy config.env.example to config.env and configure."; exit 1; }
source "${CODI_VENV_PATH}"
cd "${SCRIPT_DIR}/.."

EXPT_NAME="qwen3_06b"
SAVE_DIR="${CODI_SAVE_DIR}"
LOG_DIR="${SAVE_DIR}/logs/${EXPT_NAME}-logs"
MODEL_PATH="${CODI_QWEN3_0P6B_PATH}"
MASTER_PORT="${MASTER_PORT:-23147}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Method switches:
#   USE_DECODER=True  USE_TRAJECTORY_CONSISTENCY=True  -> simcot+sircl
#   USE_DECODER=True  USE_TRAJECTORY_CONSISTENCY=False -> simcot
#   USE_DECODER=False USE_TRAJECTORY_CONSISTENCY=False -> codi
#   USE_DECODER=False USE_TRAJECTORY_CONSISTENCY=True  -> codi+sircl
USE_DECODER="${USE_DECODER:-True}"
USE_TRAJECTORY_CONSISTENCY="${USE_TRAJECTORY_CONSISTENCY:-False}"

case "${USE_DECODER}-${USE_TRAJECTORY_CONSISTENCY}" in
	True-True)   METHOD_TAG="simcot_sircl" ;;
	True-False)  METHOD_TAG="simcot"        ;;
	False-False) METHOD_TAG="codi"          ;;
	False-True)  METHOD_TAG="codi_sircl"   ;;
	*) echo "[qwen3-train] ERROR: invalid USE_DECODER/USE_TRAJECTORY_CONSISTENCY combo: ${USE_DECODER}/${USE_TRAJECTORY_CONSISTENCY} (expect True/False)" >&2; exit 1 ;;
esac

EXPT_NAME="${EXPT_NAME}__${METHOD_TAG}"
SAVE_DIR="${SAVE_DIR}/${METHOD_TAG}"
LOG_DIR="${SAVE_DIR}/logs/${EXPT_NAME}-logs"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

CMD=(
	torchrun --nnodes 1 --master_port "${MASTER_PORT}" --nproc_per_node "${NPROC_PER_NODE}" train.py
	--output_dir "${SAVE_DIR}"
	--expt_name "${EXPT_NAME}"
	--logging_dir "${LOG_DIR}"
	--logging_steps 10
	--model_name_or_path "${MODEL_PATH}"
	--data_name icot
	--seed 11
	--model_max_length 512
	--per_device_train_batch_size 96
	--gradient_accumulation_steps 1
	--bf16
	--dataloader_num_workers 4
	--dataloader_pin_memory True
	--dataloader_persistent_workers True
	--dataloader_prefetch_factor 2
	--num_train_epochs 10
	--learning_rate 3e-4
	--max_grad_norm 1.0
	--use_lora True
	--lora_r 128
	--lora_alpha 32
	--lora_dropout 0.0
	--lora_init
	--save_strategy epoch
	--save_total_limit 30
	--save_safetensors False
	--weight_decay 0.1
	--warmup_ratio 0.05
	--lr_scheduler_type cosine
	--gradient_checkpointing True
	--ddp_find_unused_parameters False
	--do_train
	--report_to tensorboard
	--num_latent 6
	--logging_strategy steps
	--use_prj True
	--prj_dim 1024
	--prj_dropout 0.0
	--distill_loss_div_std True
	--exp_mode False
	--exp_data_num 200
	--remove_eos True
	--distill_loss_factor 20
	--explain_loss_factor 1.0
	--print_ref_model_stats False
	--max_token_num 200
	--use_decoder "${USE_DECODER}"
	--use_trajectory_consistency "${USE_TRAJECTORY_CONSISTENCY}"
	--trajectory_space_type euclidean
	--trajectory_radius_threshold 2
	--trajectory_loss_factor 0.05
	--inf_latent_iterations 6
	--greedy True
	--run_test_on_save "${RUN_TEST_ON_SAVE:-True}"
	--periodic_test_datasets "${PERIODIC_TEST_DATASETS:-gsm8k}"
	--periodic_test_batch_size "${PERIODIC_TEST_BATCH_SIZE:-32}"
)

printf '[qwen3-train] command='
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
