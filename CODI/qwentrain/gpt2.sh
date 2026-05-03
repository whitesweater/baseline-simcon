#!/bin/bash
set -euo pipefail

source /hpc2ssd/JH_DATA/spooler/yhao481/.upload/proj/baseline/.venv/bin/activate
cd /hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI

STAGE_ROOT=/hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1
SAVE_DIR="${STAGE_ROOT}/outputs/gpt2"
MODEL_PATH="${STAGE_ROOT}/models/gpt2"
HF_HOME="${STAGE_ROOT}/hf_home"
HF_DATASETS_CACHE="${STAGE_ROOT}/hf_datasets_cache"
TRANSFORMERS_CACHE="${HF_HOME}/transformers"
MODELSCOPE_CACHE="${STAGE_ROOT}/modelscope_cache"
CODI_CACHE_DIR="${STAGE_ROOT}/cache"
CODI_TOKENIZED_CACHE_DIR="${CODI_CACHE_DIR}/tokenized"
CODI_GSM8K_AUG_CACHE_DIR="${HF_DATASETS_CACHE}"
CODI_GSM8K_AUG_HF_ID="zen-E/GSM8k-Aug"

export HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE MODELSCOPE_CACHE
export CODI_CACHE_DIR CODI_TOKENIZED_CACHE_DIR CODI_GSM8K_AUG_CACHE_DIR CODI_GSM8K_AUG_HF_ID
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

EXPT_NAME="${EXPT_NAME:-gsm8k_gpt_latent_decoder-2-ddp}"
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
TRAJECTORY_RADIUS_THRESHOLD="${TRAJECTORY_RADIUS_THRESHOLD:-2}"

case "${USE_DECODER}-${USE_TRAJECTORY_CONSISTENCY}" in
	True-True)   METHOD_TAG="simcot_sircl" ;;
	True-False)  METHOD_TAG="simcot"        ;;
	False-False) METHOD_TAG="codi"          ;;
	False-True)  METHOD_TAG="codi_sircl"   ;;
	*) echo "[gpt2-train] ERROR: invalid USE_DECODER/USE_TRAJECTORY_CONSISTENCY combo: ${USE_DECODER}/${USE_TRAJECTORY_CONSISTENCY} (expect True/False)" >&2; exit 1 ;;
esac

EXPT_NAME="${EXPT_NAME}__${METHOD_TAG}"
SAVE_DIR="${SAVE_DIR}/${METHOD_TAG}"
LOG_DIR="${SAVE_DIR}/logs/${EXPT_NAME}-logs"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

CMD=(
	torchrun --standalone --nnodes 1 --master_port "${MASTER_PORT}" --nproc_per_node "${NPROC_PER_NODE}" train.py
	--output_dir "$SAVE_DIR"
	--expt_name "$EXPT_NAME"
	--logging_dir "$LOG_DIR"
	--logging_steps 10
	--model_name_or_path "$MODEL_PATH"
	--data_name icot
	--seed 11
	--model_max_length 512
	--per_device_train_batch_size 96
	--gradient_accumulation_steps 1
	--bf16
	--num_train_epochs 40
	--dataloader_num_workers 8
	--dataloader_pin_memory True
	--dataloader_persistent_workers True
	--dataloader_prefetch_factor 2
	--learning_rate 3e-3
	--max_grad_norm 2.0
	--use_lora True
	--lora_r 128 --lora_alpha 32 --lora_init
	--overwrite_output_dir True
	--save_strategy epoch
	--save_safetensors False
	--save_total_limit 40
	--weight_decay 0.1
	--warmup_ratio 0.03
	--lr_scheduler_type cosine
	--do_train
	--report_to tensorboard
	--num_latent 6
	--logging_strategy steps
	--use_prj True
	--prj_dim 768
	--prj_dropout 0.0
	--distill_loss_div_std True
	--exp_mode False
	--exp_data_num 2000
	--remove_eos True
	--print_ref_model_stats True
	--use_decoder "$USE_DECODER"
	--use_trajectory_consistency "$USE_TRAJECTORY_CONSISTENCY"
	--trajectory_space_type euclidean
	--trajectory_radius_threshold "$TRAJECTORY_RADIUS_THRESHOLD"
	--trajectory_loss_factor 0.1
	--inf_latent_iterations 6
	--greedy True
	--run_test_on_save "${RUN_TEST_ON_SAVE:-True}"
	--periodic_test_datasets "${PERIODIC_TEST_DATASETS:-gsm8k}"
	--periodic_test_batch_size "${PERIODIC_TEST_BATCH_SIZE:-32}"
	"$@"
)

printf '[gpt2-train] command='
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
