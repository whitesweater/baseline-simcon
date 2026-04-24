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

GPU_MEM_GB="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    print(0)
else:
    print(int(torch.cuda.get_device_properties(0).total_memory / 1024**3))
PY
)"

GPU_COUNT="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "${GPU_COUNT}" ]]; then
  GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
fi
if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -lt 1 ]]; then
  GPU_COUNT=1
fi
GPU_COUNT=4
LAUNCHER=(python)

if (( GPU_COUNT > 1 )); then
  LAUNCHER=(torchrun --standalone --nnodes 1 --nproc_per_node "${GPU_COUNT}")
  if [[ "${SLURM_JOB_PARTITION:-}" == "debug" ]]; then
    EXPT_NAME="gsm8k_gpt_latent_decoder-2-debug-ddp"
    PER_DEVICE_TRAIN_BATCH_SIZE=16
    GRADIENT_ACCUMULATION_STEPS=1
  elif (( GPU_MEM_GB >= 70 )); then
    EXPT_NAME="gsm8k_gpt_latent_decoder-2-ddp"
    PER_DEVICE_TRAIN_BATCH_SIZE=96
    GRADIENT_ACCUMULATION_STEPS=1
  else
    EXPT_NAME="gsm8k_gpt_latent_decoder-2-ddp"
    PER_DEVICE_TRAIN_BATCH_SIZE=32
    GRADIENT_ACCUMULATION_STEPS=1
  fi
elif [[ "${SLURM_JOB_PARTITION:-}" == "debug" ]]; then
  EXPT_NAME="gsm8k_gpt_latent_decoder-2-debug"
  PER_DEVICE_TRAIN_BATCH_SIZE=32
  GRADIENT_ACCUMULATION_STEPS=1
elif (( GPU_MEM_GB >= 70 )); then
  EXPT_NAME="gsm8k_gpt_latent_decoder-2"
  PER_DEVICE_TRAIN_BATCH_SIZE=128
  GRADIENT_ACCUMULATION_STEPS=1
else
  EXPT_NAME="gsm8k_gpt_latent_decoder-2"
  PER_DEVICE_TRAIN_BATCH_SIZE=64
  GRADIENT_ACCUMULATION_STEPS=1
fi

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  CPUS_PER_PROC=$(( SLURM_CPUS_PER_TASK / GPU_COUNT ))
  if (( CPUS_PER_PROC < 2 )); then
    DATALOADER_NUM_WORKERS=1
  else
    DATALOADER_NUM_WORKERS=$(( CPUS_PER_PROC - 1 ))
  fi
else
  DATALOADER_NUM_WORKERS=8
fi
if (( DATALOADER_NUM_WORKERS > 16 )); then
  DATALOADER_NUM_WORKERS=16
fi

LOG_DIR="${STAGE_ROOT}/logs/${EXPT_NAME}"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "[gpt2-train] partition=${SLURM_JOB_PARTITION:-manual}"
echo "[gpt2-train] gpu_mem_gb=${GPU_MEM_GB}"
echo "[gpt2-train] gpu_count=${GPU_COUNT}"
echo "[gpt2-train] model_path=${MODEL_PATH}"
echo "[gpt2-train] save_dir=${SAVE_DIR}"
echo "[gpt2-train] log_dir=${LOG_DIR}"
echo "[gpt2-train] hf_datasets_cache=${HF_DATASETS_CACHE}"
echo "[gpt2-train] launcher=${LAUNCHER[*]}"
echo "[gpt2-train] batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "[gpt2-train] grad_acc=${GRADIENT_ACCUMULATION_STEPS}"
echo "[gpt2-train] save_strategy=epoch"
echo "[gpt2-train] dataloader_num_workers=${DATALOADER_NUM_WORKERS}"

# cp scripts/train_28.20_ce_noref_new_noaux_lat6.sh "$SAVE_DIR"
# /fs-computility/mllm/shared/weixilin/coconut/ckpts/gsm_cot/gsm-cot/checkpoint_13
"${LAUNCHER[@]}" train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name "$EXPT_NAME" \
	--logging_dir "$LOG_DIR"\
	--logging_steps 10 \
	--model_name_or_path "$MODEL_PATH" \
	--data_name icot \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  	--gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
	--bf16 \
	--num_train_epochs 40 \
	--dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
	--dataloader_pin_memory True \
	--dataloader_persistent_workers True \
	--dataloader_prefetch_factor 2 \
	--learning_rate 3e-3 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--overwrite_output_dir True \
	--save_strategy epoch \
	--save_safetensors False \
	--save_total_limit 2 \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to tensorboard \
    --num_latent 6 \
    --logging_strategy "steps" \
	--use_prj True \
	--prj_dim 768 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--exp_data_num 2000 \
	--remove_eos True \
	--print_ref_model_stats True \
	--use_decoder True \
	--use_trajectory_consistency True \
	--trajectory_space_type euclidean \
	--trajectory_radius_threshold 2 \
	--trajectory_loss_factor 0.1
