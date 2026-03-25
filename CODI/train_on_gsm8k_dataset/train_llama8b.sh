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

VARIANT="${CODI_METHOD_VARIANT:-simcon}"
FORCE_SINGLE_GPU=0
PER_DEVICE_BATCH_OVERRIDE=""
GRAD_ACC_OVERRIDE=""
NUM_EPOCHS_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sircl)
      VARIANT="simcon_sircl"
      shift
      ;;
    --single-gpu)
      FORCE_SINGLE_GPU=1
      shift
      ;;
    --per-device-batch)
      if [[ $# -lt 2 ]]; then
        echo "Error: --per-device-batch requires an integer value"
        exit 1
      fi
      PER_DEVICE_BATCH_OVERRIDE="$2"
      shift 2
      ;;
    --grad-acc)
      if [[ $# -lt 2 ]]; then
        echo "Error: --grad-acc requires an integer value"
        exit 1
      fi
      GRAD_ACC_OVERRIDE="$2"
      shift 2
      ;;
    --epochs)
      if [[ $# -lt 2 ]]; then
        echo "Error: --epochs requires an integer value"
        exit 1
      fi
      NUM_EPOCHS_OVERRIDE="$2"
      shift 2
      ;;
    --variant)
      if [[ $# -lt 2 ]]; then
        echo "Error: --variant requires a value: simcon or simcon_sircl"
        exit 1
      fi
      VARIANT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--sircl] [--single-gpu] [--per-device-batch N] [--grad-acc N] [--epochs N] [--variant simcon|simcon_sircl]"
      exit 1
      ;;
  esac
done

bash "${SCRIPT_DIR}/prepare_assets.sh" --models llama8b --force-datasets

export CODI_SAVE_DIR="${CODI_MULTIMODEL_SAVE_DIR}"
export CODI_RESULT_DIR="${CODI_MULTIMODEL_RESULT_DIR}"
export CODI_CACHE_DIR="${CODI_MULTIMODEL_CACHE_DIR}"

NNODES="${CODI_TRAIN_NNODES:-1}"
NPROC_PER_NODE="${CODI_TRAIN_NPROC_PER_NODE:-4}"
TARGET_NPROC_PER_NODE="${CODI_TARGET_NPROC_PER_NODE:-4}"
BASE_MASTER_PORT="${MASTER_PORT:-22505}"
SIRCL_MASTER_PORT="${CODI_SIRCL_MASTER_PORT:-22506}"
MODEL_PATH="${CODI_MM_LLAMA8B_PATH}"
MODEL_NAME="${MODEL_PATH##*/}"
EXPT_PREFIX="${CODI_MULTIMODEL_TAG}_gsm8k_llama31_8b"
SEED=11
MODEL_MAX_LENGTH=512
PER_DEVICE_BATCH_DEFAULT=2
GRAD_ACC_DEFAULT=8
PER_DEVICE_BATCH="${PER_DEVICE_BATCH_OVERRIDE:-${CODI_PER_DEVICE_BATCH:-${PER_DEVICE_BATCH_DEFAULT}}}"
GRAD_ACC="${GRAD_ACC_OVERRIDE:-${CODI_GRAD_ACC:-${GRAD_ACC_DEFAULT}}}"
NUM_EPOCHS_DEFAULT=8
NUM_EPOCHS="${NUM_EPOCHS_OVERRIDE:-${CODI_NUM_EPOCHS:-${NUM_EPOCHS_DEFAULT}}}"
LEARNING_RATE=0.0001
PRJ_DIM=4096
NUM_LATENT=6
PRINT_REF_MODEL_STATS=True
SAVE_STRATEGY="${CODI_SAVE_STRATEGY:-epoch}"
SAVE_TOTAL_LIMIT="${CODI_SAVE_TOTAL_LIMIT:-${NUM_EPOCHS}}"
SAVE_STEPS="${CODI_SAVE_STEPS:-100}"
POST_TRAIN_EVAL="${CODI_POST_TRAIN_EVAL:-1}"
POST_TRAIN_DATASETS="${CODI_POST_TRAIN_DATASETS:-gsm8k math500 aime svamp gsm-hard asdiv}"
DEFAULT_EVAL_BATCH_SIZE=4
EVAL_BATCH_SIZE="${CODI_EVAL_BATCH_SIZE:-${DEFAULT_EVAL_BATCH_SIZE}}"
SIRCL_SPACE_TYPE="${CODI_SIRCL_SPACE_TYPE:-euclidean}"
SIRCL_RADIUS_THRESHOLD="${CODI_SIRCL_RADIUS_THRESHOLD:-2}"
SIRCL_LOSS_FACTOR="${CODI_SIRCL_LOSS_FACTOR:-0.1}"

if [[ "${FORCE_SINGLE_GPU}" == "1" ]]; then
  NPROC_PER_NODE=1
fi

if (( TARGET_NPROC_PER_NODE <= 0 || NPROC_PER_NODE <= 0 )); then
  echo "Invalid GPU config: TARGET_NPROC_PER_NODE=${TARGET_NPROC_PER_NODE}, NPROC_PER_NODE=${NPROC_PER_NODE}"
  exit 1
fi

if (( PER_DEVICE_BATCH <= 0 || GRAD_ACC <= 0 || NUM_EPOCHS <= 0 )); then
  echo "Invalid train config: PER_DEVICE_BATCH=${PER_DEVICE_BATCH}, GRAD_ACC=${GRAD_ACC}, NUM_EPOCHS=${NUM_EPOCHS}"
  exit 1
fi

if (( TARGET_NPROC_PER_NODE % NPROC_PER_NODE != 0 )); then
  echo "Current setup is unsupported because TARGET_NPROC_PER_NODE (${TARGET_NPROC_PER_NODE}) is not divisible by NPROC_PER_NODE (${NPROC_PER_NODE})"
  exit 1
fi

GRAD_ACC_EFFECTIVE=$((GRAD_ACC * TARGET_NPROC_PER_NODE / NPROC_PER_NODE))
GLOBAL_BATCH_EFFECTIVE=$((PER_DEVICE_BATCH * NPROC_PER_NODE * GRAD_ACC_EFFECTIVE))

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model path is not ready: ${MODEL_PATH}"
  echo "Run: bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama8b --force-datasets"
  exit 1
fi

if [[ ! -f "${CODI_MULTIMODEL_ICOT_CACHE_DIR}/dataset_icot_0a5b3650760a22ea.pt" ]]; then
  echo "Required icot cache is missing under: ${CODI_MULTIMODEL_ICOT_CACHE_DIR}"
  echo "Run: bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama8b --force-datasets"
  exit 1
fi

is_post_train_eval_enabled() {
  case "${POST_TRAIN_EVAL}" in
    0|false|FALSE|no|NO)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

run_post_train_eval() {
  local variant_name="$1"
  local checkpoint_root="$2"
  local sweep_result_dir="$3"
  local -a checkpoints=()
  local ckpt_dir

  if [[ ! -d "${checkpoint_root}" ]]; then
    echo "Checkpoint root does not exist for ${variant_name}: ${checkpoint_root}"
    exit 1
  fi

  while IFS= read -r ckpt_dir; do
    checkpoints+=("${ckpt_dir}")
  done < <(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

  if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "No checkpoint-* directories found for ${variant_name} under: ${checkpoint_root}"
    exit 1
  fi

  mkdir -p "${sweep_result_dir}"

  echo "=================================================================="
  echo "Variant               : ${variant_name}"
  echo "Post-train datasets   : ${POST_TRAIN_DATASETS}"
  echo "Sweep result dir      : ${sweep_result_dir}"
  echo "Checkpoint count      : ${#checkpoints[@]}"
  echo "Eval batch size       : ${EVAL_BATCH_SIZE}"
  echo "=================================================================="

  pushd "${CODI_DIR}" >/dev/null
  for ckpt_dir in "${checkpoints[@]}"; do
    echo "[eval][${variant_name}] $(basename "${ckpt_dir}")"
    python "${CODI_DIR}/test_multi_dataset.py" \
      --model_name_or_path "${MODEL_PATH}" \
      --ckpt_dir "${ckpt_dir}" \
      --datasets "${POST_TRAIN_DATASETS}" \
      --num_runs 1 \
      --result_dir "${sweep_result_dir}" \
      --seed "${SEED}" \
      --model_max_length "${MODEL_MAX_LENGTH}" \
      --bf16 \
      --lora_r 128 \
      --lora_alpha 32 \
      --lora_init \
      --batch_size "${EVAL_BATCH_SIZE}" \
      --greedy True \
      --num_latent "${NUM_LATENT}" \
      --use_prj True \
      --prj_dim "${PRJ_DIM}" \
      --prj_no_ln False \
      --prj_dropout 0.0 \
      --inf_latent_iterations 6 \
      --remove_eos True \
      --use_lora True
  done
  popd >/dev/null
}

run_variant() {
  local variant_name="$1"
  local use_trajectory="$2"
  local master_port="$3"
  local expt_name="${EXPT_PREFIX}_${variant_name}"
  local checkpoint_root="${CODI_SAVE_DIR}/${expt_name}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_${SEED}"
  local sweep_result_dir="${CODI_MULTIMODEL_RESULT_DIR}/checkpoint_sweeps/${expt_name}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_${SEED}"
  local -a train_cmd=(
    torchrun
    --nnodes "${NNODES}"
    --master_port "${master_port}"
    --nproc_per_node "${NPROC_PER_NODE}"
    "${CODI_DIR}/train.py"
    --output_dir "${CODI_SAVE_DIR}"
    --expt_name "${expt_name}"
    --logging_dir "${CODI_MULTIMODEL_LOG_DIR}/${expt_name}"
    --logging_steps 10
    --model_name_or_path "${MODEL_PATH}"
    --data_name icot
    --seed "${SEED}"
    --model_max_length "${MODEL_MAX_LENGTH}"
    --per_device_train_batch_size "${PER_DEVICE_BATCH}"
    --gradient_accumulation_steps "${GRAD_ACC_EFFECTIVE}"
    --bf16
    --dataloader_num_workers 32
    --dataloader_pin_memory True
    --dataloader_persistent_workers True
    --dataloader_prefetch_factor 2
    --num_train_epochs "${NUM_EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --max_grad_norm 2.0
    --use_lora True
    --lora_r 128
    --lora_alpha 32
    --lora_init
    --save_strategy "${SAVE_STRATEGY}"
  )

  if [[ "${SAVE_STRATEGY}" == "steps" ]]; then
    train_cmd+=(--save_steps "${SAVE_STEPS}")
  fi

  train_cmd+=(
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --save_safetensors False
    --weight_decay 0.1
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --do_train
    --report_to tensorboard
    --num_latent "${NUM_LATENT}"
    --logging_strategy steps
    --use_prj True
    --prj_dim "${PRJ_DIM}"
    --prj_dropout 0.0
    --distill_loss_div_std True
    --exp_mode False
    --exp_data_num 200
    --remove_eos True
    --distill_loss_factor 20
    --print_ref_model_stats "${PRINT_REF_MODEL_STATS}"
    --max_token_num 200
    --use_decoder True
    --use_trajectory_consistency "${use_trajectory}"
    --ddp_find_unused_parameters False
  )

  if [[ "${use_trajectory}" == "True" ]]; then
    train_cmd+=(
      --trajectory_space_type "${SIRCL_SPACE_TYPE}"
      --trajectory_radius_threshold "${SIRCL_RADIUS_THRESHOLD}"
      --trajectory_loss_factor "${SIRCL_LOSS_FACTOR}"
    )
  fi

  echo "=================================================================="
  echo "Variant               : ${variant_name}"
  echo "Stage root            : ${CODI_MULTIMODEL_ROOT}"
  echo "Model path            : ${MODEL_PATH}"
  echo "Output dir            : ${CODI_SAVE_DIR}"
  echo "Result dir            : ${CODI_RESULT_DIR}"
  echo "Cache dir             : ${CODI_CACHE_DIR}"
  echo "Expt name             : ${expt_name}"
  echo "GPUs/node             : ${NPROC_PER_NODE}"
  echo "Reference GPUs/node   : ${TARGET_NPROC_PER_NODE}"
  echo "Per-device batch      : ${PER_DEVICE_BATCH}"
  echo "Grad accum (base)     : ${GRAD_ACC}"
  echo "Grad accum (effective): ${GRAD_ACC_EFFECTIVE}"
  echo "Num epochs            : ${NUM_EPOCHS}"
  echo "Global batch effective: ${GLOBAL_BATCH_EFFECTIVE}"
  echo "Master port           : ${master_port}"
  echo "Use decoder           : True"
  echo "Use trajectory        : ${use_trajectory}"
  echo "Save strategy         : ${SAVE_STRATEGY}"
  echo "Save total limit      : ${SAVE_TOTAL_LIMIT}"
  echo "Save steps            : ${SAVE_STEPS} (used when strategy=steps)"
  echo "Post-train datasets   : ${POST_TRAIN_DATASETS}"
  echo "Checkpoint root       : ${checkpoint_root}"
  echo "Sweep result dir      : ${sweep_result_dir}"
  if [[ "${use_trajectory}" == "True" ]]; then
    echo "SIRCL space type      : ${SIRCL_SPACE_TYPE}"
    echo "SIRCL radius threshold: ${SIRCL_RADIUS_THRESHOLD}"
    echo "SIRCL loss factor     : ${SIRCL_LOSS_FACTOR}"
  fi
  echo "=================================================================="

  pushd "${CODI_DIR}" >/dev/null
  "${train_cmd[@]}"
  popd >/dev/null

  if is_post_train_eval_enabled; then
    run_post_train_eval "${variant_name}" "${checkpoint_root}" "${sweep_result_dir}"
  else
    echo "Post-train checkpoint evaluation is disabled by CODI_POST_TRAIN_EVAL=${POST_TRAIN_EVAL}"
  fi
}

echo "=================================================================="
echo "Selected variant      : ${VARIANT}"
echo "Model path            : ${MODEL_PATH}"
echo "Base master port      : ${BASE_MASTER_PORT}"
echo "SIRCL master port     : ${SIRCL_MASTER_PORT}"
echo "SIRCL loss factor     : ${SIRCL_LOSS_FACTOR}"
echo "=================================================================="

case "${VARIANT}" in
  simcon)
    run_variant "simcon" "False" "${BASE_MASTER_PORT}"
    ;;
  simcon_sircl)
    run_variant "simcon_sircl" "True" "${SIRCL_MASTER_PORT}"
    ;;
  *)
    echo "Unsupported variant: ${VARIANT}"
    echo "Expected one of: simcon, simcon_sircl"
    exit 1
    ;;
esac
