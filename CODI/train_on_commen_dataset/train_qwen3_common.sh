#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENTRY_SCRIPT_PATH="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
ENTRY_SCRIPT_NAME="$(basename "${ENTRY_SCRIPT_PATH}")"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

USER_HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}"
USER_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-}"
USER_HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-}"

if [[ ! -f "${CODI_VENV_PATH}" ]]; then
  echo "Error: CODI_VENV_PATH is invalid: ${CODI_VENV_PATH}"
  exit 1
fi
# shellcheck disable=SC1091
source "${CODI_VENV_PATH}"

SEED=11
DATA_NAME="${CODI_QWEN3_DATA_NAME:-commonsense}"
VARIANT="${CODI_METHOD_VARIANT:-simcon}"
BASE_VARIANT_NAME="simcon"
SIRCL_VARIANT_NAME="simcon_sircl"
VARIANT_USAGE="simcon|simcon_sircl"
USE_DECODER="True"
BASE_MASTER_PORT_DEFAULT=22527
SIRCL_MASTER_PORT_DEFAULT=22528
QWEN3_MODEL_KEY="${CODI_QWEN3_MODEL_KEY:-qwen3_1p7b}"

MODEL_KEY=""
MODEL_DISPLAY_NAME=""
MODEL_PATH=""
EXPT_PREFIX=""
MODEL_MAX_LENGTH=512
LEARNING_RATE="${CODI_LEARNING_RATE:-0.0005}"
PRJ_DIM=0
NUM_LATENT="${CODI_NUM_LATENT:-6}"
PRINT_REF_MODEL_STATS="${CODI_PRINT_REF_MODEL_STATS:-False}"
PRINT_LOSS="${CODI_PRINT_LOSS:-False}"
DISTILL_LOSS_DIV_STD="${CODI_DISTILL_LOSS_DIV_STD:-True}"
DISTILL_LOSS_FACTOR="${CODI_DISTILL_LOSS_FACTOR:-20}"
EXPLAIN_LOSS_FACTOR="${CODI_EXPLAIN_LOSS_FACTOR:-1.0}"
REF_LOSS_FACTOR="${CODI_REF_LOSS_FACTOR:-1.0}"
MAX_TOKEN_NUM="${CODI_MAX_TOKEN_NUM:-200}"
DDP_FIND_UNUSED_PARAMETERS="${CODI_DDP_FIND_UNUSED_PARAMETERS:-False}"
SAVE_STRATEGY_DEFAULT="epoch"
SAVE_STEPS_DEFAULT=100
DEFAULT_EVAL_BATCH_SIZE=16
PER_DEVICE_BATCH_DEFAULT=0
GRAD_ACC_DEFAULT=0
NUM_EPOCHS_DEFAULT=0
FORCE_SINGLE_GPU=0
PER_DEVICE_BATCH_OVERRIDE=""
GRAD_ACC_OVERRIDE=""
NUM_EPOCHS_OVERRIDE=""
MAX_STEPS_OVERRIDE=""
DRY_RUN=0
EXPT_SUFFIX="${CODI_EXPT_SUFFIX:-}"

if [[ -n "${EXPT_SUFFIX}" && "${EXPT_SUFFIX}" != _* ]]; then
  EXPT_SUFFIX="_${EXPT_SUFFIX}"
fi

case "${QWEN3_MODEL_KEY}" in
  qwen3|qwen3_4b)
    MODEL_KEY="qwen3"
    MODEL_DISPLAY_NAME="Qwen3-4B"
    MODEL_PATH="${CODI_MM_QWEN3_PATH}"
    EXPT_PREFIX="${CODI_COMMONSENSE_STAGE_TAG}_${DATA_NAME}_qwen3_4b"
    PRJ_DIM=2560
    DEFAULT_EVAL_BATCH_SIZE=8
    PER_DEVICE_BATCH_DEFAULT=8
    GRAD_ACC_DEFAULT=2
    NUM_EPOCHS_DEFAULT=8
    ;;
  qwen3_0p6b)
    MODEL_KEY="qwen3_0p6b"
    MODEL_DISPLAY_NAME="Qwen3-0.6B"
    MODEL_PATH="${CODI_MM_QWEN3_0P6B_PATH}"
    EXPT_PREFIX="${CODI_COMMONSENSE_STAGE_TAG}_${DATA_NAME}_qwen3_0p6b"
    PRJ_DIM=1024
    DEFAULT_EVAL_BATCH_SIZE=32
    PER_DEVICE_BATCH_DEFAULT=32
    GRAD_ACC_DEFAULT=2
    NUM_EPOCHS_DEFAULT=10
    ;;
  qwen3_1p7b)
    MODEL_KEY="qwen3_1p7b"
    MODEL_DISPLAY_NAME="Qwen3-1.7B"
    MODEL_PATH="${CODI_MM_QWEN3_1P7B_PATH}"
    EXPT_PREFIX="${CODI_COMMONSENSE_STAGE_TAG}_${DATA_NAME}_qwen3_1p7b"
    PRJ_DIM=2048
    DEFAULT_EVAL_BATCH_SIZE=16
    PER_DEVICE_BATCH_DEFAULT=16
    GRAD_ACC_DEFAULT=2
    NUM_EPOCHS_DEFAULT=10
    ;;
  *)
    echo "Error: unsupported CODI_QWEN3_MODEL_KEY=${QWEN3_MODEL_KEY}"
    echo "Supported model keys: qwen3_4b, qwen3_0p6b, qwen3_1p7b"
    exit 1
    ;;
esac

MODEL_NAME="${MODEL_PATH##*/}"
MODEL_MANIFEST_PATH="${CODI_COMMONSENSE_MANIFEST_DIR}/${MODEL_NAME}.manifest.json"

usage() {
  echo "Usage: ${ENTRY_SCRIPT_NAME} [--sircl] [--single-gpu] [--per-device-batch N] [--grad-acc N] [--epochs N] [--max-steps N] [--dry-run] [--variant ${VARIANT_USAGE}]"
}

require_existing_path() {
  local path="$1"
  local description="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Error: missing ${description}: ${path}"
    exit 1
  fi
}

require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "Error: required command is not available: ${name}"
    exit 1
  fi
}

configure_dataset_cache() {
  if [[ -n "${USER_HF_DATASETS_CACHE}" && -d "${USER_HF_DATASETS_CACHE}" ]]; then
    export HF_DATASETS_CACHE="${USER_HF_DATASETS_CACHE}"
    export HF_HUB_OFFLINE="${USER_HF_HUB_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${USER_HF_DATASETS_OFFLINE:-1}"
  fi
}

run_qwen3_preflight() {
  echo "[preflight] validating ${MODEL_DISPLAY_NAME} runtime"
  pushd "${CODI_DIR}" >/dev/null
  MODEL_PATH="${MODEL_PATH}" python - <<'PY'
import logging
import os

from transformers import AutoConfig, AutoTokenizer

from src.tokenizer_utils import load_tokenizer_with_fallback

model_path = os.environ["MODEL_PATH"]
logging.basicConfig(level=logging.WARNING)

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(f"[preflight] AutoConfig model_type={config.model_type}")
if config.model_type != "qwen3":
    raise SystemExit(f"[preflight] expected model_type=qwen3, got {config.model_type}")

slow_failed = False
try:
    slow_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print(f"[preflight] slow tokenizer loaded directly: {type(slow_tokenizer).__name__}")
except Exception as exc:
    slow_failed = True
    print(f"[preflight] slow tokenizer failed as expected: {exc.__class__.__name__}: {exc}")

tokenizer = load_tokenizer_with_fallback(model_path, use_fast=False)
tokenizer_name = type(tokenizer).__name__
print(
    f"[preflight] tokenizer resolution={tokenizer_name}, "
    f"pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}"
)

if slow_failed and not tokenizer_name.endswith("Fast"):
    raise SystemExit(
        "[preflight] slow tokenizer failed, but fallback did not resolve to a fast tokenizer."
    )
PY
  popd >/dev/null
}

validate_runtime_state() {
  require_existing_path "${CODI_VENV_PATH}" "CODI_VENV_PATH"
  require_existing_path "${CODI_COMMONSENSE_ROOT}" "CODI_COMMONSENSE_ROOT"
  require_existing_path "${MODEL_PATH}" "${MODEL_DISPLAY_NAME} model directory"
  require_existing_path "${MODEL_PATH}/config.json" "${MODEL_DISPLAY_NAME} config.json"
  require_existing_path "${MODEL_MANIFEST_PATH}" "${MODEL_DISPLAY_NAME} manifest"
  require_command python
  require_command torchrun
  run_qwen3_preflight
}

warn_missing_runtime_state_for_dry_run() {
  local missing=0
  local path
  local description

  for entry in \
    "${CODI_VENV_PATH}:CODI_VENV_PATH" \
    "${CODI_COMMONSENSE_ROOT}:CODI_COMMONSENSE_ROOT" \
    "${MODEL_PATH}:${MODEL_DISPLAY_NAME} model directory" \
    "${MODEL_PATH}/config.json:${MODEL_DISPLAY_NAME} config.json" \
    "${MODEL_MANIFEST_PATH}:${MODEL_DISPLAY_NAME} manifest"; do
    path="${entry%%:*}"
    description="${entry#*:}"
    if [[ ! -e "${path}" ]]; then
      echo "[dry-run][warn] missing ${description}: ${path}"
      missing=1
    fi
  done

  for name in python torchrun; do
    if ! command -v "${name}" >/dev/null 2>&1; then
      echo "[dry-run][warn] required command is not available: ${name}"
      missing=1
    fi
  done

  if (( missing == 0 )); then
    run_qwen3_preflight
  else
    echo "[dry-run] runtime validation skipped because stage assets are not fully ready yet"
  fi
}

print_command() {
  local label="$1"
  shift
  printf '%s' "${label}"
  printf ' %q' "$@"
  printf '\n'
}

preview_eval_commands() {
  local checkpoint_root="$1"
  local sweep_result_dir="$2"
  local -a checkpoints=()
  local ckpt_dir

  if [[ -d "${checkpoint_root}" ]]; then
    while IFS= read -r ckpt_dir; do
      checkpoints+=("${ckpt_dir}")
    done < <(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  fi

  if [[ ${#checkpoints[@]} -eq 0 ]]; then
    checkpoints=("${checkpoint_root}/checkpoint-<step>")
  fi

  for ckpt_dir in "${checkpoints[@]}"; do
    print_command "[dry-run][eval]" \
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
}

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

is_incremental_post_train_eval_enabled() {
  if ! is_post_train_eval_enabled; then
    return 1
  fi

  case "${EVAL_EACH_CHECKPOINT}" in
    0|false|FALSE|no|NO)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

get_eval_manifest_path() {
  local sweep_result_dir="$1"
  echo "${sweep_result_dir}/evaluated_checkpoints.txt"
}

is_checkpoint_ready_for_eval() {
  local ckpt_dir="$1"
  [[ -d "${ckpt_dir}" ]] || return 1
  [[ -f "${ckpt_dir}/trainer_state.json" ]] || return 1
  [[ -f "${ckpt_dir}/pytorch_model.bin" || -f "${ckpt_dir}/model.safetensors" ]]
}

has_checkpoint_been_evaluated() {
  local ckpt_dir="$1"
  local manifest_path="$2"

  [[ -f "${manifest_path}" ]] || return 1
  grep -Fxq "${ckpt_dir}" "${manifest_path}"
}

mark_checkpoint_evaluated() {
  local ckpt_dir="$1"
  local manifest_path="$2"

  mkdir -p "$(dirname "${manifest_path}")"
  touch "${manifest_path}"
  if ! has_checkpoint_been_evaluated "${ckpt_dir}" "${manifest_path}"; then
    printf '%s\n' "${ckpt_dir}" >> "${manifest_path}"
  fi
}

run_single_checkpoint_eval() {
  local variant_name="$1"
  local ckpt_dir="$2"
  local sweep_result_dir="$3"

  echo "[eval][${variant_name}] $(basename "${ckpt_dir}")"
  pushd "${CODI_DIR}" >/dev/null
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
  popd >/dev/null
}

watch_post_train_eval() {
  local variant_name="$1"
  local checkpoint_root="$2"
  local sweep_result_dir="$3"
  local train_pid="$4"
  local manifest_path
  manifest_path="$(get_eval_manifest_path "${sweep_result_dir}")"

  mkdir -p "${sweep_result_dir}"
  touch "${manifest_path}"

  echo "[eval-watch][${variant_name}] polling every ${EVAL_POLL_INTERVAL}s while train pid=${train_pid} is running"

  while kill -0 "${train_pid}" 2>/dev/null; do
    if [[ -d "${checkpoint_root}" ]]; then
      while IFS= read -r ckpt_dir; do
        if ! is_checkpoint_ready_for_eval "${ckpt_dir}"; then
          continue
        fi
        if has_checkpoint_been_evaluated "${ckpt_dir}" "${manifest_path}"; then
          continue
        fi
        if run_single_checkpoint_eval "${variant_name}" "${ckpt_dir}" "${sweep_result_dir}"; then
          mark_checkpoint_evaluated "${ckpt_dir}" "${manifest_path}"
        else
          echo "[eval-watch][${variant_name}] deferred $(basename "${ckpt_dir}") after eval failure; final catch-up will retry"
        fi
      done < <(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
    fi
    sleep "${EVAL_POLL_INTERVAL}"
  done

  echo "[eval-watch][${variant_name}] training pid ${train_pid} exited; handing off to final catch-up sweep"
}

run_post_train_eval() {
  local variant_name="$1"
  local checkpoint_root="$2"
  local sweep_result_dir="$3"
  local -a checkpoints=()
  local ckpt_dir
  local manifest_path
  manifest_path="$(get_eval_manifest_path "${sweep_result_dir}")"

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
  echo "Eval manifest         : ${manifest_path}"
  echo "Checkpoint count      : ${#checkpoints[@]}"
  echo "Eval batch size       : ${EVAL_BATCH_SIZE}"
  echo "=================================================================="

  for ckpt_dir in "${checkpoints[@]}"; do
    if ! is_checkpoint_ready_for_eval "${ckpt_dir}"; then
      echo "[eval][skip][${variant_name}] $(basename "${ckpt_dir}") is not fully written yet"
      continue
    fi
    if has_checkpoint_been_evaluated "${ckpt_dir}" "${manifest_path}"; then
      echo "[eval][skip][${variant_name}] $(basename "${ckpt_dir}") already evaluated"
      continue
    fi
    run_single_checkpoint_eval "${variant_name}" "${ckpt_dir}" "${sweep_result_dir}"
    mark_checkpoint_evaluated "${ckpt_dir}" "${manifest_path}"
  done
}

run_variant() {
  local variant_name="$1"
  local use_trajectory="$2"
  local master_port="$3"
  local expt_name="${EXPT_PREFIX}_${variant_name}${EXPT_SUFFIX}"
  local checkpoint_root="${CODI_COMMONSENSE_SAVE_DIR}/${expt_name}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_${SEED}"
  local sweep_result_dir="${CODI_COMMONSENSE_RESULT_DIR}/checkpoint_sweeps/${expt_name}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_${SEED}"
  local -a train_cmd=(
    torchrun
    --nnodes "${NNODES}"
    --master_port "${master_port}"
    --nproc_per_node "${NPROC_PER_NODE}"
    "${CODI_DIR}/train.py"
    --output_dir "${CODI_COMMONSENSE_SAVE_DIR}"
    --expt_name "${expt_name}"
    --logging_dir "${CODI_COMMONSENSE_LOG_DIR}/${expt_name}"
    --logging_steps 10
    --model_name_or_path "${MODEL_PATH}"
    --data_name "${DATA_NAME}"
    --seed "${SEED}"
    --model_max_length "${MODEL_MAX_LENGTH}"
    --per_device_train_batch_size "${PER_DEVICE_BATCH}"
    --gradient_accumulation_steps "${GRAD_ACC_EFFECTIVE}"
    --bf16
    --num_train_epochs "${NUM_EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --max_grad_norm 2.0
    --use_lora True
    --lora_r 128
    --lora_alpha 32
    --lora_init
    --save_strategy "${SAVE_STRATEGY}"
  )

  if [[ -n "${MAX_STEPS_OVERRIDE}" ]]; then
    train_cmd+=(--max_steps "${MAX_STEPS_OVERRIDE}")
  fi

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
    --distill_loss_div_std "${DISTILL_LOSS_DIV_STD}"
    --exp_mode False
    --exp_data_num 200
    --remove_eos True
    --distill_loss_factor "${DISTILL_LOSS_FACTOR}"
    --explain_loss_factor "${EXPLAIN_LOSS_FACTOR}"
    --ref_loss_factor "${REF_LOSS_FACTOR}"
    --print_loss "${PRINT_LOSS}"
    --print_ref_model_stats "${PRINT_REF_MODEL_STATS}"
    --max_token_num "${MAX_TOKEN_NUM}"
    --use_decoder "${USE_DECODER}"
    --use_trajectory_consistency "${use_trajectory}"
    --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}"
  )

  if [[ "${use_trajectory}" == "True" ]]; then
    train_cmd+=(
      --trajectory_space_type "${SIRCL_SPACE_TYPE}"
      --trajectory_radius_threshold "${SIRCL_RADIUS_THRESHOLD}"
      --trajectory_loss_factor "${SIRCL_LOSS_FACTOR}"
    )
  fi

  echo "=================================================================="
  echo "Entry script          : ${ENTRY_SCRIPT_NAME}"
  echo "Variant               : ${variant_name}"
  echo "Stage root            : ${CODI_COMMONSENSE_ROOT}"
  echo "Data name             : ${DATA_NAME}"
  echo "Model name            : ${MODEL_DISPLAY_NAME}"
  echo "Model path            : ${MODEL_PATH}"
  echo "Manifest path         : ${MODEL_MANIFEST_PATH}"
  echo "Model key             : ${QWEN3_MODEL_KEY}"
  echo "Output dir            : ${CODI_COMMONSENSE_SAVE_DIR}"
  echo "Result dir            : ${CODI_COMMONSENSE_RESULT_DIR}"
  echo "Cache dir             : ${CODI_COMMONSENSE_CACHE_DIR}"
  echo "HF_DATASETS_CACHE     : ${HF_DATASETS_CACHE:-<unset>}"
  echo "Expt name             : ${expt_name}"
  echo "Experiment suffix     : ${EXPT_SUFFIX:-<none>}"
  echo "GPUs/node             : ${NPROC_PER_NODE}"
  echo "Reference GPUs/node   : ${TARGET_NPROC_PER_NODE}"
  echo "Per-device batch      : ${PER_DEVICE_BATCH}"
  echo "Grad accum (base)     : ${GRAD_ACC}"
  echo "Grad accum (effective): ${GRAD_ACC_EFFECTIVE}"
  echo "Num epochs            : ${NUM_EPOCHS}"
  echo "Num latent            : ${NUM_LATENT}"
  echo "Learning rate         : ${LEARNING_RATE}"
  echo "Distill/std normalize : ${DISTILL_LOSS_DIV_STD}"
  echo "Distill loss factor   : ${DISTILL_LOSS_FACTOR}"
  echo "Explain loss factor   : ${EXPLAIN_LOSS_FACTOR}"
  echo "Ref loss factor       : ${REF_LOSS_FACTOR}"
  echo "Print loss            : ${PRINT_LOSS}"
  echo "Print ref stats       : ${PRINT_REF_MODEL_STATS}"
  echo "Projection dim        : ${PRJ_DIM}"
  echo "Eval batch size       : ${EVAL_BATCH_SIZE}"
  echo "Max token num         : ${MAX_TOKEN_NUM}"
  echo "DDP find unused       : ${DDP_FIND_UNUSED_PARAMETERS}"
  echo "Max steps override    : ${MAX_STEPS_OVERRIDE:-<epoch-controlled>}"
  echo "Global batch effective: ${GLOBAL_BATCH_EFFECTIVE}"
  echo "Master port           : ${master_port}"
  echo "Use decoder           : ${USE_DECODER}"
  echo "Use trajectory        : ${use_trajectory}"
  echo "Dry run               : ${DRY_RUN}"
  echo "Save strategy         : ${SAVE_STRATEGY}"
  echo "Save total limit      : ${SAVE_TOTAL_LIMIT}"
  echo "Save steps            : ${SAVE_STEPS} (used when strategy=steps)"
  echo "Post-train eval flag  : ${POST_TRAIN_EVAL}"
  echo "Eval each checkpoint  : ${EVAL_EACH_CHECKPOINT}"
  echo "Eval poll interval    : ${EVAL_POLL_INTERVAL}s"
  echo "Post-train datasets   : ${POST_TRAIN_DATASETS}"
  echo "Checkpoint root       : ${checkpoint_root}"
  echo "Sweep result dir      : ${sweep_result_dir}"
  if [[ "${use_trajectory}" == "True" ]]; then
    echo "SIRCL space type      : ${SIRCL_SPACE_TYPE}"
    echo "SIRCL radius threshold: ${SIRCL_RADIUS_THRESHOLD}"
    echo "SIRCL loss factor     : ${SIRCL_LOSS_FACTOR}"
  fi
  echo "=================================================================="

  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command "[dry-run][train]" "${train_cmd[@]}"
    if is_post_train_eval_enabled; then
      if is_incremental_post_train_eval_enabled; then
        echo "[dry-run] checkpoints will be evaluated incrementally as they are saved"
      else
        echo "[dry-run] checkpoints will be evaluated only in the final catch-up sweep"
      fi
      preview_eval_commands "${checkpoint_root}" "${sweep_result_dir}"
    else
      echo "[dry-run] post-train checkpoint evaluation is disabled by CODI_POST_TRAIN_EVAL=${POST_TRAIN_EVAL}"
    fi
    return 0
  fi

  local train_pid=""
  local watcher_pid=""
  local train_status=0
  local watcher_status=0

  (
    cd "${CODI_DIR}"
    "${train_cmd[@]}"
  ) &
  train_pid=$!

  if is_incremental_post_train_eval_enabled; then
    watch_post_train_eval "${variant_name}" "${checkpoint_root}" "${sweep_result_dir}" "${train_pid}" &
    watcher_pid=$!
  fi

  set +e
  wait "${train_pid}"
  train_status=$?
  if [[ -n "${watcher_pid}" ]]; then
    wait "${watcher_pid}"
    watcher_status=$?
  fi
  set -e

  if is_post_train_eval_enabled; then
    run_post_train_eval "${variant_name}" "${checkpoint_root}" "${sweep_result_dir}"
  else
    echo "Post-train checkpoint evaluation is disabled by CODI_POST_TRAIN_EVAL=${POST_TRAIN_EVAL}"
  fi

  if (( watcher_status != 0 )); then
    echo "[warn] checkpoint watcher exited with status ${watcher_status}; final catch-up sweep completed anyway"
  fi

  if (( train_status != 0 )); then
    echo "Training failed for ${variant_name} with status ${train_status}"
    exit "${train_status}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sircl)
      VARIANT="${SIRCL_VARIANT_NAME}"
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
    --max-steps)
      if [[ $# -lt 2 ]]; then
        echo "Error: --max-steps requires an integer value"
        exit 1
      fi
      MAX_STEPS_OVERRIDE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --variant)
      if [[ $# -lt 2 ]]; then
        echo "Error: --variant requires a value: ${VARIANT_USAGE}"
        exit 1
      fi
      VARIANT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

case "${VARIANT}" in
  "${BASE_VARIANT_NAME}"|"${SIRCL_VARIANT_NAME}")
    ;;
  *)
    echo "Unsupported variant: ${VARIANT}"
    echo "Expected one of: ${VARIANT_USAGE}"
    exit 1
    ;;
esac

export CODI_SAVE_DIR="${CODI_COMMONSENSE_SAVE_DIR}"
export CODI_RESULT_DIR="${CODI_COMMONSENSE_RESULT_DIR}"
export CODI_CACHE_DIR="${CODI_COMMONSENSE_CACHE_DIR}"

NNODES="${CODI_TRAIN_NNODES:-1}"
NPROC_PER_NODE="${CODI_TRAIN_NPROC_PER_NODE:-4}"
TARGET_NPROC_PER_NODE="${CODI_TARGET_NPROC_PER_NODE:-4}"
BASE_MASTER_PORT="${MASTER_PORT:-${BASE_MASTER_PORT_DEFAULT}}"
SIRCL_MASTER_PORT="${CODI_SIRCL_MASTER_PORT:-${SIRCL_MASTER_PORT_DEFAULT}}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH_OVERRIDE:-${CODI_PER_DEVICE_BATCH:-${PER_DEVICE_BATCH_DEFAULT}}}"
GRAD_ACC="${GRAD_ACC_OVERRIDE:-${CODI_GRAD_ACC:-${GRAD_ACC_DEFAULT}}}"
NUM_EPOCHS="${NUM_EPOCHS_OVERRIDE:-${CODI_NUM_EPOCHS:-${NUM_EPOCHS_DEFAULT}}}"
SAVE_STRATEGY="${CODI_SAVE_STRATEGY:-${SAVE_STRATEGY_DEFAULT}}"
SAVE_TOTAL_LIMIT="${CODI_SAVE_TOTAL_LIMIT:-${NUM_EPOCHS}}"
SAVE_STEPS="${CODI_SAVE_STEPS:-${SAVE_STEPS_DEFAULT}}"
POST_TRAIN_EVAL="${CODI_POST_TRAIN_EVAL:-1}"
EVAL_EACH_CHECKPOINT="${CODI_EVAL_EACH_CHECKPOINT:-1}"
EVAL_POLL_INTERVAL="${CODI_EVAL_POLL_INTERVAL:-30}"
POST_TRAIN_DATASETS="${CODI_POST_TRAIN_DATASETS:-commonsense}"
EVAL_BATCH_SIZE="${CODI_EVAL_BATCH_SIZE:-${DEFAULT_EVAL_BATCH_SIZE}}"
SIRCL_SPACE_TYPE="${CODI_SIRCL_SPACE_TYPE:-euclidean}"
SIRCL_RADIUS_THRESHOLD="${CODI_SIRCL_RADIUS_THRESHOLD:-4}"
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

if [[ -n "${MAX_STEPS_OVERRIDE}" ]]; then
  if ! [[ "${MAX_STEPS_OVERRIDE}" =~ ^[0-9]+$ ]] || (( MAX_STEPS_OVERRIDE <= 0 )); then
    echo "Error: --max-steps must be a positive integer, got: ${MAX_STEPS_OVERRIDE}"
    exit 1
  fi
fi

if (( TARGET_NPROC_PER_NODE % NPROC_PER_NODE != 0 )); then
  echo "Current setup is unsupported because TARGET_NPROC_PER_NODE (${TARGET_NPROC_PER_NODE}) is not divisible by NPROC_PER_NODE (${NPROC_PER_NODE})"
  exit 1
fi

GRAD_ACC_EFFECTIVE=$((GRAD_ACC * TARGET_NPROC_PER_NODE / NPROC_PER_NODE))
GLOBAL_BATCH_EFFECTIVE=$((PER_DEVICE_BATCH * NPROC_PER_NODE * GRAD_ACC_EFFECTIVE))

configure_dataset_cache

if [[ "${DRY_RUN}" == "1" ]]; then
  warn_missing_runtime_state_for_dry_run
else
  validate_runtime_state
fi

echo "=================================================================="
echo "Selected variant      : ${VARIANT}"
echo "Selected model key    : ${QWEN3_MODEL_KEY}"
echo "Selected data name    : ${DATA_NAME}"
echo "Model name            : ${MODEL_DISPLAY_NAME}"
echo "Model path            : ${MODEL_PATH}"
echo "Manifest path         : ${MODEL_MANIFEST_PATH}"
echo "Projection dim        : ${PRJ_DIM}"
echo "Base master port      : ${BASE_MASTER_PORT}"
echo "SIRCL master port     : ${SIRCL_MASTER_PORT}"
echo "SIRCL loss factor     : ${SIRCL_LOSS_FACTOR}"
echo "Dry run               : ${DRY_RUN}"
echo "=================================================================="

case "${VARIANT}" in
  "${BASE_VARIANT_NAME}")
    run_variant "${BASE_VARIANT_NAME}" "False" "${BASE_MASTER_PORT}"
    ;;
  "${SIRCL_VARIANT_NAME}")
    run_variant "${SIRCL_VARIANT_NAME}" "True" "${SIRCL_MASTER_PORT}"
    ;;
esac
