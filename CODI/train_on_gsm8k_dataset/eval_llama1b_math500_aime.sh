#!/bin/bash
set -euo pipefail

CKPT_DIR="${1:-}"
if [[ -z "${CKPT_DIR}" ]]; then
  echo "Usage: $0 <ckpt_dir>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${CODI_DIR}/config.env" || { echo "Error: config.env not found. Copy config.env.example to config.env and configure."; exit 1; }
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
# shellcheck disable=SC1091
source "${CODI_VENV_PATH}" || { echo "Error: CODI_VENV_PATH is invalid: ${CODI_VENV_PATH}"; exit 1; }

bash "${SCRIPT_DIR}/prepare_assets.sh" --models llama1b --force-datasets

export CODI_RESULT_DIR="${CODI_MULTIMODEL_RESULT_DIR}"
MODEL_PATH="${CODI_MM_LLAMA1B_PATH}"
MODEL_MAX_LENGTH=512
PRJ_DIM=2048
NUM_LATENT=6
EVAL_DATASETS="${CODI_POST_TRAIN_DATASETS:-gsm8k math500 aime svamp gsm-hard asdiv}"
EVAL_BATCH_SIZE="${CODI_EVAL_BATCH_SIZE:-16}"
RESULTS_DIR="${CODI_MULTIMODEL_RESULT_DIR}/manual_checkpoint_eval/llama1b"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model path is not ready: ${MODEL_PATH}"
  echo "Run: bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama1b --force-datasets"
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

pushd "${CODI_DIR}" >/dev/null
python "${CODI_DIR}/test_multi_dataset.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --ckpt_dir "${CKPT_DIR}" \
  --datasets "${EVAL_DATASETS}" \
  --num_runs 1 \
  --result_dir "${RESULTS_DIR}" \
  --seed 11 \
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
