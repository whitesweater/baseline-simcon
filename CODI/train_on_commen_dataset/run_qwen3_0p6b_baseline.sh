#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

if [[ ! -f "${CODI_VENV_PATH}" ]]; then
  echo "Error: CODI_VENV_PATH is invalid: ${CODI_VENV_PATH}"
  exit 1
fi
# shellcheck disable=SC1091
source "${CODI_VENV_PATH}"

MODEL_PATH="${CODI_MM_QWEN3_0P6B_PATH}"
OUTPUT_DIR="${CODI_COMMONSENSE_RESULT_DIR}/baseline/qwen3_0p6b_raw_commonsense"
BATCH_SIZE="${CODI_BASELINE_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${CODI_BASELINE_MAX_NEW_TOKENS:-256}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Error: missing Qwen3-0.6B model directory: ${MODEL_PATH}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=================================================================="
echo "Baseline model path : ${MODEL_PATH}"
echo "Output dir          : ${OUTPUT_DIR}"
echo "Batch size          : ${BATCH_SIZE}"
echo "Max new tokens      : ${MAX_NEW_TOKENS}"
echo "=================================================================="

pushd "${CODI_DIR}" >/dev/null
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
python "${CODI_DIR}/test_baseline.py" \
  --model_path "${MODEL_PATH}" \
  --datasets "commonsense" \
  --batch_size "${BATCH_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --output_dir "${OUTPUT_DIR}" \
  --greedy
popd >/dev/null
