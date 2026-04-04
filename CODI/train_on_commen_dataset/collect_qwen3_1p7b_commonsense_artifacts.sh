#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../train_on_gsm8k_dataset/env.sh"

VARIANT="simcon"
if [[ "${1:-}" == "--sircl" ]]; then
  VARIANT="simcon_sircl"
  shift
fi

MODEL_NAME="Qwen3-1.7B"
NUM_EPOCHS="${CODI_NUM_EPOCHS:-10}"
LEARNING_RATE="${CODI_LEARNING_RATE:-0.0005}"
EXPT_SUFFIX="${CODI_EXPT_SUFFIX:-}"
if [[ -n "${EXPT_SUFFIX}" && "${EXPT_SUFFIX}" != _* ]]; then
  EXPT_SUFFIX="_${EXPT_SUFFIX}"
fi
EXPT_NAME="${CODI_MULTIMODEL_TAG}_commonsense_qwen3_1p7b_${VARIANT}${EXPT_SUFFIX}"
CHECKPOINT_ROOT="${CODI_MULTIMODEL_SAVE_DIR}/${EXPT_NAME}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_11"
SWEEP_RESULT_DIR="${CODI_MULTIMODEL_RESULT_DIR}/checkpoint_sweeps/${EXPT_NAME}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_11"
LOG_ROOT="${CODI_MULTIMODEL_LOG_DIR}/${EXPT_NAME}"

latest_match() {
  local pattern="$1"
  ls -1dt ${pattern} 2>/dev/null | head -n 1 || true
}

echo "Experiment name : ${EXPT_NAME}"
echo "Checkpoint root : ${CHECKPOINT_ROOT}"
echo "Sweep result dir: ${SWEEP_RESULT_DIR}"
echo "Log root        : ${LOG_ROOT}"
echo

echo "Checkpoint dirs:"
find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V || true
echo

echo "Summary files:"
find "${SWEEP_RESULT_DIR}" -type f \( -name 'all_results.csv' -o -name 'comparison_matrix.csv' -o -name '*.json' \) 2>/dev/null | sort || true
echo

echo "Latest slurm stdout: $(latest_match "${SCRIPT_DIR}/../logs/slurm_*_qwen3_csqa*.out")"
echo "Latest slurm stderr: $(latest_match "${SCRIPT_DIR}/../logs/slurm_*_qwen3_csqa*.err")"
