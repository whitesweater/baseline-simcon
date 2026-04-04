#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

VARIANT="simcon"
EXPT_SUFFIX="${CODI_EXPT_SUFFIX:-}"
MODEL_NAME="Qwen3-1.7B"
LEARNING_RATE="${CODI_LEARNING_RATE:-0.0005}"
NUM_EPOCHS="${CODI_NUM_EPOCHS:-10}"
SEED=11

if [[ -n "${EXPT_SUFFIX}" && "${EXPT_SUFFIX}" != _* ]]; then
  EXPT_SUFFIX="_${EXPT_SUFFIX}"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sircl)
      VARIANT="simcon_sircl"
      shift
      ;;
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

case "${VARIANT}" in
  simcon|simcon_sircl)
    ;;
  *)
    echo "Unsupported variant: ${VARIANT}" >&2
    exit 1
    ;;
esac

EXPT_NAME="${CODI_COMMONSENSE_STAGE_TAG}_commonsense_qwen3_1p7b_${VARIANT}${EXPT_SUFFIX}"
CHECKPOINT_ROOT="${CODI_COMMONSENSE_SAVE_DIR}/${EXPT_NAME}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_${SEED}"
SWEEP_RESULT_DIR="${CODI_COMMONSENSE_RESULT_DIR}/checkpoint_sweeps/${EXPT_NAME}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_${SEED}"
SUMMARY_DIR="${SWEEP_RESULT_DIR}/summary"
TB_LOG_DIR="${CODI_COMMONSENSE_LOG_DIR}/${EXPT_NAME}"

LATEST_CKPT=""
if [[ -d "${CHECKPOINT_ROOT}" ]]; then
  LATEST_CKPT="$(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)"
fi

echo "Stage root       : ${CODI_COMMONSENSE_ROOT}"
echo "Experiment name  : ${EXPT_NAME}"
echo "Checkpoint root  : ${CHECKPOINT_ROOT}"
echo "Latest checkpoint: ${LATEST_CKPT:-<none yet>}"
echo "TensorBoard logs : ${TB_LOG_DIR}"
echo "Sweep result dir : ${SWEEP_RESULT_DIR}"
echo "Summary CSV      : ${SUMMARY_DIR}/all_results.csv"
echo "Matrix CSV       : ${SUMMARY_DIR}/comparison_matrix.csv"
echo "Slurm debug logs : /hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI/logs/slurm_*_qwen_csqa_1p7b_debug.{out,err}"
echo "Slurm full logs  : /hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI/logs/slurm_*_qwen_csqa_1p7b.{out,err}"

if [[ -f "${SUMMARY_DIR}/comparison_matrix.csv" ]]; then
  echo "--- comparison_matrix.csv (head) ---"
  head -20 "${SUMMARY_DIR}/comparison_matrix.csv"
fi

if [[ -f "${SUMMARY_DIR}/all_results.csv" ]]; then
  echo "--- all_results.csv (head) ---"
  head -20 "${SUMMARY_DIR}/all_results.csv"
fi
