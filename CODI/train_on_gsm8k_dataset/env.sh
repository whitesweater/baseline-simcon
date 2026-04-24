#!/bin/bash

# This file is intended to be sourced by other scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_ROOT="$(cd "${CODI_DIR}/.." && pwd)"

if [[ -f "${CODI_DIR}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${CODI_DIR}/config.env"
fi

export CODI_VENV_PATH="${CODI_VENV_PATH:-${BASELINE_ROOT}/.venv/bin/activate}"
export CODI_RUN_ROOT="${CODI_RUN_ROOT:-${BASELINE_ROOT}/CODI_rebuttal_runs/rebuttal_20260325}"
export CODI_TRAIN_NNODES="${CODI_TRAIN_NNODES:-1}"
export CODI_TRAIN_NPROC_PER_NODE="${CODI_TRAIN_NPROC_PER_NODE:-4}"

# A fully separated workspace for the GSM8K multi-backbone rebuttal stage.
export CODI_MULTIMODEL_TAG="${CODI_MULTIMODEL_TAG:-multimodel_gsm8k_math500_aime_v1}"
export CODI_MULTIMODEL_ROOT="${CODI_RUN_ROOT}/${CODI_MULTIMODEL_TAG}"
export CODI_MULTIMODEL_MODEL_ROOT="${CODI_MULTIMODEL_ROOT}/models"
export CODI_MULTIMODEL_SAVE_DIR="${CODI_MULTIMODEL_ROOT}/outputs"
export CODI_MULTIMODEL_RESULT_DIR="${CODI_MULTIMODEL_ROOT}/results"
export CODI_MULTIMODEL_LOG_DIR="${CODI_MULTIMODEL_ROOT}/logs"
export CODI_MULTIMODEL_MANIFEST_DIR="${CODI_MULTIMODEL_ROOT}/manifests"
export CODI_MULTIMODEL_CACHE_DIR="${CODI_MULTIMODEL_ROOT}/cache"

# Stage-specific cache locations to avoid mixing with previous runs.
export CODI_MULTIMODEL_HF_HOME="${CODI_MULTIMODEL_ROOT}/hf_home"
export CODI_MULTIMODEL_HF_DATASETS_CACHE="${CODI_MULTIMODEL_ROOT}/hf_datasets_cache"
export CODI_MULTIMODEL_MODELSCOPE_CACHE="${CODI_MULTIMODEL_ROOT}/modelscope_cache"
export HF_HOME="${HF_HOME:-${CODI_MULTIMODEL_HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CODI_MULTIMODEL_HF_DATASETS_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CODI_MULTIMODEL_HF_HOME}/transformers}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${CODI_MULTIMODEL_MODELSCOPE_CACHE}}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export CODI_GSM8K_AUG_HF_ID="${CODI_GSM8K_AUG_HF_ID:-zen-E/GSM8k-Aug}"
export CODI_GSM8K_AUG_CACHE_DIR="${CODI_GSM8K_AUG_CACHE_DIR:-/data/yhao/hf_datasets_cache}"

# Legacy dataset-cache paths are kept for compatibility with older scripts.
export CODI_MULTIMODEL_ICOT_CACHE_DIR="${CODI_MULTIMODEL_CACHE_DIR}/dataset_cache"
export CODI_MULTIMODEL_SOURCE_CACHE_DIR="${CODI_CACHE_DIR:-/data/yhao/sim-con/CODI/cache}"

# Downloaded model paths for this stage.
export CODI_MM_LLAMA1B_PATH="${CODI_MULTIMODEL_MODEL_ROOT}/Llama-3.2-1B-Instruct"
export CODI_MM_LLAMA3B_PATH="${CODI_MULTIMODEL_MODEL_ROOT}/Llama-3.2-3B-Instruct"
export CODI_MM_LLAMA8B_PATH="${CODI_MULTIMODEL_MODEL_ROOT}/Meta-Llama-3.1-8B-Instruct"
export CODI_MM_QWEN3_PATH="${CODI_MULTIMODEL_MODEL_ROOT}/Qwen3-4B"
export CODI_MM_QWEN3_8B_PATH="${CODI_MULTIMODEL_MODEL_ROOT}/Qwen3-8B"
export CODI_MM_QWEN3_0P6B_PATH="${CODI_MULTIMODEL_MODEL_ROOT}/Qwen3-0.6B-Base"
export CODI_MM_QWEN3_1P7B_PATH="${CODI_MULTIMODEL_MODEL_ROOT}/Qwen3-1.7B"

mkdir -p \
  "${CODI_MULTIMODEL_ROOT}" \
  "${CODI_MULTIMODEL_MODEL_ROOT}" \
  "${CODI_MULTIMODEL_SAVE_DIR}" \
  "${CODI_MULTIMODEL_RESULT_DIR}" \
  "${CODI_MULTIMODEL_LOG_DIR}" \
  "${CODI_MULTIMODEL_MANIFEST_DIR}" \
  "${CODI_MULTIMODEL_ICOT_CACHE_DIR}" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${CODI_GSM8K_AUG_CACHE_DIR}" \
  "${TRANSFORMERS_CACHE}" \
  "${MODELSCOPE_CACHE}"
