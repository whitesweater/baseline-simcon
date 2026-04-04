#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
# shellcheck disable=SC1091
source "${CODI_VENV_PATH}" || { echo "Error: CODI_VENV_PATH is invalid: ${CODI_VENV_PATH}"; exit 1; }

MODELS=(llama1b llama3b llama8b qwen3 qwen3_0p6b qwen3_1p7b)
DATASETS=(gsm8k math500 aime svamp gsm-hard asdiv)
INCLUDE_DATASETS=true
FORCE_DATASETS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      MODELS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --skip-datasets)
      INCLUDE_DATASETS=false
      shift
      ;;
    --force-datasets)
      FORCE_DATASETS=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--models llama1b llama3b llama8b qwen3 qwen3_0p6b qwen3_1p7b] [--skip-datasets] [--force-datasets]"
      exit 1
      ;;
  esac
done

if [[ -f /root/.bashrc ]]; then
  # shellcheck disable=SC1091
  source /root/.bashrc >/dev/null 2>&1 || true
fi

switch_download_env() {
  local backend="$1"
  case "${backend}" in
    modelscope)
      unset HF_ENDPOINT
      if declare -F proxy_off >/dev/null; then
        proxy_off >/dev/null 2>&1 || true
      fi
      ;;
    hf-mirror)
      export HF_ENDPOINT="https://hf-mirror.com"
      if declare -F proxy_off >/dev/null; then
        proxy_off >/dev/null 2>&1 || true
      fi
      ;;
    hf)
      unset HF_ENDPOINT
      if declare -F proxy_on >/dev/null; then
        proxy_on silent >/dev/null 2>&1 || true
      fi
      ;;
    *)
      echo "Unsupported backend: ${backend}"
      exit 1
      ;;
  esac
}

prepare_gsm8k_aug_source() {
  echo "[info] icot training now reads GSM8K-Aug directly from Hugging Face datasets."
  echo "[info] legacy dataset_icot_*.pt cache is no longer required."
  echo "[info] gsm8k cache dir: ${CODI_GSM8K_AUG_CACHE_DIR:-${HF_DATASETS_CACHE:-/data/yhao/hf_datasets_cache}}"
}

download_models() {
  local backend
  local model
  local success
  for model in "${MODELS[@]}"; do
    success=false
    for backend in modelscope hf-mirror hf; do
      switch_download_env "${backend}"
      echo "[try] ${model} via ${backend}"
      if python "${SCRIPT_DIR}/prepare_assets.py" models \
          --backend "${backend}" \
          --dest-root "${CODI_MULTIMODEL_MODEL_ROOT}" \
          --manifest-root "${CODI_MULTIMODEL_MANIFEST_DIR}" \
          --models "${model}"; then
        success=true
        break
      fi
      echo "[retry] ${model} failed via ${backend}"
    done
    if [[ "${success}" != "true" ]]; then
      echo "[error] failed to download model: ${model}"
      exit 1
    fi
  done
}

warm_datasets() {
  local backend
  local success=false
  for backend in hf-mirror hf; do
    switch_download_env "${backend}"
    echo "[try] warming datasets via ${backend}"
    if python "${SCRIPT_DIR}/prepare_assets.py" datasets \
        --manifest-root "${CODI_MULTIMODEL_MANIFEST_DIR}" \
        --datasets "${DATASETS[@]}"; then
      success=true
      break
    fi
    echo "[retry] dataset warm-up failed via ${backend}"
  done
  if [[ "${success}" != "true" ]]; then
    echo "[error] failed to warm dataset caches"
    exit 1
  fi
}

echo "=================================================================="
echo "Stage root : ${CODI_MULTIMODEL_ROOT}"
echo "Model root : ${CODI_MULTIMODEL_MODEL_ROOT}"
echo "Cache root : ${CODI_MULTIMODEL_CACHE_DIR}"
echo "Result root: ${CODI_MULTIMODEL_RESULT_DIR}"
echo "Models     : ${MODELS[*]}"
echo "Datasets   : ${DATASETS[*]}"
echo "=================================================================="

prepare_gsm8k_aug_source
download_models

if [[ "${INCLUDE_DATASETS}" == "true" || "${FORCE_DATASETS}" == "true" ]]; then
  warm_datasets
fi

cat <<EOF

[done] stage assets are ready.

Next commands:
  bash CODI/train_on_gsm8k_dataset/train_llama1b.sh
  bash CODI/train_on_gsm8k_dataset/train_llama3b.sh
  bash CODI/train_on_gsm8k_dataset/train_llama8b.sh
  bash CODI/train_on_gsm8k_dataset/train_qwen3.sh
  bash CODI/train_on_gsm8k_dataset/train_qwen3_0p6b.sh
  bash CODI/train_on_gsm8k_dataset/train_qwen3_codi.sh
  bash CODI/train_on_gsm8k_dataset/train_qwen3_1p7b.sh
  bash CODI/train_on_gsm8k_dataset/train_qwen3_1p7b_codi.sh

These training scripts now auto-evaluate each completed checkpoint as soon as it is saved,
then run a final catch-up sweep for anything still missing on:
  ${DATASETS[*]}

Primary rebuttal train_* entry points run one SIM-CoT variant at a time:
  default: simcon
  pass --sircl or --variant simcon_sircl for simcon_sircl

Optional Qwen3-4B CODI entry:
  default: codi
  pass --sircl or --variant codi_sircl for codi_sircl

Optional Qwen3-1.7B entry points:
  bash CODI/train_on_gsm8k_dataset/train_qwen3_0p6b.sh
  bash CODI/train_on_gsm8k_dataset/train_qwen3_1p7b.sh
  bash CODI/train_on_gsm8k_dataset/train_qwen3_1p7b_codi.sh

Manual single-checkpoint eval fallback:
  bash CODI/train_on_gsm8k_dataset/eval_llama1b_math500_aime.sh <ckpt_dir>

EOF
