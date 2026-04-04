#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
# shellcheck disable=SC1091
source "${CODI_VENV_PATH}" || { echo "Error: CODI_VENV_PATH is invalid: ${CODI_VENV_PATH}"; exit 1; }

MODELS=(qwen3 qwen3_0p6b qwen3_1p7b)
DATASETS=(commonsense)
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
      echo "Usage: $0 [--models qwen3 qwen3_0p6b qwen3_1p7b] [--skip-datasets] [--force-datasets]"
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
          --dest-root "${CODI_COMMONSENSE_MODEL_ROOT}" \
          --manifest-root "${CODI_COMMONSENSE_MANIFEST_DIR}" \
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
        --manifest-root "${CODI_COMMONSENSE_MANIFEST_DIR}" \
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
echo "Stage root : ${CODI_COMMONSENSE_ROOT}"
echo "Model root : ${CODI_COMMONSENSE_MODEL_ROOT}"
echo "Cache root : ${CODI_COMMONSENSE_CACHE_DIR}"
echo "Result root: ${CODI_COMMONSENSE_RESULT_DIR}"
echo "Models     : ${MODELS[*]}"
echo "Datasets   : ${DATASETS[*]}"
echo "=================================================================="

download_models

if [[ "${INCLUDE_DATASETS}" == "true" || "${FORCE_DATASETS}" == "true" ]]; then
  warm_datasets
fi

cat <<EOF

[done] CommonsenseQA Qwen stage assets are ready.

Next commands:
  bash CODI/train_on_commen_dataset/run_qwen3_0p6b_baseline.sh
  bash CODI/train_on_commen_dataset/train_qwen3_0p6b.sh
  bash CODI/train_on_commen_dataset/train_qwen3_0p6b.sh --sircl
  bash CODI/train_on_commen_dataset/train_qwen3_1p7b.sh
  bash CODI/train_on_commen_dataset/train_qwen3_1p7b.sh --sircl

These launchers now evaluate each completed checkpoint as soon as it is saved.
To disable live checkpoint evaluation and fall back to the end-of-run catch-up sweep:
  CODI_EVAL_EACH_CHECKPOINT=0 bash CODI/train_on_commen_dataset/train_qwen3_0p6b.sh

Manual single-checkpoint eval fallback:
  python CODI/test_multi_dataset.py --datasets commonsense --num_runs 1 \
    --model_name_or_path <base_model> --ckpt_dir <checkpoint_dir> --result_dir <result_dir>

EOF
