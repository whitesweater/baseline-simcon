#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
# shellcheck disable=SC1091
source "${CODI_VENV_PATH}" || { echo "Error: CODI_VENV_PATH is invalid: ${CODI_VENV_PATH}"; exit 1; }

MODELS=(llama1b llama3b llama8b qwen3)
DATASETS=(gsm8k math500 aime)
INCLUDE_DATASETS=true
FORCE_DATASETS=false
REQUIRED_ICOT_CACHE="dataset_icot_0a5b3650760a22ea.pt"

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
      echo "Usage: $0 [--models llama1b llama3b llama8b qwen3] [--skip-datasets] [--force-datasets]"
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

copy_icot_cache() {
  local src_dir="${CODI_MULTIMODEL_SOURCE_CACHE_DIR}/dataset_cache"
  local dst_dir="${CODI_MULTIMODEL_ICOT_CACHE_DIR}"
  mkdir -p "${dst_dir}"
  if [[ -f "${dst_dir}/${REQUIRED_ICOT_CACHE}" ]]; then
    echo "[skip] required icot cache already exists in stage cache: ${dst_dir}/${REQUIRED_ICOT_CACHE}"
    return
  fi
  if [[ ! -d "${src_dir}" ]]; then
    echo "[error] source icot cache dir not found: ${src_dir}"
    echo "Training on icot cannot start without ${REQUIRED_ICOT_CACHE}"
    exit 1
  fi
  shopt -s nullglob
  local files=("${src_dir}"/dataset_icot_*.pt)
  shopt -u nullglob
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "[error] no dataset_icot_*.pt found under ${src_dir}"
    exit 1
  fi
  cp -a "${src_dir}"/dataset_icot_*.pt "${dst_dir}/"
  if [[ ! -f "${dst_dir}/${REQUIRED_ICOT_CACHE}" ]]; then
    echo "[error] copied icot cache does not include required file: ${REQUIRED_ICOT_CACHE}"
    exit 1
  fi
  echo "[ok] copied icot cache into stage cache: ${dst_dir}"
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

copy_icot_cache
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

After llama1b training:
  bash CODI/train_on_gsm8k_dataset/eval_llama1b_math500_aime.sh <ckpt_dir>

EOF
