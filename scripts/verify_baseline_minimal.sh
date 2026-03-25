#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/data/yhao/baseline"
EXPECTED_REAL=""
PYTHON_BIN="python3.11"
BOOTSTRAP_VENV=0
INSTALL_REQUIREMENTS=0
RUN_PREPARE_ASSETS=1
RUN_SMOKE_CHECK=1
STAGE_ROOT_REL="CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1"

usage() {
  cat <<'EOF'
Usage:
  scripts/verify_baseline_minimal.sh [options]

Options:
  --repo-root PATH            Migrated repo root. Default: /data/yhao/baseline
  --expected-real PATH        Assert that readlink -f <repo-root> equals this path
  --python-bin CMD            Python used to create .venv. Default: python3.11
  --bootstrap-venv            Create/reuse .venv and install dependencies
  --install-requirements      After pip install -e ., also run pip install -r requirements.txt
  --skip-prepare-assets       Skip prepare_assets.sh validation
  --skip-smoke-check          Skip python train.py --help smoke check
  -h, --help                  Show this help
EOF
}

log() {
  printf '[verify] %s\n' "$*"
}

fail() {
  printf '[verify][error] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      [[ $# -ge 2 ]] || fail "--repo-root requires a value"
      REPO_ROOT="$2"
      shift 2
      ;;
    --expected-real)
      [[ $# -ge 2 ]] || fail "--expected-real requires a value"
      EXPECTED_REAL="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || fail "--python-bin requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --bootstrap-venv)
      BOOTSTRAP_VENV=1
      shift
      ;;
    --install-requirements)
      INSTALL_REQUIREMENTS=1
      shift
      ;;
    --skip-prepare-assets)
      RUN_PREPARE_ASSETS=0
      shift
      ;;
    --skip-smoke-check)
      RUN_SMOKE_CHECK=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

require_cmd readlink
require_cmd bash

[[ -d "$REPO_ROOT" ]] || fail "Repo root does not exist: $REPO_ROOT"

if [[ -n "$EXPECTED_REAL" ]]; then
  resolved_repo="$(readlink -f "$REPO_ROOT")"
  resolved_expected="$(readlink -f "$EXPECTED_REAL" 2>/dev/null || printf '%s' "$EXPECTED_REAL")"
  [[ "$resolved_repo" == "$resolved_expected" ]] || fail "Repo root resolves to $resolved_repo, expected $resolved_expected"
  log "Path compatibility check passed: $REPO_ROOT -> $resolved_repo"
fi

MODEL_CONFIG_PATH="$REPO_ROOT/$STAGE_ROOT_REL/models/Llama-3.2-1B-Instruct/config.json"
ICOT_CACHE_PATH="$REPO_ROOT/$STAGE_ROOT_REL/cache/dataset_cache/dataset_icot_0a5b3650760a22ea.pt"
MULTIARITH_PATH="$REPO_ROOT/CODI/local_datasets/multiarith/train_42.json"
CONFIG_ENV_PATH="$REPO_ROOT/CODI/config.env"

[[ -f "$CONFIG_ENV_PATH" ]] || fail "Missing required config file: $CONFIG_ENV_PATH"
[[ -f "$MULTIARITH_PATH" ]] || fail "Missing required dataset file: $MULTIARITH_PATH"
[[ -f "$ICOT_CACHE_PATH" ]] || fail "Missing required icot cache: $ICOT_CACHE_PATH"
[[ -f "$MODEL_CONFIG_PATH" ]] || fail "Missing synced model config: $MODEL_CONFIG_PATH"
log "Required files are present"

if (( BOOTSTRAP_VENV )); then
  require_cmd "$PYTHON_BIN"
  log "Bootstrapping virtual environment with $PYTHON_BIN"
  cd "$REPO_ROOT"
  if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
  fi
  [[ -f .venv/bin/activate ]] || fail "Virtual environment is incomplete: $REPO_ROOT/.venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install -U pip setuptools wheel
  python -m pip install -e .
  if (( INSTALL_REQUIREMENTS )); then
    python -m pip install -r requirements.txt
  fi
fi

[[ -f "$REPO_ROOT/.venv/bin/activate" ]] || fail "Virtual environment not found. Re-run with --bootstrap-venv or create $REPO_ROOT/.venv first"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

log "Checking core imports"
python -c 'import torch, transformers, datasets, modelscope; print("ok")'

if (( RUN_PREPARE_ASSETS )); then
  log "Running prepare_assets.sh validation"
  bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama1b --force-datasets
fi

if (( RUN_SMOKE_CHECK )); then
  log "Running train.py --help smoke check"
  (
    cd "$REPO_ROOT/CODI"
    python train.py --help >/dev/null
  )
fi

log "Minimal migration verification passed"
