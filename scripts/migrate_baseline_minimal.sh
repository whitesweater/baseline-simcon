#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXCLUDE_FILE="${SCRIPT_DIR}/minimal_migration_rsync_excludes.txt"
VERIFY_SCRIPT_REL="scripts/verify_baseline_minimal.sh"
STAGE_ROOT_REL="CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1"

SRC="${REPO_ROOT}"
DST_HOST=""
DST_REAL=""
DST_LINK="/data/yhao/baseline"
PYTHON_BIN="python3.11"
SSH_CONFIG=""
DRY_RUN=0
SKIP_REMOTE_PREPARE=0
BOOTSTRAP_VENV=1
INSTALL_REQUIREMENTS=0
RUN_VERIFY=1
SYNC_WORKSPACE=1
SYNC_MODELS=1

usage() {
  cat <<'EOF'
Usage:
  scripts/migrate_baseline_minimal.sh --dst-host user@new-host --dst-real /real/path/baseline [options]

Options:
  --src PATH                  Source repo root. Default: auto-detected repo root
  --dst-host HOST             Required. SSH target such as user@new-host
  --dst-real PATH             Required. Real destination directory on target host
  --dst-link PATH             Compatibility path on target. Default: /data/yhao/baseline
  --python-bin CMD            Remote Python for .venv. Default: python3.11
  --ssh-config FILE           Optional SSH config file, e.g. /root/.codex/skills/hpc-login-ssh/references/ssh_config_hpc
  --install-requirements      After pip install -e ., also run pip install -r requirements.txt
  --no-bootstrap-venv         Skip remote .venv bootstrap
  --no-verify                 Skip remote verification
  --skip-remote-prepare       Do not create remote dir/link before rsync
  --workspace-only            Sync workspace only; skip model sync and verification
  --models-only               Sync models only; skip workspace sync and verification
  --dry-run                   Pass -n to rsync and skip remote bootstrap/verification
  -h, --help                  Show this help

Examples:
  scripts/migrate_baseline_minimal.sh \
    --dst-host user@new-host \
    --dst-real /mnt/exp/baseline

  scripts/migrate_baseline_minimal.sh \
    --dst-host gpu-a \
    --dst-real /mnt/exp/baseline \
    --no-bootstrap-venv \
    --no-verify
EOF
}

log() {
  printf '[migrate] %s\n' "$*"
}

fail() {
  printf '[migrate][error] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

is_model_ready() {
  local model_dir="$1"
  [[ -d "$model_dir" ]] || return 1
  [[ -f "$model_dir/config.json" ]] || return 1
  [[ -f "$model_dir/model.safetensors" || -f "$model_dir/model.safetensors.index.json" || -f "$model_dir/pytorch_model.bin" ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      [[ $# -ge 2 ]] || fail "--src requires a value"
      SRC="$2"
      shift 2
      ;;
    --dst-host)
      [[ $# -ge 2 ]] || fail "--dst-host requires a value"
      DST_HOST="$2"
      shift 2
      ;;
    --dst-real)
      [[ $# -ge 2 ]] || fail "--dst-real requires a value"
      DST_REAL="$2"
      shift 2
      ;;
    --dst-link)
      [[ $# -ge 2 ]] || fail "--dst-link requires a value"
      DST_LINK="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || fail "--python-bin requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --ssh-config)
      [[ $# -ge 2 ]] || fail "--ssh-config requires a value"
      SSH_CONFIG="$2"
      shift 2
      ;;
    --install-requirements)
      INSTALL_REQUIREMENTS=1
      shift
      ;;
    --no-bootstrap-venv)
      BOOTSTRAP_VENV=0
      shift
      ;;
    --no-verify)
      RUN_VERIFY=0
      shift
      ;;
    --skip-remote-prepare)
      SKIP_REMOTE_PREPARE=1
      shift
      ;;
    --workspace-only)
      SYNC_WORKSPACE=1
      SYNC_MODELS=0
      BOOTSTRAP_VENV=0
      RUN_VERIFY=0
      shift
      ;;
    --models-only)
      SYNC_WORKSPACE=0
      SYNC_MODELS=1
      BOOTSTRAP_VENV=0
      RUN_VERIFY=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      BOOTSTRAP_VENV=0
      RUN_VERIFY=0
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

[[ -n "$DST_HOST" ]] || fail "--dst-host is required"
[[ -n "$DST_REAL" ]] || fail "--dst-real is required"
[[ -d "$SRC" ]] || fail "Source repo root does not exist: $SRC"
[[ -f "$EXCLUDE_FILE" ]] || fail "Missing exclude file: $EXCLUDE_FILE"
[[ -f "$SRC/$VERIFY_SCRIPT_REL" ]] || fail "Missing verify script: $SRC/$VERIFY_SCRIPT_REL"
if [[ -n "$SSH_CONFIG" ]]; then
  [[ -f "$SSH_CONFIG" ]] || fail "SSH config file does not exist: $SSH_CONFIG"
fi

require_cmd ssh
require_cmd rsync
require_cmd readlink

if [[ -f "$SRC/CODI/config.env" ]]; then
  # shellcheck disable=SC1090
  source "$SRC/CODI/config.env"
fi

resolve_model_source() {
  local dest_name="$1"
  local env_var="$2"
  local fallback="$3"
  local stage_model_path="$SRC/$STAGE_ROOT_REL/models/$dest_name"
  local candidate=""
  local resolved=""

  if [[ -e "$stage_model_path" || -L "$stage_model_path" ]]; then
    resolved="$(readlink -f "$stage_model_path")"
    if is_model_ready "$resolved"; then
      printf '%s\n' "$resolved"
      return 0
    fi
  fi

  if [[ -n "${!env_var:-}" ]]; then
    candidate="${!env_var}"
    if is_model_ready "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  if is_model_ready "$fallback"; then
    printf '%s\n' "$fallback"
    return 0
  fi

  return 1
}

MODEL_DESTS=(
  "Llama-3.2-1B-Instruct|CODI_LLAMA1B_PATH|/data/yhao/sim-con/modelscope/LLM-Research/Llama-3___2-1B-Instruct"
  "Llama-3.2-3B-Instruct|CODI_LLAMA3B_PATH|/data/yhao/sim-con/modelscope/LLM-Research/Llama-3___2-3B-Instruct"
  "Meta-Llama-3.1-8B-Instruct|CODI_LLAMA8B_PATH|/data/yhao/sim-con/modelscope/LLM-Research/Llama-3___2-8B-Instruct"
  "Qwen3-4B|CODI_QWEN_PATH|/data/yhao/rank/models/Qwen3-4B"
)

declare -a MODEL_SOURCES=()
if (( SYNC_MODELS )); then
  for spec in "${MODEL_DESTS[@]}"; do
    IFS='|' read -r dest_name env_var fallback <<<"$spec"
    source_dir="$(resolve_model_source "$dest_name" "$env_var" "$fallback")" || fail "Could not resolve a ready source directory for model: $dest_name"
    MODEL_SOURCES+=("${dest_name}|${source_dir}")
  done
fi

run_rsync() {
  local -a cmd=(rsync -aH --info=progress2 --partial --append-verify)
  if [[ -n "$SSH_CONFIG" ]]; then
    cmd+=(-e "ssh -F $SSH_CONFIG")
  fi
  if (( DRY_RUN )); then
    cmd+=(-n)
  fi
  cmd+=("$@")
  "${cmd[@]}"
}

ssh_cmd() {
  if [[ -n "$SSH_CONFIG" ]]; then
    ssh -F "$SSH_CONFIG" "$@"
  else
    ssh "$@"
  fi
}

remote_prepare() {
  (( SKIP_REMOTE_PREPARE )) && return 0
  log "Preparing remote directory and compatibility link"
  ssh_cmd "$DST_HOST" bash -s -- "$DST_REAL" "$DST_LINK" <<'EOF'
set -euo pipefail
dst_real="$1"
dst_link="$2"

mkdir -p "$dst_real"

if [[ "$dst_link" == "$dst_real" ]]; then
  exit 0
fi

link_parent="$(dirname "$dst_link")"
mkdir -p "$link_parent"

if [[ -e "$dst_link" && ! -L "$dst_link" ]]; then
  echo "Refusing to replace existing non-symlink path: $dst_link" >&2
  exit 1
fi

ln -sfn "$dst_real" "$dst_link"
EOF
}

sync_workspace() {
  log "Syncing workspace to $DST_HOST:$DST_REAL"
  run_rsync --exclude-from="$EXCLUDE_FILE" "$SRC/" "$DST_HOST:$DST_REAL/"
}

sync_models() {
  local dest_name=""
  local source_dir=""
  local remote_dir=""

  for model_spec in "${MODEL_SOURCES[@]}"; do
    IFS='|' read -r dest_name source_dir <<<"$model_spec"
    remote_dir="$DST_REAL/$STAGE_ROOT_REL/models/$dest_name/"
    log "Syncing model $dest_name from $source_dir"
    run_rsync "$source_dir/" "$DST_HOST:$remote_dir"
  done
}

run_remote_verify() {
  (( RUN_VERIFY )) || return 0
  log "Running remote bootstrap and verification"
  ssh_cmd "$DST_HOST" bash -s -- "$DST_LINK" "$DST_REAL" "$PYTHON_BIN" "$BOOTSTRAP_VENV" "$INSTALL_REQUIREMENTS" <<'EOF'
set -euo pipefail
repo_root="$1"
expected_real="$2"
python_bin="$3"
bootstrap_venv="$4"
install_requirements="$5"

cd "$repo_root"

args=(scripts/verify_baseline_minimal.sh --repo-root "$repo_root" --expected-real "$expected_real" --python-bin "$python_bin")
if [[ "$bootstrap_venv" == "1" ]]; then
  args+=(--bootstrap-venv)
fi
if [[ "$install_requirements" == "1" ]]; then
  args+=(--install-requirements)
fi

bash "${args[0]}" "${args[@]:1}"
EOF
}

log "Source repo: $SRC"
log "Destination: $DST_HOST:$DST_REAL"
log "Compatibility link: $DST_LINK"
if [[ -n "$SSH_CONFIG" ]]; then
  log "SSH config: $SSH_CONFIG"
fi
log "Workspace sync: $SYNC_WORKSPACE | Model sync: $SYNC_MODELS | Bootstrap venv: $BOOTSTRAP_VENV | Verify: $RUN_VERIFY"

remote_prepare

if (( SYNC_WORKSPACE )); then
  sync_workspace
fi

if (( SYNC_MODELS )); then
  sync_models
fi

if (( ! DRY_RUN )); then
  run_remote_verify
else
  log "Dry-run mode enabled; skipped remote bootstrap and verification"
fi

log "Minimal migration workflow completed"
