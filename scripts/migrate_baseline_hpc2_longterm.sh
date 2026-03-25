#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXCLUDE_FILE="${SCRIPT_DIR}/hpc2_longterm_overlay_rsync_excludes.txt"
LOCALIZE_SCRIPT_REL="scripts/localize_codi_config_env.py"
VERIFY_SCRIPT_REL="scripts/verify_baseline_minimal.sh"

DST_HOST="hpc2-vpn"
DST_ROOT="/hpc2hdd/home/yhao481/jhupload/proj/baseline"
DST_PARENT="/hpc2hdd/home/yhao481/jhupload/proj"
CACHE_ROOT="/hpc2hdd/home/yhao481/jhupload/cache"
SSH_CONFIG="/root/.codex/skills/hpc-login-ssh/references/ssh_config_hpc_via_remote_vpn"
BRANCH="$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || true)"
BOOTSTRAP_BRANCH="main"
INSTALL_REQUIREMENTS=0
RUN_VERIFY=1
BOOTSTRAP_VENV=1
START_BACKGROUND_MODELS=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/migrate_baseline_hpc2_longterm.sh [options]

Options:
  --dst-host HOST             SSH target. Default: hpc2-vpn
  --dst-root PATH             Fixed HPC2 repo root. Default: /hpc2hdd/home/yhao481/jhupload/proj/baseline
  --cache-root PATH           Fixed HPC2 cache root. Default: /hpc2hdd/home/yhao481/jhupload/cache
  --ssh-config FILE           SSH config for the remote VPN path
  --branch NAME               Preferred Git branch on HPC2. Default: current local branch
  --bootstrap-branch NAME     Remote branch used when preferred branch is not on origin. Default: main
  --install-requirements      After editable install, also run pip install -r requirements.txt
  --no-bootstrap-venv         Skip the remote uv / .venv bootstrap
  --no-verify                 Skip the remote import + prepare_assets + smoke verification
  --no-background-models      Skip the remote background download for llama3b/llama8b/qwen3
  --dry-run                   Dry-run rsync and skip remote mutation after git bootstrap
  -h, --help                  Show this help
EOF
}

log() {
  printf '[hpc2-migrate] %s\n' "$*"
}

fail() {
  printf '[hpc2-migrate][error] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dst-host)
      [[ $# -ge 2 ]] || fail "--dst-host requires a value"
      DST_HOST="$2"
      shift 2
      ;;
    --dst-root)
      [[ $# -ge 2 ]] || fail "--dst-root requires a value"
      DST_ROOT="$2"
      DST_PARENT="$(dirname "$DST_ROOT")"
      shift 2
      ;;
    --cache-root)
      [[ $# -ge 2 ]] || fail "--cache-root requires a value"
      CACHE_ROOT="$2"
      shift 2
      ;;
    --ssh-config)
      [[ $# -ge 2 ]] || fail "--ssh-config requires a value"
      SSH_CONFIG="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || fail "--branch requires a value"
      BRANCH="$2"
      shift 2
      ;;
    --bootstrap-branch)
      [[ $# -ge 2 ]] || fail "--bootstrap-branch requires a value"
      BOOTSTRAP_BRANCH="$2"
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
    --no-background-models)
      START_BACKGROUND_MODELS=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      BOOTSTRAP_VENV=0
      RUN_VERIFY=0
      START_BACKGROUND_MODELS=0
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

[[ -n "$BRANCH" ]] || fail "Could not determine a local branch. Pass --branch explicitly."
[[ -f "$SSH_CONFIG" ]] || fail "Missing SSH config: $SSH_CONFIG"
[[ -f "$EXCLUDE_FILE" ]] || fail "Missing exclude file: $EXCLUDE_FILE"
[[ -f "$REPO_ROOT/$LOCALIZE_SCRIPT_REL" ]] || fail "Missing localize helper: $REPO_ROOT/$LOCALIZE_SCRIPT_REL"
[[ -f "$REPO_ROOT/$VERIFY_SCRIPT_REL" ]] || fail "Missing verify script: $REPO_ROOT/$VERIFY_SCRIPT_REL"

require_cmd ssh
require_cmd rsync
require_cmd git

ORIGIN_URL="$(git -C "${REPO_ROOT}" remote get-url origin 2>/dev/null || true)"
[[ -n "$ORIGIN_URL" ]] || fail "Could not resolve origin URL from the local repo"

ssh_cmd() {
  ssh -F "$SSH_CONFIG" "$@"
}

run_rsync() {
  local -a cmd=(rsync -aH --info=progress2 --partial --append-verify -e "ssh -F $SSH_CONFIG")
  if (( DRY_RUN )); then
    cmd+=(-n)
  fi
  cmd+=("$@")
  "${cmd[@]}"
}

verify_remote_access() {
  log "Verifying remote VPN access to $DST_HOST"
  ssh_cmd "$DST_HOST" "hostname; whoami; pwd"
}

remote_git_bootstrap() {
  log "Bootstrapping remote repo at $DST_ROOT"
  ssh_cmd "$DST_HOST" bash -s -- "$DST_ROOT" "$DST_PARENT" "$ORIGIN_URL" "$BRANCH" "$BOOTSTRAP_BRANCH" <<'EOF'
set -euo pipefail
dst_root="$1"
dst_parent="$2"
origin_url="$3"
branch="$4"
bootstrap_branch="$5"

mkdir -p "$dst_parent"

if [[ -e "$dst_root" && ! -d "$dst_root" ]]; then
  echo "Destination exists and is not a directory: $dst_root" >&2
  exit 1
fi

if [[ ! -d "$dst_root/.git" ]]; then
  if [[ -d "$dst_root" ]] && find "$dst_root" -mindepth 1 -maxdepth 1 | read -r _; then
    echo "Destination exists but is not a git repo: $dst_root" >&2
    exit 1
  fi
  rm -rf "$dst_root"
  git clone "$origin_url" "$dst_root"
fi

cd "$dst_root"
git remote set-url origin "$origin_url"
git fetch origin --prune

target_branch="$bootstrap_branch"
if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
  target_branch="$branch"
fi

if git show-ref --verify --quiet "refs/heads/$target_branch"; then
  git checkout "$target_branch"
else
  git checkout -B "$target_branch" "origin/$target_branch"
fi

if git diff --quiet && git diff --cached --quiet && [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
  git pull --ff-only origin "$target_branch"
else
  echo "[remote-git][warn] working tree is dirty; skipped fast-forward pull"
fi

if [[ "$target_branch" != "$branch" ]]; then
  git checkout -B "$branch"
fi

printf '[remote-git] repo=%s\n' "$(pwd)"
printf '[remote-git] branch=%s\n' "$(git branch --show-current)"
EOF
}

sync_overlay() {
  log "Syncing current workspace overlay to $DST_HOST:$DST_ROOT"
  run_rsync --exclude-from="$EXCLUDE_FILE" "$REPO_ROOT/" "$DST_HOST:$DST_ROOT/"
}

localize_remote_config() {
  log "Localizing CODI/config.env inside the HPC2 copy"
  ssh_cmd "$DST_HOST" "python3 '$DST_ROOT/$LOCALIZE_SCRIPT_REL' --repo-root '$DST_ROOT' --cluster hpc2 --cache-root '$CACHE_ROOT'"
}

bootstrap_remote_venv() {
  (( BOOTSTRAP_VENV )) || return 0
  log "Bootstrapping remote uv-managed Python 3.11 environment"
  ssh_cmd "$DST_HOST" bash -s -- "$DST_ROOT" "$CACHE_ROOT" "$INSTALL_REQUIREMENTS" <<'EOF'
set -euo pipefail
repo_root="$1"
cache_root="$2"
install_requirements="$3"

export PATH="$HOME/.local/bin:$PATH"
uv_bin="$(command -v uv || true)"
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin="$HOME/.local/bin/uv"
fi
if [[ -z "$uv_bin" ]]; then
  echo "uv is not available on HPC2" >&2
  exit 1
fi

export UV_CACHE_DIR="${cache_root}/uv"
mkdir -p "$UV_CACHE_DIR"

cd "$repo_root"
"$uv_bin" python install 3.11
"$uv_bin" venv --python 3.11 .venv
source .venv/bin/activate
"$uv_bin" pip install -U pip setuptools wheel
"$uv_bin" pip install -e .
if [[ "$install_requirements" == "1" && -f requirements.txt ]]; then
  "$uv_bin" pip install -r requirements.txt
fi
python -V
EOF
}

start_background_model_downloads() {
  (( START_BACKGROUND_MODELS )) || return 0
  log "Starting background downloads for llama3b / llama8b / qwen3 on HPC2"
  ssh_cmd "$DST_HOST" bash -s -- "$DST_ROOT" <<'EOF'
set -euo pipefail
repo_root="$1"
stage_root="${repo_root}/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1"
log_dir="${stage_root}/logs"
mkdir -p "$log_dir"
log_file="${log_dir}/prepare_assets_background_$(date -u +%Y%m%d_%H%M%S).log"
pid_file="${log_dir}/prepare_assets_background.pid"

cd "$repo_root"
source .venv/bin/activate
nohup bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama3b llama8b qwen3 --skip-datasets >"$log_file" 2>&1 < /dev/null &
echo "$!" >"$pid_file"
echo "[background] pid=$(cat "$pid_file")"
echo "[background] log=$log_file"
EOF
}

run_remote_verify() {
  (( RUN_VERIFY )) || return 0
  log "Running remote verification with llama1b + datasets smoke check"
  ssh_cmd "$DST_HOST" bash -s -- "$DST_ROOT" <<'EOF'
set -euo pipefail
repo_root="$1"

cd "$repo_root"
source .venv/bin/activate
python -c 'import torch, transformers, datasets, modelscope; print("ok")'
bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama1b --force-datasets
(
  cd CODI
  python train.py --help >/dev/null
)
git status --short
EOF
}

log "Local repo: $REPO_ROOT"
log "Remote repo: $DST_HOST:$DST_ROOT"
log "Remote cache root: $CACHE_ROOT"
log "Remote branch preference: $BRANCH (bootstrap: $BOOTSTRAP_BRANCH)"

verify_remote_access
remote_git_bootstrap

if (( DRY_RUN )); then
  log "Dry-run enabled; skipping rsync overlay and remote mutations after bootstrap"
  exit 0
fi

sync_overlay
localize_remote_config
bootstrap_remote_venv
start_background_model_downloads
run_remote_verify

log "HPC2 long-term migration workflow completed"
