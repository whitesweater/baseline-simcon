#!/bin/bash
# OOM Watchdog: monitors Slurm jobs and auto-retries with lower batch size on OOM.
# Usage: bash oom_watchdog.sh
# Runs in the background, polls every 60s.
# Logs to logs/oom_watchdog.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${CODI_DIR}/logs/oom_watchdog.log"
POLL_INTERVAL=60
BATCH_STEP=4
MIN_BATCH=16
MAX_RETRIES=2

mkdir -p "${CODI_DIR}/logs"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# --- Job registry ---
# Each entry: "JOB_ID:SLURM_SCRIPT:BATCH_SIZE:RETRY_COUNT"
declare -a JOBS=()

register_job() {
  local job_id="$1" script="$2" batch="$3" retries="${4:-0}"
  JOBS+=("${job_id}:${script}:${batch}:${retries}")
  log "REGISTER job=${job_id} script=$(basename "${script}") batch=${batch} retries=${retries}"
}

is_job_running() {
  local job_id="$1"
  squeue -j "${job_id}" -h -o "%T" 2>/dev/null | grep -qE 'RUNNING|PENDING'
}

check_oom() {
  local job_id="$1"
  # Check sacct for OOM state
  local state
  state=$(sacct -j "${job_id}" -n -o State -X 2>/dev/null | tr -d ' ' | head -1)
  if [[ "${state}" == "OUT_OF_MEMORY" ]]; then
    return 0
  fi
  # Check slurm stderr log for CUDA OOM
  local err_files
  err_files=$(ls "${CODI_DIR}"/logs/slurm_${job_id}_*.err 2>/dev/null)
  for f in ${err_files}; do
    if grep -qi "CUDA out of memory\|OutOfMemoryError\|torch.OutOfMemoryError" "${f}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

ensure_job_gone() {
  # scancel + wait until job fully disappears from squeue (max 120s)
  local job_id="$1"
  local max_wait=120
  local waited=0

  # First check if already gone
  if ! squeue -j "${job_id}" -h -o "%T" 2>/dev/null | grep -qE 'RUNNING|PENDING|COMPLETING'; then
    log "Job ${job_id} already gone."
    return 0
  fi

  log "CANCEL job ${job_id}, waiting for it to fully terminate..."
  scancel "${job_id}" 2>/dev/null

  while (( waited < max_wait )); do
    sleep 5
    waited=$((waited + 5))
    local state
    state=$(squeue -j "${job_id}" -h -o "%T" 2>/dev/null | tr -d ' ' | head -1)
    if [[ -z "${state}" ]]; then
      log "Job ${job_id} terminated after ${waited}s."
      return 0
    fi
    log "Job ${job_id} still in state=${state}, waited ${waited}s/${max_wait}s..."
  done

  log "WARNING: job ${job_id} did not terminate within ${max_wait}s!"
  return 1
}

resubmit() {
  local old_job_id="$1" script="$2" old_batch="$3" retries="$4"
  local new_batch=$((old_batch - BATCH_STEP))
  local new_retries=$((retries + 1))

  if (( new_batch < MIN_BATCH )); then
    log "STOP batch would be ${new_batch} (below min ${MIN_BATCH}), giving up on $(basename "${script}")"
    return 1
  fi
  if (( new_retries > MAX_RETRIES )); then
    log "STOP max retries (${MAX_RETRIES}) reached for $(basename "${script}")"
    return 1
  fi

  # Ensure the old job is fully gone before submitting
  if ! ensure_job_gone "${old_job_id}"; then
    log "ERROR: old job ${old_job_id} still alive, aborting resubmit"
    return 1
  fi

  log "RESUBMIT $(basename "${script}") batch ${old_batch} -> ${new_batch} (retry ${new_retries}/${MAX_RETRIES})"
  local output
  output=$(cd "${CODI_DIR}" && sbatch "${script}" --per-device-batch "${new_batch}" 2>&1)
  local new_job_id
  new_job_id=$(echo "${output}" | grep -oP 'Submitted batch job \K[0-9]+')
  if [[ -z "${new_job_id}" ]]; then
    log "ERROR sbatch failed: ${output}"
    return 1
  fi
  register_job "${new_job_id}" "${script}" "${new_batch}" "${new_retries}"
  return 0
}

# ─── Auto-detect running jobs ───
detect_running_jobs() {
  log "Detecting running jobs..."
  local job_id job_name
  while IFS='|' read -r job_id job_name; do
    job_id=$(echo "${job_id}" | tr -d ' ')
    job_name=$(echo "${job_name}" | tr -d ' ')
    case "${job_name}" in
      qwen3_1p7b)
        register_job "${job_id}" "${SCRIPT_DIR}/slurm_train_qwen3_1p7b.sh" 24 0
        ;;
      qwen3_1p7b_sirc*)
        register_job "${job_id}" "${SCRIPT_DIR}/slurm_train_qwen3_1p7b_sircl.sh" 24 0
        ;;
      llama1b_geo*)
        register_job "${job_id}" "${SCRIPT_DIR}/slurm_train_llama1b_geodesic_only.sh" 20 0
        ;;
      *)
        log "SKIP unknown job ${job_id} (${job_name})"
        ;;
    esac
  done < <(squeue -u "$(whoami)" -h -o "%i|%j" 2>/dev/null)
}

# ─── Main loop ───
main() {
  log "========== OOM Watchdog started (pid=$$) =========="
  log "Config: poll=${POLL_INTERVAL}s step=${BATCH_STEP} min_batch=${MIN_BATCH} max_retries=${MAX_RETRIES}"
  detect_running_jobs

  if [[ ${#JOBS[@]} -eq 0 ]]; then
    log "No matching jobs found. Exiting."
    exit 0
  fi

  while true; do
    sleep "${POLL_INTERVAL}"

    local -a ACTIVE_JOBS=()
    for entry in "${JOBS[@]}"; do
      IFS=':' read -r job_id script batch retries <<< "${entry}"

      if is_job_running "${job_id}"; then
        ACTIVE_JOBS+=("${entry}")
        continue
      fi

      # Job finished — check why
      log "Job ${job_id} ($(basename "${script}")) no longer running, checking..."

      if check_oom "${job_id}"; then
        log "OOM detected for job ${job_id} (batch=${batch})"
        if resubmit "${job_id}" "${script}" "${batch}" "${retries}"; then
          # New job added to JOBS by resubmit(); pick it up next loop
          :
        fi
      else
        local state
        state=$(sacct -j "${job_id}" -n -o State -X 2>/dev/null | tr -d ' ' | head -1)
        log "Job ${job_id} ended with state=${state:-unknown} (not OOM), no action."
      fi
    done

    # Merge: keep active + newly registered
    local -a MERGED=()
    for entry in "${ACTIVE_JOBS[@]}"; do
      MERGED+=("${entry}")
    done
    # Add any new jobs registered by resubmit
    for entry in "${JOBS[@]}"; do
      local found=0
      for existing in "${ACTIVE_JOBS[@]}"; do
        if [[ "${entry}" == "${existing}" ]]; then
          found=1
          break
        fi
      done
      if [[ "${found}" -eq 0 ]]; then
        # Check if this is a newly registered job (not in active but still queued)
        IFS=':' read -r jid _ _ _ <<< "${entry}"
        if is_job_running "${jid}"; then
          MERGED+=("${entry}")
        fi
      fi
    done
    JOBS=("${MERGED[@]}")

    if [[ ${#JOBS[@]} -eq 0 ]]; then
      log "All jobs completed. Watchdog exiting."
      break
    fi

    log "Tracking ${#JOBS[@]} job(s): $(printf '%s ' "${JOBS[@]}")"
  done

  log "========== OOM Watchdog stopped =========="
}

main "$@"
