#!/bin/bash
# Watch Qwen3-1.7B CoT-SFT checkpoints and evaluate each new checkpoint.
#
# Usage:
#   bash scripts/watch_eval_cot_qwen3_1p7b.sh [GPU_ID] [POLL_SECONDS]

set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID=${1:-3}
POLL_SECONDS=${2:-300}
CKPT_DIR="./ckpts/gsm-qwen3-1p7b-cot-sft"
STATE_FILE="${CKPT_DIR}/.evaluated_checkpoints_gsm8k.txt"
LOG_DIR="./logs"

mkdir -p "$LOG_DIR" "$CKPT_DIR"
touch "$STATE_FILE"

echo "======================================"
echo "Watching Qwen3-1.7B CoT-SFT checkpoints"
echo "======================================"
echo "Working directory: $(pwd)"
echo "Checkpoint dir   : $CKPT_DIR"
echo "State file       : $STATE_FILE"
echo "GPU for eval     : $GPU_ID"
echo "Poll seconds     : $POLL_SECONDS"
echo ""

evaluate_checkpoint() {
    local ckpt_path="$1"
    local ckpt_name
    ckpt_name=$(basename "$ckpt_path")
    local eval_log="${LOG_DIR}/eval_qwen3_1p7b_${ckpt_name}_$(date +%Y%m%d_%H%M%S).log"

    echo "[$(date '+%F %T')] evaluating ${ckpt_path}"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    MASTER_PORT=29523 \
    bash scripts/eval_cot_qwen3_1p7b.sh 1 "$ckpt_path" \
        2>&1 | tee "$eval_log"

    echo "$ckpt_path" >> "$STATE_FILE"
    echo "[$(date '+%F %T')] finished ${ckpt_path}"
}

checkpoint_ready() {
    local ckpt_path="$1"
    local size_before
    local size_after

    if [[ ! -f "$ckpt_path" ]]; then
        return 1
    fi

    size_before=$(stat -c %s "$ckpt_path")
    sleep 10
    if [[ ! -f "$ckpt_path" ]]; then
        return 1
    fi
    size_after=$(stat -c %s "$ckpt_path")

    [[ "$size_before" -gt 0 && "$size_before" -eq "$size_after" ]]
}

while true; do
    while IFS= read -r ckpt_path; do
        [[ -z "$ckpt_path" ]] && continue

        if grep -Fxq "$ckpt_path" "$STATE_FILE"; then
            continue
        fi

        if [[ ! -f "${ckpt_path}" && ! -d "${ckpt_path}" ]]; then
            continue
        fi

        if ! checkpoint_ready "$ckpt_path"; then
            echo "[$(date '+%F %T')] checkpoint still being written, will retry later: ${ckpt_path}"
            continue
        fi

        evaluate_checkpoint "$ckpt_path"
    done < <(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type f -name 'checkpoint_*' | sort -V)

    sleep "$POLL_SECONDS"
done
