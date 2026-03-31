#!/bin/bash
# Evaluate a GSM8K CoT-SFT baseline trained with Qwen3-4B.
#
# Usage:
#   bash scripts/eval_cot_qwen3.sh [NUM_GPUS] [CHECKPOINT_PATH]

set -euo pipefail

NUM_GPUS=${1:-4}
CHECKPOINT_PATH=${2:-}
CONFIG_FILE="args/gsm_cot_qwen3_eval.yaml"
MODEL_DIR="/data/yhao/rank/models/Qwen3-4B"
CKPT_DIR="./ckpts/gsm-qwen3-cot-sft"
MASTER_PORT=${MASTER_PORT:-29522}

echo "======================================"
echo "Qwen3-4B CoT-SFT Evaluation"
echo "======================================"
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG_FILE"
echo "Model path: $MODEL_DIR"
echo ""

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Qwen3 model directory not found: $MODEL_DIR"
    echo "Please update model_id in $CONFIG_FILE if your local path is different."
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH=$(ls -v "${CKPT_DIR}"/checkpoint_* 2>/dev/null | tail -1 || true)
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: no checkpoint found under ${CKPT_DIR}"
    exit 1
fi

TMP_CONFIG=$(mktemp /tmp/gsm_cot_qwen3_eval.XXXXXX.yaml)
trap 'rm -f "$TMP_CONFIG"' EXIT
sed "s|^load_model_path:.*|load_model_path: ${CHECKPOINT_PATH}|" "$CONFIG_FILE" > "$TMP_CONFIG"

echo "Using checkpoint: $CHECKPOINT_PATH"

torchrun --nnodes 1 \
         --master_port "$MASTER_PORT" \
         --nproc_per_node "$NUM_GPUS" \
         run.py "$TMP_CONFIG"

echo ""
echo "Evaluation completed."
