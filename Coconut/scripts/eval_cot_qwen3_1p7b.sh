#!/bin/bash
# Evaluate a GSM8K CoT-SFT baseline trained with Qwen3-1.7B.
#
# Usage:
#   bash scripts/eval_cot_qwen3_1p7b.sh [NUM_GPUS] [CHECKPOINT_PATH]

set -euo pipefail

NUM_GPUS=${1:-4}
CHECKPOINT_PATH=${2:-}
CONFIG_FILE="args/gsm_cot_qwen3_1p7b_eval.yaml"
MODEL_DIR="/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Qwen3-1.7B"
CKPT_DIR="./ckpts/gsm-qwen3-1p7b-cot-sft"
MASTER_PORT=${MASTER_PORT:-29523}

echo "======================================"
echo "Qwen3-1.7B CoT-SFT Evaluation"
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
    echo "Error: Qwen3-1.7B model directory not found: $MODEL_DIR"
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

TMP_CONFIG=$(mktemp /tmp/gsm_cot_qwen3_1p7b_eval.XXXXXX.yaml)
trap 'rm -f "$TMP_CONFIG"' EXIT
sed "s|^load_model_path:.*|load_model_path: ${CHECKPOINT_PATH}|" "$CONFIG_FILE" > "$TMP_CONFIG"

echo "Using checkpoint: $CHECKPOINT_PATH"

torchrun --nnodes 1 \
         --master_port "$MASTER_PORT" \
         --nproc_per_node "$NUM_GPUS" \
         run.py "$TMP_CONFIG"

echo ""
echo "Evaluation completed."
