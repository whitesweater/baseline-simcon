#!/bin/bash
# Train a GSM8K CoT-SFT baseline with Qwen3-4B.
#
# Usage:
#   bash scripts/train_cot_qwen3.sh [NUM_GPUS]

set -euo pipefail

NUM_GPUS=${1:-4}
CONFIG_FILE="args/gsm_cot_qwen3.yaml"
MODEL_DIR="/data/yhao/rank/models/Qwen3-4B"

echo "======================================"
echo "Qwen3-4B CoT-SFT Training"
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

if [ ! -f "./data/gsm_train.json" ] || [ ! -f "./data/gsm_test.json" ]; then
    echo "Error: GSM data files are missing under ./data"
    echo "Expected: ./data/gsm_train.json and ./data/gsm_test.json"
    exit 1
fi

torchrun --nnodes 1 \
         --nproc_per_node "$NUM_GPUS" \
         run.py "$CONFIG_FILE"

echo ""
echo "Training completed."
echo "Checkpoints saved to: ./ckpts/gsm-qwen3-cot-sft/"
