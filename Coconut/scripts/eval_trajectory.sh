#!/bin/bash
# Evaluation script for Coconut with Trajectory Consistency Loss
#
# Usage: bash scripts/eval_trajectory.sh [NUM_GPUS]

set -e

# Configuration
NUM_GPUS=${1:-8}  # Default to 8 GPUs
CONFIG_FILE="args/gsm_coconut_trajectory_eval.yaml"

echo "======================================"
echo "Coconut + Trajectory Consistency Evaluation"
echo "======================================"
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG_FILE"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Run evaluation
echo "Starting evaluation..."
echo ""

torchrun --nnodes 1 \
         --nproc_per_node $NUM_GPUS \
         run.py $CONFIG_FILE

echo ""
echo "Evaluation completed!"
