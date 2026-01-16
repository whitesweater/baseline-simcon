#!/bin/bash
# Training script for Coconut with Trajectory Consistency Loss
# 
# This script trains Coconut with trajectory consistency loss on GSM8k dataset.
# The trajectory consistency loss constrains latent tokens to stay within a radius
# around their geometric center (Fréchet mean), encouraging more coherent reasoning.
#
# Prerequisites:
# 1. GSM8k dataset prepared at ./data/gsm_train.json and ./data/gsm_test.json
# 2. Pretrained model at ./pretrained/gpt2 (or update model_id in config)
# 3. Coconut baseline checkpoint at ./ckpts/gsm-coconut/checkpoint_24 (or update load_model_path)
#
# Usage: bash scripts/train_trajectory.sh [NUM_GPUS]

set -e

# Configuration
NUM_GPUS=${1:-8}  # Default to 8 GPUs
CONFIG_FILE="args/gsm_coconut_trajectory.yaml"

echo "======================================"
echo "Coconut + Trajectory Consistency Training"
echo "======================================"
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG_FILE"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if data directory exists
if [ ! -d "./data" ]; then
    echo "Warning: ./data directory not found."
    echo "Please prepare GSM8k dataset at ./data/gsm_train.json and ./data/gsm_test.json"
    echo ""
fi

# Check if pretrained model exists
if [ ! -d "./pretrained/gpt2" ]; then
    echo "Warning: ./pretrained/gpt2 not found."
    echo "Please download GPT-2 model or update model_id in the config."
    echo ""
fi

# Run training
echo "Starting training..."
echo ""

torchrun --nnodes 1 \
         --nproc_per_node $NUM_GPUS \
         run.py $CONFIG_FILE

echo ""
echo "Training completed!"
echo "Checkpoints saved to: ./ckpts/gsm_coconut_trajectory/"
