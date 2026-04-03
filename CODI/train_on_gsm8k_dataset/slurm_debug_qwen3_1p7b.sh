#!/bin/bash
#SBATCH -p debug
#SBATCH -J qwen3_1p7b_dbg
#SBATCH -o logs/slurm_%j_qwen3_1p7b_debug.out
#SBATCH -e logs/slurm_%j_qwen3_1p7b_debug.err
#SBATCH -n 8
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH -D /hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI

set -euo pipefail

echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Job Name     : ${SLURM_JOB_NAME}"
echo "Partition    : ${SLURM_JOB_PARTITION}"
echo "Node         : ${SLURM_NODELIST}"
echo "GPUs         : ${SLURM_GPUS_ON_NODE:-2}"
echo "Start Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

mkdir -p logs

source /hpc2hdd/home/yhao481/jhupload/proj/baseline/.venv/bin/activate

# 2 GPUs, small batch, only run 3 steps to verify everything works
export CODI_TRAIN_NPROC_PER_NODE=2
export CODI_TARGET_NPROC_PER_NODE=2

bash train_on_gsm8k_dataset/train_qwen3_1p7b.sh \
  --per-device-batch 2 \
  --grad-acc 1 \
  --max-steps 3 \
  "$@"

echo "=========================================="
echo "End Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
