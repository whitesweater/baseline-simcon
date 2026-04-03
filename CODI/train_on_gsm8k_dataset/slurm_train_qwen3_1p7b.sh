#!/bin/bash
#SBATCH -p i64m1tga800u
#SBATCH -J qwen3_1p7b
#SBATCH -o logs/slurm_%j_qwen3_1p7b.out
#SBATCH -e logs/slurm_%j_qwen3_1p7b.err
#SBATCH -n 24
#SBATCH --gres=gpu:6
#SBATCH --mem=300G
#SBATCH -D /hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI

set -euo pipefail

echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Job Name     : ${SLURM_JOB_NAME}"
echo "Partition    : ${SLURM_JOB_PARTITION}"
echo "Node         : ${SLURM_NODELIST}"
echo "GPUs         : ${SLURM_GPUS_ON_NODE:-6}"
echo "Start Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

mkdir -p logs

source /hpc2hdd/home/yhao481/jhupload/proj/baseline/.venv/bin/activate

# Override GPU count to match Slurm allocation
export CODI_TRAIN_NPROC_PER_NODE=6
export CODI_TARGET_NPROC_PER_NODE=6

bash train_on_gsm8k_dataset/train_qwen3_1p7b.sh "$@"

echo "=========================================="
echo "End Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
