#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -J qwen_csqa_0p6b_base
#SBATCH -o logs/slurm_%j_qwen_csqa_0p6b_base.out
#SBATCH -e logs/slurm_%j_qwen_csqa_0p6b_base.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -D /hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI

set -euo pipefail

echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID:-manual}"
echo "Job Name     : ${SLURM_JOB_NAME:-qwen_csqa_0p6b_base}"
echo "Partition    : ${SLURM_JOB_PARTITION:-i64m1tga40u}"
echo "Node         : ${SLURM_NODELIST:-local-shell}"
echo "GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "Start Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

mkdir -p logs

source /hpc2hdd/home/yhao481/jhupload/proj/baseline/.venv/bin/activate

export CODI_BASELINE_BATCH_SIZE="${CODI_BASELINE_BATCH_SIZE:-32}"
export CODI_BASELINE_MAX_NEW_TOKENS="${CODI_BASELINE_MAX_NEW_TOKENS:-256}"

bash train_on_commen_dataset/run_qwen3_0p6b_baseline.sh "$@"

echo "=========================================="
echo "End Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
