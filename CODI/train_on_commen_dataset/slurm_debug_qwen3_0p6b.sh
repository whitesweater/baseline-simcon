#!/bin/bash
#SBATCH -p debug
#SBATCH -J qwen_csqa_0p6b_dbg
#SBATCH -o logs/slurm_%j_qwen_csqa_0p6b_debug.out
#SBATCH -e logs/slurm_%j_qwen_csqa_0p6b_debug.err
#SBATCH -n 8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=00:25:00
#SBATCH -D /hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI

set -euo pipefail

echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID:-manual}"
echo "Job Name     : ${SLURM_JOB_NAME:-qwen_csqa_0p6b_dbg}"
echo "Partition    : ${SLURM_JOB_PARTITION:-debug}"
echo "Node         : ${SLURM_NODELIST:-local-shell}"
echo "GPUs         : ${SLURM_GPUS_ON_NODE:-2}"
echo "Start Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

mkdir -p logs

source /hpc2hdd/home/yhao481/jhupload/proj/baseline/.venv/bin/activate

export CODI_TRAIN_NPROC_PER_NODE=2
export CODI_TARGET_NPROC_PER_NODE=2
export CODI_SAVE_STRATEGY=steps
export CODI_SAVE_STEPS=3
export CODI_SAVE_TOTAL_LIMIT=1
export CODI_POST_TRAIN_EVAL=1
export CODI_EVAL_EACH_CHECKPOINT=1
export CODI_POST_TRAIN_DATASETS="commonsense"
export CODI_DDP_FIND_UNUSED_PARAMETERS=True
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

bash train_on_commen_dataset/train_qwen3_0p6b.sh \
  --per-device-batch 8 \
  --grad-acc 1 \
  --max-steps 3 \
  "$@"

echo "=========================================="
echo "End Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
