#!/bin/bash
#SBATCH -p i64m1tga800u
#SBATCH -J qwen_csqa_0p6b
#SBATCH -o logs/slurm_%j_qwen_csqa_0p6b.out
#SBATCH -e logs/slurm_%j_qwen_csqa_0p6b.err
#SBATCH -n 16
#SBATCH --gres=gpu:4
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH -D /hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI

set -euo pipefail

echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Job Name     : ${SLURM_JOB_NAME}"
echo "Partition    : ${SLURM_JOB_PARTITION}"
echo "Node         : ${SLURM_NODELIST}"
echo "GPUs         : ${SLURM_GPUS_ON_NODE:-4}"
echo "Start Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

mkdir -p logs

source /hpc2hdd/home/yhao481/jhupload/proj/baseline/.venv/bin/activate

export CODI_TRAIN_NPROC_PER_NODE=4
export CODI_TARGET_NPROC_PER_NODE=4
export CODI_POST_TRAIN_EVAL=1
export CODI_EVAL_EACH_CHECKPOINT=1
export CODI_POST_TRAIN_DATASETS="commonsense"
export CODI_DDP_FIND_UNUSED_PARAMETERS=True
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

bash train_on_commen_dataset/train_qwen3_0p6b.sh "$@"

echo "=========================================="
echo "End Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
