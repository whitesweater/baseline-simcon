#!/bin/bash
#SBATCH -p i64m1tga800u
#SBATCH -J qwen_csqa_1p7b
#SBATCH -o logs/slurm_%j_qwen_csqa_1p7b.out
#SBATCH -e logs/slurm_%j_qwen_csqa_1p7b.err
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

# Full-run lane: match the broader queue allocation and keep post-train
# CommonsenseQA checkpoint sweeps enabled by default.
export CODI_TRAIN_NPROC_PER_NODE=6
export CODI_TARGET_NPROC_PER_NODE=6
export CODI_POST_TRAIN_EVAL=1
export CODI_POST_TRAIN_DATASETS="commonsense"

bash train_on_commen_dataset/train_qwen3_1p7b.sh "$@"

echo "=========================================="
echo "End Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
