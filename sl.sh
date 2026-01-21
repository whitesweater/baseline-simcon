#!/bin/bash
#SBATCH -p emergency_gpu
#SBATCH -J llama1b_euclidean
#SBATCH -o euc_%j.out
#SBATCH -e euc_%j.err
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH -D /hpc2hdd/home/yhao481/jhupload/baseline/CODI

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"

# ===== 激活 uv/venv（关键）=====
source /hpc2hdd/home/yhao481/jhupload/baseline/.venv/bin/activate

# ===== 自检：确认用的是 venv 的 python 且 torch 可 import =====
echo "Python: $(which python)"
python -c "import sys; print('sys.executable:', sys.executable)"
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=false

# ===== 启动训练 =====
bash /data/yhao/baseline/CODI/scripts/codi_euc.sh

echo "Job finished at $(date)"