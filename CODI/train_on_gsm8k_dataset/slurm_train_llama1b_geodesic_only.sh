#!/bin/bash
#SBATCH -p i64m1tga800u
#SBATCH -J llama1b_geo
#SBATCH -o logs/slurm_%j_llama1b_geodesic_only.out
#SBATCH -e logs/slurm_%j_llama1b_geodesic_only.err
#SBATCH -n 16
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH -D /hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI

set -euo pipefail

source ../.venv/bin/activate
source config.env

export CODI_TRAIN_NPROC_PER_NODE="${CODI_TRAIN_NPROC_PER_NODE:-6}"
export CODI_TARGET_NPROC_PER_NODE="${CODI_TARGET_NPROC_PER_NODE:-${CODI_TRAIN_NPROC_PER_NODE}}"
export CODI_GEODESIC_DEVIATION_THRESHOLD="${CODI_GEODESIC_DEVIATION_THRESHOLD:-0.25}"
export CODI_GEODESIC_LOSS_FACTOR="${CODI_GEODESIC_LOSS_FACTOR:-0.1}"
export CODI_GEODESIC_CURVATURE="${CODI_GEODESIC_CURVATURE:--1.0}"

bash train_on_gsm8k_dataset/train_llama1b_geodesic_only.sh "$@"
