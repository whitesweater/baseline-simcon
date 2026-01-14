#!/bin/bash
# gsm-hard, multi-arith, svamp, gsm8k

# Load environment config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found. Copy config.env.example to config.env and configure."; exit 1; }

uv run test.py \
	--data_name "gsm8k" \
	--output_dir "${CODI_SAVE_DIR}/testoutput" \
	--model_name_or_path "${CODI_LLAMA1B_PATH}" \
	--seed 11 \
	--model_max_length 512 \
	--bf16 \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--batch_size 128 \
	--greedy True \
	--num_latent 6 \
	--use_prj True \
	--prj_dim 2048 \
	--prj_no_ln False \
	--prj_dropout 0.0 \
	--inf_latent_iterations 6 \
	--inf_num_iterations 1 \
	--remove_eos True \
	--trajectory_radius_threshold 2 \
	--trajectory_max_acceleration 1.0 \
	--trajectory_action_lambda_energy 1.0 \
	--trajectory_action_lambda_length 0.1 \
	--trajectory_curvature -1.0 \
	--use_lora True \
	--ckpt_dir "${CODI_CKPT_DIR:-${CODI_SAVE_DIR}/baseModel}"