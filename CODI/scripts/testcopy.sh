#!/bin/bash
# gsm-hard, multi-arith, svamp, gsm8k, commonsense
# 用法: ./testcopy.sh [dataset_name] [ckpt_dir]
#   dataset_name: gsm8k (默认), gsm-hard, multi-arith, svamp, commonsense
#   ckpt_dir: checkpoint 目录路径（可选）

# Load environment config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found. Copy config.env.example to config.env and configure."; exit 1; }

# 解析命令行参数
DATA_NAME="${1:-gsm8k}"
CKPT_DIR="${2:-${CODI_CKPT_DIR:-${CODI_SAVE_DIR}/gsm8k_llama1b_latent_decoder-trajectory-euclidean/Llama-3.2-1B-Instruct/ep_10/lr_0.0008/seed_11/checkpoint-29990}}"

echo "=========================================="
echo "测试配置:"
echo "  数据集: ${DATA_NAME}"
echo "  Checkpoint: ${CKPT_DIR}"
echo "  结果目录: ${CODI_RESULT_DIR}/${DATA_NAME}/"
echo "=========================================="

# 创建数据集专用结果目录
mkdir -p "${CODI_RESULT_DIR}/${DATA_NAME}"

python test.py \
	--data_name "${DATA_NAME}" \
	--output_dir "${CODI_SAVE_DIR}/testoutput/${DATA_NAME}" \
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
	--ckpt_dir "${CKPT_DIR}"

echo ""
echo "✅ 测试完成！结果已保存到: ${CODI_RESULT_DIR}/${DATA_NAME}/"