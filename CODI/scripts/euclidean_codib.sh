#!/bin/bash
# LLaMA 1B Training with Decoder + Euclidean Trajectory Consistency
# 用法: ./euclidean.sh [EXPT_NAME] [--test]
#   EXPT_NAME: 实验名称，默认 decoder-trajectory-euclidean
#   --test:    训练完成后自动测试

# Load environment config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found. Copy config.env.example to config.env and configure."; exit 1; }
source /data/yhao/baseline/.venv/bin/activate

# 参数解析
EXPT_NAME="${1:-decoder-trajectory-euclidean}"
AUTO_TEST=false
for arg in "$@"; do
    if [[ "$arg" == "--test" ]]; then
        AUTO_TEST=true
    fi
done
MASTER_PORT=22487

SAVE_DIR="${CODI_SAVE_DIR}"
mkdir -p "${SAVE_DIR}"

echo "=========================================="
echo "EXPT_NAME: ${EXPT_NAME}, AUTO_TEST: ${AUTO_TEST}"
echo "=========================================="

torchrun --nnodes 1 --master_port ${MASTER_PORT} --nproc_per_node 4 train.py \
	--output_dir "${SAVE_DIR}" \
	--expt_name ${EXPT_NAME} \
	--logging_dir "${SAVE_DIR}/logs/${EXPT_NAME}-logs" \
	--logging_steps 10 \
	--model_name_or_path "${CODI_LLAMA1B_PATH}" \
	--data_name icot \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 32 \
	--gradient_accumulation_steps 1 \
	--bf16 \
	--dataloader_num_workers 4 \
	--dataloader_pin_memory True \
	--dataloader_persistent_workers True \
	--dataloader_prefetch_factor 2 \
	--num_train_epochs 14 \
	--learning_rate 8e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 \
	--lora_alpha 32 \
	--lora_init \
	--save_strategy epoch \
	--save_total_limit 20 \
	--save_safetensors False \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type cosine \
	--do_train \
	--report_to tensorboard \
	--num_latent 6 \
	--logging_strategy steps \
	--use_prj True \
	--prj_dim 2048 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--exp_data_num 200 \
	--remove_eos True \
	--distill_loss_factor 20 \
	--print_ref_model_stats False \
	--max_token_num 200 \
	--use_decoder False \
	--use_trajectory_consistency False \
	--trajectory_space_type euclidean \
	--trajectory_radius_threshold 2 \
	--trajectory_loss_factor 0.1

# =============================================================================
# 训练完成后自动测试
# =============================================================================
if [[ "$AUTO_TEST" == "true" ]]; then
    echo ""
    echo "=========================================="
    echo "训练完成，开始自动测试..."
    echo "=========================================="
    
    # 使用 batch_test_multi.sh 进行测试
    TRAINED_DIR="${SAVE_DIR}" \
    RESULTS_DIR="${CODI_RESULT_DIR}/${EXPT_NAME}" \
    "${SCRIPT_DIR}/batch_test_multi.sh" -m "${EXPT_NAME}" -d "gsm8k"
    
    echo "=========================================="
    echo "测试完成！结果保存在: ${CODI_RESULT_DIR}/${EXPT_NAME}"
    echo "=========================================="
fi
