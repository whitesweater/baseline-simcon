#!/bin/bash

# 验证模型是否正常（从 HuggingFace 自动下载）

SAVE_DIR=/data/yhao/baseline/CODI/outputs/verify_test
MODEL_NAME="internlm/SIM_COT-LLaMA3-CODI-1B"

# 代理配置（端口 7890）
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
echo "代理已启用: 127.0.0.1:7890"

# HuggingFace 镜像配置
# export HF_ENDPOINT="https://hf-mirror.com"
# export HF_HUB_OFFLINE=0
echo "使用 HuggingFace 镜像: ${HF_ENDPOINT}"

echo "================================"
echo "开始验证模型..."
echo "================================"

python test.py \
    --data_name "gsm8k" \
    --output_dir "$SAVE_DIR" \
    --model_name_or_path "$MODEL_NAME" \
    --seed 11 \
    --model_max_length 512 \
    --bf16 \
    --lora_r 128 \
    --lora_alpha 32 \
    --lora_init \
    --batch_size 4 \
    --greedy True \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 2048 \
    --prj_no_ln False \
    --prj_dropout 0.0 \
    --inf_latent_iterations 6 \
    --inf_num_iterations 1 \
    --remove_eos True \
    --use_lora True

echo ""
echo "================================"
echo "验证完成！检查输出："
echo "  日志: $SAVE_DIR/"
echo "================================"
