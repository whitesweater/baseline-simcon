SAVE_DIR=${SAVE_DIR:-/data/yhao/baseline/CODI/outputs/test_llama1b_eval}
# 留空则不加载本地微调权重，直接用 HuggingFace 完整模型权重
CKPT_DIR=${CKPT_DIR:-}

# HuggingFace config: 默认使用 hf-mirror.com，独立缓存目录避免旧缓存污染；可在执行前覆盖
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# export HF_HOME="${HF_HOME:-/data/yhao/baseline/CODI/.hfhome_mirror}"
export HF_HUB_OFFLINE=0
export HF_HUB_FORCE_DOWNLOAD=1

python test.py \
	--data_name "gsm8k" \
	--output_dir "$SAVE_DIR" \
	--model_name_or_path /data/yhao/sim-con/modelscope/LLM-Research/models--internlm--SIM_COT-LLaMA3-CODI-1B/ \
	--seed 11 \
	--model_max_length 512 \
	--bf16 \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--batch_size 16 \
	--greedy True \
	--num_latent 6 \
	--use_prj True \
	--prj_dim 2048 \
	--prj_no_ln False \
	--prj_dropout 0.0 \
	--inf_latent_iterations 6 \
	--inf_num_iterations 1 \
	--remove_eos True \
	--use_lora True \
	${CKPT_DIR:+--ckpt_dir "$CKPT_DIR"}
