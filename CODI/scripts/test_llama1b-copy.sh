# gsm-hard, multi-arith, svamp, gsm8k

export HF_ENDPOINT=https://hf-mirror.com
uv run test.py \
	--data_name "gsm8k" \
	--output_dir /data/yhao/baseline/CODI/outputs/testoutput \
	--model_name_or_path /data/yhao/sim-con/modelscope/LLM-Research/Llama-3.2-1B-Instruct \
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
	--use_lora True \
	--ckpt_dir /data/yhao/baseline/CODI/outputs/gsm8k_llama1b_latent_baseline-decoder-debug/Llama-3.2-1B-Instruct/ep_10/lr_0.0008/seed_11/checkpoint-29900