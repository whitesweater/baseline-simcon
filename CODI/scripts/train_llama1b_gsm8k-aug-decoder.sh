SAVE_DIR=/data/user/yhao481/proj/baseline-simcon/CODI/outputs

mkdir -p "${SAVE_DIR}"
export HF_ENDPOINT=https://hf-mirror.com

python train.py \
	--output_dir "${SAVE_DIR}" \
	--expt_name gsm8k_llama1b_latent_baseline-decoder-debug \
	--logging_dir "${SAVE_DIR}/decoder-logs" \
	--logging_steps 10 \
	--model_name_or_path /data/user/yhao481/LLM-Research/Llama-3.2-1B-Instruct \
	--data_name icot \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 32 \
	--gradient_accumulation_steps 4 \
	--bf16 \
	--dataloader_num_workers 4 \
	--dataloader_pin_memory True \
	--dataloader_persistent_workers True \
	--dataloader_prefetch_factor 2 \
	--num_train_epochs 10 \
	--learning_rate 8e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 \
	--lora_alpha 32 \
	--lora_init \
	--save_strategy steps \
	--save_steps 100 \
	--save_total_limit 2 \
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
	--use_decoder True
