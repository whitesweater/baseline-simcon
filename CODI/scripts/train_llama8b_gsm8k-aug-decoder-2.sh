SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found. Copy config.env.example to config.env and configure."; exit 1; }

SAVE_DIR="${CODI_SAVE_DIR}"

mkdir -p "$SAVE_DIR"

# cp scripts/train_28.20_ce_llama1b_dynamic-teacher_factor-exp_lat6.sh "$SAVE_DIR"

torchrun --nnodes 1 --nproc_per_node 4 train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name gsm8k_llama8b_latent_baseline-decoder-2-9_8b \
	--logging_dir "$SAVE_DIR/8b_logs"\
	--logging_steps 10 \
	--model_name_or_path "${CODI_LLAMA8B_PATH}" \
	--data_name icot \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 2 \
  	--gradient_accumulation_steps 8 \
	--bf16 \
	--num_train_epochs 6 \
	--dataloader_num_workers 32 \
	--dataloader_pin_memory True \
	--dataloader_persistent_workers True \
	--dataloader_prefetch_factor 2 \
	--learning_rate 1e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy steps \
	--save_steps 100 \
	--save_total_limit 2 \
	--save_safetensors False \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to tensorboard \
   --num_latent 6 \
   --logging_strategy "steps" \
	--use_prj True \
	--prj_dim 4096 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--exp_data_num 200 \
	--remove_eos True \
	--distill_loss_factor 20 \
	--print_ref_model_stats True \
	--max_token_num 200 \
	--use_decoder True

