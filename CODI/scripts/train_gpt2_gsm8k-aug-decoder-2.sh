#!/bin/bash
set -euo pipefail

source /hpc2ssd/JH_DATA/spooler/yhao481/.upload/proj/baseline/.venv/bin/activate
cd /hpc2ssd/JH_DATA/spooler/yhao481/.upload/proj/baseline/CODI

SAVE_DIR=/hpc2ssd/JH_DATA/spooler/yhao481/.upload/proj/baseline/CODI_rebuttal_runs/rebuttal_20260325/outputs
LOG_DIR=/hpc2ssd/JH_DATA/spooler/yhao481/.upload/proj/baseline/CODI_rebuttal_runs/rebuttal_20260325/logs/gsm8k_gpt_latent_decoder-2
MODEL_PATH=/hpc2ssd/JH_DATA/spooler/yhao481/.upload/proj/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/gpt2

mkdir -p "$SAVE_DIR" "$LOG_DIR"

# cp scripts/train_28.20_ce_noref_new_noaux_lat6.sh "$SAVE_DIR"
# /fs-computility/mllm/shared/weixilin/coconut/ckpts/gsm_cot/gsm-cot/checkpoint_13
python train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name gsm8k_gpt_latent_decoder-2 \
	--logging_dir "$LOG_DIR"\
	--logging_steps 10 \
	--model_name_or_path "$MODEL_PATH" \
	--data_name icot \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 64 \
  	--gradient_accumulation_steps 2 \
	--bf16 \
	--num_train_epochs 40 \
	--learning_rate 3e-3 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "no" \
	--save_safetensors False \
	--save_total_limit 1 \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to tensorboard \
    --num_latent 6 \
    --logging_strategy "steps" \
	--use_prj True \
	--prj_dim 768 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--exp_data_num 2000 \
	--remove_eos True \
	--print_ref_model_stats True \
	--use_decoder True \
	--use_trajectory_consistency True \
	--trajectory_space_type euclidean \
	--trajectory_radius_threshold 2 \
	--trajectory_loss_factor 0.1

