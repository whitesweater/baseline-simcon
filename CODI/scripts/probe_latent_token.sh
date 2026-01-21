SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env" || { echo "Error: config.env not found. Copy config.env.example to config.env and configure."; exit 1; }

# SAVE_DIR=${SAVE_DIR:-/data/yhao/baseline/CODI/outputs/test_llama1b_probe}

SAVE_DIR="${CODI_SAVE_DIR}"

python probe_latent_token.py \
	--data_name "zen-E/GSM8k-Aug" \
	--output_dir "$SAVE_DIR/probe" \
	--model_name_or_path "/data/yhao/sim-con/modelscope/LLM-Research/Llama-3.2-1B-Instruct" \
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
	--ckpt_dir "/data/yhao/baseline/CODI/outputs/trained/eucLong"