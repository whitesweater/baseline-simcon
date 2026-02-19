cd /data/yhao/baseline/CODI && source /data/yhao/baseline/.venv/bin/activate && \
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python test_baseline.py \
    --model_path "/data/yhao/sim-con/modelscope/LLM-Research/Llama-3.2-1B-Instruct" \
    --datasets "gsm8k svamp gsm-hard asdiv commonsense" \
    --batch_size 128 \
    --output_dir "./results/baseline-llama1b" \
    --greedy