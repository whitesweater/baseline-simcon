#!/bin/bash
# Prepare all evaluation datasets and run multi-dataset eval for LLaMA3, Qwen3-4B,
# and Qwen3-1.7B CoT-SFT models.
# Usage: bash scripts/batch_eval_cot_sft.sh
set -e

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

##############################################################################
# Step 1: Prepare datasets (fix answer types + download missing)
##############################################################################
echo ""
echo "=========================================="
echo "Step 1: Preparing evaluation datasets"
echo "=========================================="

# Fix existing datasets: ensure all answers are strings
python3 -c "
import json, os

fixes = {
    'data/gsm-hard_train.json': True,
    'data/svamp_all.json': True,
    'data/multi-arith_test.json': True,
}

for path, needed in fixes.items():
    if not os.path.exists(path):
        continue
    data = json.load(open(path))
    changed = False
    for d in data:
        if not isinstance(d['answer'], str):
            val = d['answer']
            d['answer'] = str(int(val)) if isinstance(val, float) and val == int(val) else str(val)
            changed = True
    if changed:
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f'Fixed answer types in {path}')
    else:
        print(f'OK: {path}')
"

# Prepare missing datasets (asdiv, math500, aime)
for ds in asdiv math500 aime; do
    expected="data/${ds}_*.json"
    if ls $expected 1>/dev/null 2>&1; then
        echo "OK: $ds already exists"
    else
        echo "Preparing $ds ..."
        python3 preprocessing/all_data.py --dataset $ds
    fi
done

echo ""
echo "Available datasets:"
ls -la data/*.json

##############################################################################
# Step 2: Evaluate LLaMA3 CoT-SFT
##############################################################################
echo ""
echo "=========================================="
echo "Step 2: Evaluating LLaMA3 CoT-SFT"
echo "=========================================="

mkdir -p logs
LLAMA3_TMP=""
QWEN3_TMP=""
QWEN3_1P7B_TMP=""
trap 'rm -f "${LLAMA3_TMP:-}" "${QWEN3_TMP:-}" "${QWEN3_1P7B_TMP:-}"' EXIT

render_eval_config() {
    local template_file="$1"
    local checkpoint_path="$2"
    local out_file="$3"
    sed "s|^load_model_path:.*|load_model_path: ${checkpoint_path}|" "$template_file" > "$out_file"
}

# Find best checkpoint (use last one by default)
LLAMA3_CKPT_DIR="./ckpts/gsm-cot-llama3"
LLAMA3_BEST=$(ls -v ${LLAMA3_CKPT_DIR}/checkpoint_* 2>/dev/null | tail -1)
if [ -z "$LLAMA3_BEST" ]; then
    echo "ERROR: No LLaMA3 checkpoint found in ${LLAMA3_CKPT_DIR}"
else
    echo "Using LLaMA3 checkpoint: ${LLAMA3_BEST}"
    LLAMA3_TMP=$(mktemp /tmp/gsm_cot_llama3_eval.XXXXXX.yaml)
    render_eval_config "args/gsm_cot_llama3_eval.yaml" "$LLAMA3_BEST" "$LLAMA3_TMP"

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    torchrun --nnodes 1 --master_port 29531 --nproc_per_node 1 run_eval.py "$LLAMA3_TMP" \
        -d all \
        2>&1 | tee logs/eval_llama3_multi_$(date +%Y%m%d_%H%M%S).log
fi

##############################################################################
# Step 3: Evaluate Qwen3-4B CoT-SFT
##############################################################################
echo ""
echo "=========================================="
echo "Step 3: Evaluating Qwen3-4B CoT-SFT"
echo "=========================================="

QWEN3_CKPT_DIR="./ckpts/gsm-qwen3-cot-sft"
QWEN3_BEST=$(ls -v ${QWEN3_CKPT_DIR}/checkpoint_* 2>/dev/null | tail -1)
if [ -z "$QWEN3_BEST" ]; then
    echo "ERROR: No Qwen3 checkpoint found in ${QWEN3_CKPT_DIR}"
else
    echo "Using Qwen3 checkpoint: ${QWEN3_BEST}"
    QWEN3_TMP=$(mktemp /tmp/gsm_cot_qwen3_eval.XXXXXX.yaml)
    render_eval_config "args/gsm_cot_qwen3_eval.yaml" "$QWEN3_BEST" "$QWEN3_TMP"

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    torchrun --nnodes 1 --master_port 29532 --nproc_per_node 1 run_eval.py "$QWEN3_TMP" \
        -d all \
        2>&1 | tee logs/eval_qwen3_multi_$(date +%Y%m%d_%H%M%S).log
fi

##############################################################################
# Step 4: Evaluate Qwen3-1.7B CoT-SFT
##############################################################################
echo ""
echo "=========================================="
echo "Step 4: Evaluating Qwen3-1.7B CoT-SFT"
echo "=========================================="

QWEN3_1P7B_CKPT_DIR="./ckpts/gsm-qwen3-1p7b-cot-sft"
QWEN3_1P7B_BEST=$(ls -v ${QWEN3_1P7B_CKPT_DIR}/checkpoint_* 2>/dev/null | tail -1)
if [ -z "$QWEN3_1P7B_BEST" ]; then
    echo "ERROR: No Qwen3-1.7B checkpoint found in ${QWEN3_1P7B_CKPT_DIR}"
else
    echo "Using Qwen3-1.7B checkpoint: ${QWEN3_1P7B_BEST}"
    QWEN3_1P7B_TMP=$(mktemp /tmp/gsm_cot_qwen3_1p7b_eval.XXXXXX.yaml)
    render_eval_config "args/gsm_cot_qwen3_1p7b_eval.yaml" "$QWEN3_1P7B_BEST" "$QWEN3_1P7B_TMP"

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    torchrun --nnodes 1 --master_port 29533 --nproc_per_node 1 run_eval.py "$QWEN3_1P7B_TMP" \
        -d all \
        2>&1 | tee logs/eval_qwen3_1p7b_multi_$(date +%Y%m%d_%H%M%S).log
fi

echo ""
echo "=========================================="
echo "Done! Check results in ckpts/*/multi_eval_*.json"
echo "=========================================="
