# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for **SIM-CoT** (Supervised Implicit Chain-of-Thought) and **SIRCL** (trajectory stability plugin). Currently in **rebuttal/revision phase**, focused on cross-backbone experiments across LLaMA-1B/3B/8B and Qwen3-4B on math reasoning tasks (GSM8K, Math500, AIME).

Documentation is primarily in Chinese. The project has extensive internal guides — read them in this order: `PROJECT_GUIDE.md` → `NEWCOMER_HANDOVER.md` → `CODI/README.md` → `CODI/TESTING_GUIDE.md`.

## Repository Layout

- **Git root is `baseline/`**, not `CODI/` — all git operations happen here
- **`CODI/`** — Primary research code (training, evaluation, model, losses, datasets)
- **`Coconut/`** — Alternative backbone (GPT-2/LLaMA via FSDP), actively used for cross-backbone rebuttal
- **`CODI_rebuttal_runs/`** — Isolated output root for rebuttal experiments (since 2026-03-25). New runs go here, not `CODI/outputs/`
- **`CODI/local_datasets/`** — Vendored JSON data (coin_flip, multiarith, svamp) — the runtime source of truth
- **`CODI/SemCoT/`** — External reference repo only, NOT a runtime dependency
- **`CODI/results_useful/`** and **`CODI/final_use_model_codi_sim_sircl/`** — Trusted historical results and paper-final checkpoints
- **`scripts/`** — Migration/deployment utilities (not training)

## Method ↔ Code Name Mapping

| Code name | Paper name | Flag combination |
|-----------|-----------|-----------------|
| `codi` | CODI | `use_decoder=False`, `use_trajectory_consistency=False` |
| `codi_sircl` | CODI + SIRCL | `use_decoder=False`, `use_trajectory_consistency=True` |
| `simcon` | **SIM-CoT** | `use_decoder=True`, `use_trajectory_consistency=False` |
| `simcon_sircl` | **SIM-CoT + SIRCL** | `use_decoder=True`, `use_trajectory_consistency=True` |

**SIRCL** is a pluggable trajectory consistency loss, not a standalone method. It can be added to any backbone (CODI, SIM-CoT, Coconut).

## Key Commands

### Environment Setup
```bash
cd /data/yhao/baseline
source .venv/bin/activate
cd CODI && source config.env
```

### Training (current rebuttal entry points)
```bash
# Prepare models and datasets first
bash CODI/train_on_gsm8k_dataset/prepare_assets.sh

# Training scripts (each runs on 1 node × 4 GPUs via torchrun)
bash CODI/train_on_gsm8k_dataset/train_llama1b.sh
bash CODI/train_on_gsm8k_dataset/train_llama3b.sh
bash CODI/train_on_gsm8k_dataset/train_llama8b.sh
bash CODI/train_on_gsm8k_dataset/train_qwen3.sh
```

### Evaluation
```bash
# Single dataset
python CODI/test.py --model_name <model> --ckpt_dir <path> --data_name gsm8k

# Multi-dataset (loads model once, tests all datasets)
python CODI/test_multi_dataset.py ...

# Batch testing
bash CODI/scripts/batch_test_multi.sh

# Extended eval (Math500, AIME)
bash CODI/train_on_gsm8k_dataset/eval_llama1b_math500_aime.sh
```

### Coconut Training
```bash
cd Coconut
# Stage 1: Coconut baseline
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_coconut.yaml
# Stage 2: SIM-CoT (set load_model_path in YAML to Stage 1 checkpoint)
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_simcot.yaml
```

### HPC Migration
```bash
bash scripts/migrate_baseline_hpc2_longterm.sh
bash scripts/verify_baseline_minimal.sh --repo-root /data/yhao/baseline
```

## Architecture

### Core Files
- `CODI/train.py` — `CustomTrainer` (extends HuggingFace `Trainer`): multi-loss computation, dynamic scheduling via `step_ratio`
- `CODI/src/model.py` — CODI model: CausalLM + LoRA + latent loop + projection layers + multi-loss
- `CODI/test.py` / `CODI/test_multi_dataset.py` — Evaluation with latent iteration inference
- `CODI/src/tokenizer_utils.py` — `load_tokenizer_with_fallback()` for consistent special tokens (pad, bot, eot)

### Loss Modules (all in `CODI/src/`)
- `trajectory_consistency.py` — Fréchet mean constraint (Euclidean/Hyperbolic)
- `trajectory_acceleration.py` — Second-order smoothness
- `trajectory_action.py` — Path energy (least action)
- `trajectory_geodesic.py` — Geodesic deviation
- `rank_diversity.py` — Rank collapse prevention

### Training Arguments (non-obvious)
- `num_latent` — Number of latent token iterations
- `use_decoder` / `use_prj` — Enable SIM-CoT decoder / projection layer
- `distill_loss_factor` (default 20), `explain_loss_factor` (default 1.0) — Loss weighting
- `max_token_num` — Sequence length cap for OOM prevention
- `trajectory_space_type` — `euclidean` or `hyperbolic`

### Special Tokens
`pad_id`, `bot_id` (begin-of-thought), `eot_id` (end-of-thought). Token ID consistency between train and test is critical — always use `load_tokenizer_with_fallback()`.

### LoRA Config
Default: rank 128, alpha 16, dropout 0.05. Target modules are model-family-specific (LLaMA: q/k/v/o/gate/up/down_proj).

## Environment Configuration

`CODI/config.env` is per-machine (gitignored). Key variables:
- `CODI_RUN_ROOT`, `CODI_SAVE_DIR`, `CODI_RESULT_DIR` — Experiment paths
- `CODI_MM_LLAMA1B_PATH`, `CODI_MM_LLAMA3B_PATH`, etc. — Model paths
- `HF_ENDPOINT` — HuggingFace mirror (use `https://hf-mirror.com` in China)

Stage workspace env via `CODI/train_on_gsm8k_dataset/env.sh` sets up isolated directory hierarchy under `CODI_rebuttal_runs/`.

## Trust Priority (for conflicting information)

Paper final version > `results_useful/` > `PROJECT_GUIDE.md` > `NEWCOMER_HANDOVER.md` > active scripts > legacy `CODI/outputs/`

## Package Management

Python >=3.10, <3.13. Uses `uv` with `pyproject.toml`. Virtual env at `.venv/`. Core deps: torch 2.5.1, transformers 4.46.2, peft 0.13.0+, accelerate 1.7.0+.

`icot` dataset cache is the authoritative entry point — loading from cache is intentional, not a hack.
