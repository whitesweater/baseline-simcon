# Current Defaults

Use this file only for the current `baseline` checkout. Verify against the repo files before changing anything.

## Repo Anchors

- Repo root guide: `PROJECT_GUIDE.md`
- CODI env: `CODI/config.env`
- Stage env: `CODI/train_on_gsm8k_dataset/env.sh`
- Asset prep entry: `CODI/train_on_gsm8k_dataset/prepare_assets.sh`
- Asset prep implementation: `CODI/train_on_gsm8k_dataset/prepare_assets.py`

## Current Rebuttal Isolation

- New runs are isolated under `CODI_RUN_ROOT=/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325`
- The current multimodel stage tag defaults to `multimodel_gsm8k_math500_aime_v1`
- Avoid mixing new products back into historical `CODI/outputs` or `CODI/results`

## Current Backbone Keys

- `llama1b`
- `llama3b`
- `llama8b`
- `qwen3`

These keys map to stage-local model directories via `CODI/train_on_gsm8k_dataset/env.sh`.

## Current Warmed Datasets

- `gsm8k`
- `math500`
- `aime`
- `svamp`
- `gsm-hard`
- `asdiv`

## Current Download Priority

1. local existing path from `CODI/config.env`
2. `ModelScope`
3. `HF Mirror`
4. original Hugging Face with proxy

## Current Hardware Baseline

- `1` machine
- `4` GPUs per node
- `H800 80GB`

Single-GPU runs are supported by the training scripts, but the default planning assumption stays at `1 x 4`.
