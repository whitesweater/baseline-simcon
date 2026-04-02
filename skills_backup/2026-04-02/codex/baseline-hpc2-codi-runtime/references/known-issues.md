# Known HPC2 Issues For Baseline / CODI

## Current Validated Paths

- Shared repo root: `/hpc2hdd/home/yhao481/jhupload/proj/baseline`
- Active stage root: `/hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1`
- Active container `.venv`: `/hpc2ssd/JH_DATA/spooler/yhao481/.upload/proj/baseline/.venv`

## Environment Notes

- `torch 2.11.0+cu130`
- `transformers 4.46.2`
- `peft 0.15.2` is the validated fix for the HPC2 `llama3b` training path

## Known Non-Blocking Warning

The global asset watcher can still fail on `Qwen3` because the container `transformers` build does not recognize `model_type=qwen3`. This does not block `llama1b`, `llama3b`, or `llama8b` when those stage assets already exist.

## Monitoring Signals That Actually Matter

- `torchrun` plus all worker `train.py` processes remain alive
- the training log keeps growing
- progress lines advance
- `loss`, `ce_loss`, and `distill_loss` keep updating
- GPUs remain allocated in `nvidia-smi`
