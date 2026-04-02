---
name: baseline-hpc2-codi-runtime
description: Use when the `baseline` / `CODI` workspace must run on an HPC2 GPU development container after migration, especially when the fixed shared project root is `/hpc2hdd/home/yhao481/jhupload/proj/baseline`, the authoritative entry points are `CODI/train_on_gsm8k_dataset/*`, stage assets must remain under `CODI_rebuttal_runs`, or current HPC2-specific issues such as container-local `.venv` paths, `peft` versus `torch` compatibility, and non-blocking `Qwen3` watcher failures must be handled without changing the repo structure.
---

# Baseline HPC2 CODI Runtime

## Overview

This skill is the project-specific layer on top of `hpc2-gpu-runtime` and `baseline-rebuttal-prepare`. Use it after you are already inside one HPC2 GPU container and need to prepare assets, launch or monitor `CODI` training, and handle the current baseline-specific environment quirks safely.

## Workflow

### 1. Re-establish project context before touching the runtime

If the task is mainly about project facts, method mapping, or handover rules, read these first from the active checkout:

- `PROJECT_GUIDE.md`
- `NEWCOMER_HANDOVER.md`
- `CODI/config.env`
- `CODI/train_on_gsm8k_dataset/env.sh`
- If it exists, also read `/root/.codex/memories/baseline-collaboration-memory.md` for durable collaboration rules such as how to handle ambiguity and retrospective updates.

Do not invent alternate roots when the project already fixes them.

### 2. Keep the shared project root fixed

Current validated HPC2 project root:

```bash
/hpc2hdd/home/yhao481/jhupload/proj/baseline
```

Current validated stage root:

```bash
/hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1
```

All new logs, outputs, models, caches, manifests, and results should remain under that stage tree instead of drifting back into historical `CODI/outputs` or `CODI/results`.

### 3. Use existing project entry points instead of ad hoc wrappers

Authoritative setup entry point:

```bash
bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama3b --force-datasets
```

Authoritative training entry points:

```bash
bash CODI/train_on_gsm8k_dataset/train_llama1b.sh
bash CODI/train_on_gsm8k_dataset/train_llama3b.sh
bash CODI/train_on_gsm8k_dataset/train_llama8b.sh
bash CODI/train_on_gsm8k_dataset/train_qwen3.sh
```

Default variant behavior:

- default run is `simcon`
- only pass `--sircl` when the user explicitly asks for SIRCL

### 4. Verify the active runtime instead of assuming the interpreter path

Inside the current validated HPC2 GPU container, code and environment do not have to live under the same prefix.

Known-good pattern from this workflow:

- code path under `/hpc2hdd/home/yhao481/jhupload/proj/baseline`
- active `.venv` / `torchrun` path under `/hpc2ssd/JH_DATA/spooler/yhao481/.upload/proj/baseline/.venv`

Always verify with:

```bash
which python
which torchrun
python -V
```

If training is already healthy, do not rewrite paths just to force them to match.

### 5. Treat some failures as blocking and others as non-blocking

Blocking for `llama3b` or other active backbone:

- missing requested model directory under the stage `models/`
- missing `icot` cache
- broken `torchrun` or import path inside the active container

Non-blocking for a `llama3b` run:

- the global watcher still failing on `Qwen3` because the current `transformers` build does not recognize `model_type=qwen3`

Do not delay `llama3b` training just because `Qwen3` is not green.

### 6. Prefer environment fixes over repo monkeypatches

For the current validated HPC2 container:

- `torch 2.11.0+cu130`
- `transformers 4.46.2`
- `peft 0.15.2` is the known-good pin for the active `llama3b` run

If `peft 0.18.x` triggers `DTensor` or distributed import failures, fix the container environment first. Do not keep repository monkeypatches that stub out `DTensor` or break later imports.

### 7. Monitor training from process state, log growth, and GPU usage together

Use all three views:

```bash
ps -u yhao481 -o pid,etimes,cmd | grep -E 'torchrun|train.py'
tail -n 60 <train_log>
nvidia-smi
```

For tqdm-style logs, convert carriage returns before grepping progress:

```bash
perl -pe 's/\r/\n/g' <train_log> | grep -E '[0-9]+/59980 \[' | tail
```

## Resources

- Known issues: [references/known-issues.md](references/known-issues.md)
- Use together with `baseline-rebuttal-prepare`
- Use together with `hpc2-gpu-runtime`
