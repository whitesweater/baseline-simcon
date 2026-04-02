---
name: hpc2-codi-train-triage
description: Use when CODI training on HKUST-GZ HPC2 must be monitored, diagnosed, or relaunched across GPU development containers and Slurm, especially when access goes through the remote VPN jump host, dev containers are preferred over queue jobs, outputs must stay under the rebuttal stage root, and reproducibility constraints forbid changing decoder training, num_latent, max_token_num, or loss logic.
---

# HPC2 CODI Train Triage

## Overview

This skill captures the validated workflow for the `baseline` / `CODI` workspace on HPC2 when runs must be checked, recovered, or relaunched without drifting away from the intended experiment definition.

Current validated environment facts:

- jump host: `ubuntu@43.134.118.168`
- remote SOCKS5 proxy on jump host: `127.0.0.1:1080`
- shared project root: `/hpc2hdd/home/yhao481/jhupload/proj/baseline`
- container code root: `/hpc2ssd/JH_DATA/spooler/yhao481/.upload/proj/baseline`
- stage root: `/hpc2hdd/home/yhao481/jhupload/proj/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1`

Prefer GPU development containers over Slurm jobs when a suitable running container is available. Use Slurm only when no suitable dev container is free or the model does not fit the available container capacity.

## Hard Constraints

Preserve the experiment definition unless the user explicitly approves a change.

Do not change:

- whether the decoder is trained
- `num_latent`
- `max_token_num`
- loss terms or method logic

Treat these as safer knobs, but only when the user has already indicated they are acceptable:

- per-device batch size
- gradient accumulation
- epoch count
- runtime placement: dev container versus Slurm
- runtime-only memory toggles such as allocator settings or gradient checkpointing

Do not silently keep retrying a disproven configuration.

## Workflow

### 1. Refresh runtime targets first

Use `hpc2-gpu-runtime` to list current containers and only consider services in `运行` state as real GPU runtimes.

Ignore:

- `等待`: not ready yet
- `退出`: not a usable runtime target

### 2. Bind each task to one runtime

After choosing a target container, verify immediately:

```bash
hostname
whoami
pwd
nvidia-smi
```

Keep all outputs, logs, checkpoints, and results under the current stage root instead of drifting back into older `CODI/outputs` or `CODI/results`.

### 3. Monitor training from three signals together

Always use all three views:

```bash
ps -eo pid,etimes,cmd | grep -E 'torchrun|train.py' | grep -v grep
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
perl -pe 's/\r/\n/g' <train_log> | grep -E '[0-9]+/[0-9]+ \[' | tail -n 20
```

Also inspect failure signatures directly:

```bash
grep -nE 'Traceback|RuntimeError|OutOfMemory|CUDA out of memory|ChildFailedError|NCCL' <train_log> | tail -n 20
```

Interpretation:

- healthy run: process alive, GPU memory in real use, log mtime advances, tqdm step count increases
- data-prep phase: ranks are CPU-busy, GPU memory is near zero, log only shows dataset formatting or loading; wait one more check window before deciding it is stuck
- failed run: traceback present, child processes gone, or GPU usage drops to zero and stays there after an error

### 4. Estimate progress and ETA from the log

Use the latest tqdm line:

- current step
- total step
- elapsed time
- per-step seconds

A rough ETA is:

`(total_steps - current_step) * seconds_per_step`

Prefer the live tqdm line over scheduler wall-clock time.

### 5. Prefer dev containers before queue jobs

If a dev container can hold the run, use it first.

Only move to Slurm when:

- no suitable GPU dev container is free
- the user explicitly wants queue execution
- the model cannot be run reliably in the available dev containers

### 6. Verify queue jobs early instead of trusting RUNNING state

On HPC2, a Slurm job can show as `RUNNING` while still being invalid for the intended GPU training.

Check immediately:

```bash
export PATH=/opt/slurm/bin:$PATH
squeue -u yhao481
scontrol show job <job_id>
```

Then inspect the first part of the log. Red flags:

- `CUDA initialization: The NVIDIA driver on your system is too old`
- ranks print `cpu` instead of `cuda`
- progress stays at `0/N` for hours

Treat such a job as a bad runtime match, not as a merely slow GPU run. Cancel it and choose a better environment.

## Current Validated Failure Signatures

These are already-observed failure modes for this workspace and should be treated as real prior evidence.

### 8B on 4xA800 dev container, no gradient checkpointing

- `per_device_batch=1`
- `grad_acc=16`
- fails on the first backward pass with CUDA OOM

### 8B on 4xA800 dev container, non-reentrant gradient checkpointing

- still fails on the first backward pass with CUDA OOM
- the extra failing allocation was about 1 GiB in the validated run

### 8B on 4xA800 dev container, reentrant gradient checkpointing

- avoids the first OOM
- then fails under DDP plus LoRA with:
  `Expected to mark a variable ready only once`

### Slurm job using the dev-container `.venv`

- queue node driver can be too old for the dev-container torch build
- the log may warn about the old NVIDIA driver
- ranks may print `cpu`
- first step can take multiple hours

Do not keep such a job alive just because Slurm says `RUNNING`.

## Recovery Order

When multiple runs exist, use this order:

1. Leave a healthy run untouched.
2. Confirm whether the failed run died in data prep, first forward, first backward, or queue-side CPU fallback.
3. Do not relaunch the same invalid configuration again.
4. Prefer a runtime change before a training-definition change.

In this workspace, that means:

- keep a healthy 3B run running
- do not repeatedly relaunch 8B with a combination already shown to OOM or hit DDP checkpointing conflicts

## Environment Checks Before Relaunch

Before trusting a runtime, verify the active Python and torch stack:

```bash
which python
which torchrun
python -V
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Do not assume the same `.venv` works equally well in:

- GPU dev containers
- Slurm queue nodes

## Resources

Use together with:

- `hpc2-gpu-runtime`
- `baseline-hpc2-codi-runtime`
- `hpc-login-ssh`
- `hpc-sbatch`
