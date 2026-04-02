---
name: train-monitor-eta
description: Use when a CLI training job must be monitored for real progress, health, resource usage, or rough completion time, especially when the run uses torchrun, tqdm-style logs, detached shells, or scheduler jobs that may look alive while doing the wrong thing.
---

# Train Monitor ETA

## Overview

This skill monitors training runs using multiple signals together instead of trusting any single one.

The minimum reliable monitoring set is:

- process state
- accelerator usage
- log growth
- progress lines
- failure signatures

Use this skill for:

- detached training jobs
- torchrun or distributed jobs
- Slurm jobs
- container-based training
- ETA estimation

## Core Rule

Never trust one signal alone.

Examples of misleading single signals:

- a process exists, but the run already failed internally
- a scheduler says `RUNNING`, but the job is on CPU instead of GPU
- the log file exists, but it is no longer growing
- GPU memory is allocated, but no step is progressing

## Monitoring Workflow

### 1. Check process state

```bash
ps -eo pid,etimes,cmd | grep -E 'torchrun|train.py' | grep -v grep
```

Useful fields:

- pid
- elapsed time
- launch command

### 2. Check accelerator usage

For GPUs:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Interpretation:

- high memory plus nontrivial utilization usually means active training
- near-zero memory on all devices usually means idle or failed
- near-zero utilization with high memory can mean startup, deadlock, or data stall

### 3. Normalize tqdm logs before grepping

For carriage-return logs:

```bash
perl -pe 's/\r/\n/g' <train_log> | grep -E '[0-9]+/[0-9]+ \[' | tail -n 20
```

This is often necessary because raw `tail` may hide real progress updates.

### 4. Check failure signatures directly

```bash
grep -nE 'Traceback|RuntimeError|OutOfMemory|CUDA out of memory|ChildFailedError|NCCL' <train_log> | tail -n 20
```

### 5. Check log freshness

Use log mtime and file size when needed:

```bash
stat -c '%y %s' <train_log>
```

## State Classification

### Healthy

Usually all of these are true:

- training processes exist
- progress lines are advancing
- log mtime keeps changing
- accelerator memory is in real use
- no terminal traceback is present

### Startup / data preparation

Usually:

- processes exist
- GPU utilization may still be low
- logs show dataset formatting, cache building, shard loading, or tokenizer work

Do one more check window before declaring it stuck.

### Stalled

Possible signals:

- processes still exist
- log stops growing
- same progress line repeats for a long time
- utilization stays near zero
- no new checkpoints appear

### Failed

Usually one of:

- traceback in the log
- child processes gone
- GPU memory dropped back to near zero after an error
- scheduler still says running, but the actual training subprocess is gone

### False-positive scheduler run

For queued jobs, also verify the runtime is correct.

Red flags:

- job is `RUNNING` but ranks print `cpu`
- driver mismatch warnings
- first step takes implausibly long for the intended hardware

## ETA Estimation

Use the latest progress line:

- current step
- total step
- elapsed time
- step time if shown

Rough ETA:

`remaining_steps * seconds_per_step`

If the log only gives elapsed time and current step:

`seconds_per_step = elapsed_seconds / current_step`

Prefer a rough but honest estimate over false precision.

## Reporting Template

When summarizing a run, include:

- current step and total step
- percent complete
- epoch if available
- current GPU or accelerator usage
- health classification
- rough ETA
- whether any failure signatures were found

Good shape:

- `3B simcon`: `12100/59970`, `20.2%`, healthy, `~45h` remaining, GPUs at `~70/80 GB`

## Common Mistakes

- reading raw tqdm logs without normalizing carriage returns
- trusting `RUNNING` from the scheduler without checking the real process
- interpreting startup preprocessing as a deadlock too early
- reporting ETA from stale logs

## Resource

Use together with environment- or cluster-specific runtime skills when they exist, but keep this skill as the generic monitoring layer.
