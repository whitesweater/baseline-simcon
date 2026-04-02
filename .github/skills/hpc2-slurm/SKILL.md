---
name: hpc2-slurm
description: "Use when: writing or reviewing Slurm sbatch scripts for HKUST-GZ HPC2, choosing queues/partitions, setting GPU/CPU/memory resources, submitting or managing jobs, debugging PENDING reasons, or estimating wall-time limits. Covers A800, A40, CPU-only, large-memory, and debug partitions."
---

# HPC2 Slurm Job Submission & Queue Reference

Quick-reference for the HKUST-GZ HPC Phase-2 Slurm cluster.
Official docs: <https://docs.hpc.hkust-gz.edu.cn/docs/hpc12/slurm/job2>

## When To Use

- Writing `#SBATCH` headers for a new training/eval script
- Choosing the right partition for GPU count, wall-time, or cost
- Diagnosing why a job is PENDING
- Managing running jobs (cancel, suspend, extend)

---

## Queue / Partition Cheat-Sheet

### GPU Partitions (A800 80 GB)

| Partition | Mode | Max Resource | Wall-time | Node Spec |
|-----------|------|-------------|-----------|-----------|
| `i64m1tga800u` | Shared (low priority) | 128 cores / 16 GPUs | 7 days | 2×Intel 8358P 32C, 8×A800-SXM4-80GB, 1024 GB RAM |
| `i64m1tga800ue` | Exclusive (mid priority) | 64 cores / 8 GPUs | 7 days | same |
| `emergency_gpu` | Emergency (high priority) | 64 cores / 8 GPUs | 7 days | same |
| `long_gpu` | Shared (low priority) | 128 cores / 16 GPUs | **14 days** | same |

### GPU Partitions (A40 48 GB)

| Partition | Mode | Max Resource | Wall-time | Node Spec |
|-----------|------|-------------|-----------|-----------|
| `i64m1tga40u` | Shared (low) | 128 cores / 16 GPUs | 7 days | 2×Intel 8358P 32C, 8×A40-48GB, 1024 GB RAM |
| `i64m1tga40ue` | Exclusive (mid) | 64 cores / 8 GPUs | 7 days | same |
| `emergency_gpua40` | Emergency (high) | 64 cores / 8 GPUs | 7 days | same |

### CPU-Only Partitions

| Partition | Mode | Max Resource | Wall-time | Node Spec |
|-----------|------|-------------|-----------|-----------|
| `i64m512u` | Shared (low) | 1024 cores | 7 days | 2×Intel 8358P 32C, 512 GB RAM |
| `i64m512ue` | Exclusive (mid) | 1024 cores | 7 days | same |
| `emergency_cpu` | Emergency (high) | 512 cores | 7 days | same |
| `long_cpu` | Shared (low) | 1024 cores | **14 days** | same |
| `i64m512r` | Shared (low) | 128 cores | 7 days | same + 6×1.92 TB data disks |
| `i64m512re` | Exclusive (mid) | 128 cores | 7 days | same |
| `a128m512u` | Shared (low) | 256 cores | 7 days | 2×AMD EPYC 7763 64C, 512 GB RAM |
| `a128m512ue` | Exclusive (mid) | 128 cores | 7 days | same |

### Large-Memory Partitions

| Partition | Mode | Max Resource | Wall-time | Node Spec |
|-----------|------|-------------|-----------|-----------|
| `i96m3tu` | Shared (low) | 192 cores | 7 days | 4×Intel 6348H 24C, **3 TB** RAM |
| `i96m3tue` | Exclusive (mid) | 192 cores | 7 days | same |

### Debug Partition (Free)

| Partition | Max Resource | Wall-time | Notes |
|-----------|-------------|-----------|-------|
| `debug` | 16 cores / 2 GPUs | **30 min** | Free. Mixed nodes: A40-48 GB GPU or CPU-only. For quick smoke tests only. |

---

## Partition Selection Decision Tree

```
Need > 7 days wall-time?
  ├─ Yes, GPU → long_gpu (14 days, A800)
  ├─ Yes, CPU → long_cpu (14 days)
  └─ No
       ├─ Quick debug (< 30 min, ≤ 2 GPUs) → debug (FREE)
       ├─ Need A800 80 GB
       │    ├─ Shared (cheaper) → i64m1tga800u
       │    └─ Exclusive (whole node) → i64m1tga800ue
       ├─ A40 48 GB is enough → i64m1tga40u / i64m1tga40ue
       ├─ CPU only → i64m512u
       └─ Huge memory (> 512 GB) → i96m3tu
```

---

## Job Submission

### Script Mode (recommended)

```bash
sbatch my_job.sh
```

#### Minimal `#SBATCH` Template (GPU)

```bash
#!/bin/bash
#SBATCH -p i64m1tga800u          # partition
#SBATCH -J my_train               # job name
#SBATCH -o logs/slurm_%j.out      # stdout  (%j = job ID)
#SBATCH -e logs/slurm_%j.err      # stderr
#SBATCH -n 16                     # total CPU cores
#SBATCH --gres=gpu:4              # number of GPUs
#SBATCH --mem=200G                # memory (optional, but recommended)
#SBATCH --time=7-00:00:00         # wall-time  D-HH:MM:SS
#SBATCH -D /path/to/workdir       # working directory

set -euo pipefail

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}, GPUs=${SLURM_GPUS_ON_NODE:-?}"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

# activate env
source /path/to/.venv/bin/activate

# run training (torchrun example)
torchrun --nproc_per_node=4 train.py ...

echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
```

### CLI Mode (one-liner)

```bash
sbatch -p i64m1tga800u -n 8 --gres=gpu:1 -o output_%j.txt -e err_%j.txt job.sh
```

### Interactive Mode

```bash
srun -p i64m1tga800u -n 4 --mem=8G --gres=gpu:1 --time=01:00:00 --pty bash
```

### Array Jobs

```bash
#SBATCH --array=1-10          # 10 sub-jobs, index 1..10
#SBATCH -o output_%A_%a.txt   # %A = main job ID, %a = task index
```
Access index via `$SLURM_ARRAY_TASK_ID`.

---

## Resource Sizing Rules of Thumb

| Model Size | GPUs | Partition | CPU cores (`-n`) | Memory (`--mem`) |
|-----------|------|-----------|-------------------|-----------------|
| ≤ 1B params | 1–2 | debug (smoke) / i64m1tga800u | 8 | 60 G |
| 1B–3B | 2–4 | i64m1tga800u | 16 | 120–200 G |
| 3B–8B | 4–8 | i64m1tga800u / i64m1tga800ue | 24–32 | 200–400 G |
| > 8B | 8 (full node) | i64m1tga800ue | 64 | 800 G+ |

General: allocate **~4 CPU cores per GPU** and **~40–50 GB RAM per GPU** as a baseline; adjust from there.

---

## Job Management Commands

| Task | Command |
|------|---------|
| Submit | `sbatch my_job.sh` |
| View my jobs | `squeue -u $USER` |
| Filter by state | `squeue -u $USER -t PENDING,RUNNING` |
| Job details | `scontrol show job <jobid>` |
| Why PENDING? | `scontrol show job <jobid>` → look for `Reason=` field |
| History | `sacct -u $USER` |
| Cancel | `scancel <jobid>` |
| Cancel all mine | `scancel -u $USER` |
| Suspend | `scontrol suspend <jobid>` |
| Resume | `scontrol resume <jobid>` |
| Array sub-job detail | `scontrol show job <jobid>_<taskid>` |

---

## Wall-Time & Extension

- Default limit: **7 days** (most queues), **14 days** (`long_gpu`, `long_cpu`), **30 min** (`debug`)
- Jobs can be extended **once** via the HPC web portal (max +7 days): <https://hpc2login.hpc.hkust-gz.edu.cn/> → 我的作业 → 延长
- Always set `--time=` explicitly; if omitted the queue default applies

---

## Common PENDING Reasons

| Reason | Meaning | Fix |
|--------|---------|-----|
| `Priority` | Other higher-priority jobs ahead | Wait, or use `emergency_*` queues |
| `Resources` | Not enough free nodes/GPUs | Reduce `--gres`/`-n` or wait |
| `QOSMaxGRESPerUser` | You hit per-user GPU limit | Cancel other jobs or use fewer GPUs |
| `AssocMaxJobsLimit` | Too many concurrent jobs | Wait for one to finish |

---

## Tips for This Project

- **Debug smoke test**: use `debug` queue with `--gres=gpu:2 --time=00:25:00 -n 8 --mem=60G` and `--max-steps 3`
- **Production training (A800)**: `i64m1tga800u` with 4–8 GPUs; set `CODI_TRAIN_NPROC_PER_NODE` to match `--gres=gpu:N`
- **Long runs**: `long_gpu` gives 14 days; still extendable once via web portal
- **Logs**: always `mkdir -p logs` before job runs; use `logs/slurm_%j_<tag>.{out,err}` naming
- **torchrun + Slurm**: set `--nproc_per_node` equal to `--gres=gpu:N`; do NOT use `--nnodes > 1` unless multi-node
