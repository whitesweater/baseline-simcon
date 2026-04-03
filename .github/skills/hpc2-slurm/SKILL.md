---
name: hpc2-slurm
description: "Use when: writing or reviewing Slurm sbatch scripts for HKUST-GZ HPC2, choosing queues or partitions, sizing GPU or CPU or memory requests, checking real-time free GPU capacity, counting how many users are truly waiting in queue, diagnosing PENDING reasons, or estimating wait time. Covers A800, A40, CPU-only, large-memory, and debug partitions."
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

## Queue Inspection Workflow

Use this workflow when the user asks questions like "how many free cards are there", "can I get 6 or 8 GPUs now", "how many people are waiting", or "how long do I need to wait".

This workflow must stay partition-agnostic. It applies to A800, A40, CPU-only, large-memory, and debug queues. Replace the partition name, resource type, and per-node capacity according to the queue being inspected.

### Ground Truth Order

Do not trust a single command.

1. Use `sinfo -N` to get a fast node-level overview.
2. Use `scontrol show node <node>` to verify any node that appears interesting, especially when judging whether a whole node is actually free.
3. Use `squeue -w <node>` to confirm which jobs are occupying that node.
4. Use `squeue -t PENDING` plus `scontrol show job <jobid>` to decide whether a queued job is really blocked on resources.

Reason: `sinfo` is fast, but `GRES_USED` can be stale or misleading for a short period. If a conclusion depends on whether a node has all 8 GPUs free, always verify with `scontrol show node` and check that `AllocTRES` is empty.

### Recommended Inspection Commands

Node-level overview for any partition:

```bash
sinfo -p <partition> -N -O "NodeList:15,StateLong:15,Gres:20,GresUsed:30,CPUsState:20"
```

If the partition is CPU-only or large-memory and GPU columns are not useful:

```bash
sinfo -p <partition> -N -O "NodeList:15,StateLong:15,CPUsState:20,Memory:12,AllocMem:12"
```

All pending jobs for the target partition set:

```bash
squeue -p <partition1>,<partition2> -t PENDING \
  -o "%.10i %.8u %.18P %.6D %.6C %.15b %.25R %.20V"
```

Verify whether a specific node is truly free:

```bash
scontrol show node <node> | grep -E "NodeName|State|AllocTRES|Gres|CPUAlloc|CPUTot|RealMemory|AllocMem"
```

See all jobs on a node:

```bash
squeue -w <node> -o "%.10i %.20P %.8T %.10M %.12l %.6D %.6C %.10b %.30R" --all
```

Inspect one pending job in detail:

```bash
scontrol show job <jobid>
```

Optional: show only your jobs for the same partition set:

```bash
squeue -u $USER -p <partition1>,<partition2>
```

### How To Count Free Capacity Correctly

First identify the resource dimension that actually gates scheduling.

- GPU queues: GPUs per node are usually the first constraint
- CPU-only queues: CPU cores and memory are usually the first constraint
- Large-memory queues: free memory can be the main bottleneck even if CPU cores remain

For GPU queues, compute free GPUs per node by subtracting used GPUs from the node total.

- `gpu:a800:3(...)` means 3 used, so 5 free on an 8-GPU node
- `gpu:a800:8(...)` means full, so 0 free
- `State=IDLE` with empty `AllocTRES` means the whole node is free
- `draining` or `drng` means do not count that node as available capacity for new long jobs

For CPU-oriented queues, use the same logic with the dominant resource:

- CPU free capacity: from `CPUsState` or `CPUAlloc` versus `CPUTot`
- Memory free capacity: from `RealMemory` versus `AllocMem`

When answering whether a user can start a job now, always use two numbers:

- Total free capacity across the inspected partition set
- Maximum free capacity on any single node

For GPU jobs, the second number determines whether a single-node job can start immediately.

### How To Count "How Many People Are Waiting"

Do not count every PENDING job equally.

Count these separately:

- Resource-blocked jobs: `Reason=Resources` or `Priority`
- Non-resource blocked jobs: `DependencyNeverSatisfied`, `JobHeldUser`, `QOSMaxGRESPerUser`, and similar administrative or dependency reasons

When the user asks "多少人在排队", report:

1. Number of pending jobs blocked by resources
2. Number of unique users among those resource-blocked jobs
3. Optionally mention dead or irrelevant pending jobs separately

Example interpretation:

- One `PENDING` job with `DependencyNeverSatisfied` does not mean one real user is waiting for GPUs
- One user may have multiple pending jobs; count both job count and unique user count

### How To Estimate Wait Time

Use conservative language. Avoid pretending Slurm exposes an exact ETA.

1. Check whether a node already has enough free GPUs for the requested job size
2. If not, inspect the least-occupied nodes with `squeue -w <node>`
3. Compare running job elapsed time `TIME` against `TIME_LIMIT`
4. State whether the wait looks immediate, short, or uncertain

Suggested wording:

- Immediate: a node already has enough free GPUs now
- Short wait: only 1 or 2 GPUs need to be freed on a lightly occupied node, and one occupant has a short time limit
- Uncertain: large jobs are spread across nodes with long remaining limits, so no defensible ETA

Never claim a full node is available unless that node is verified with `scontrol show node`.

### Reporting Template

For queue-capacity questions, always include these four sentences, in this order, even if the user asked only one part.

1. Total free capacity sentence
2. Maximum single-node free capacity sentence
3. Real queue depth sentence
4. Immediate schedulability sentence

Required meaning of the four sentences:

- Total free capacity: summarize aggregate free GPUs, or aggregate free CPUs and memory for non-GPU queues
- Maximum single-node free capacity: state the best node-level capacity currently available
- Real queue depth: count only jobs truly blocked by `Resources` or `Priority`, and report both job count and unique user count
- Immediate schedulability: answer directly whether the requested job can start now; if no job size was specified, state the largest immediately schedulable single-node size

Recommended additions after the four required sentences:

- A per-node summary for the most relevant nodes, ordered by free capacity descending
- A short caveat when `sinfo` and `scontrol` disagree
- A brief note on the likely bottleneck if pending jobs exist despite visible idle resources

Compact GPU example:

```text
当前 A800 总空闲约 24 张。
当前单节点最多空 4 张。
真正因 Resources 或 Priority 排队的有 2 个作业，来自 1 个用户。
如果你要 6 卡单节点作业，现在不能立刻上；如果要 4 卡，可以立即调度。
```

Compact CPU example:

```text
当前该 CPU 队列总空闲约 320 核，空闲内存约 900G。
当前单节点最多空 64 核，空闲内存约 220G。
真正因 Resources 或 Priority 排队的有 3 个作业，来自 2 个用户。
如果你要 32 核 120G，现在可以立刻上。
```

If there is ambiguity, say so explicitly and name the command used for verification.

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
| `DependencyNeverSatisfied` | Upstream dependency will never complete | Cancel and resubmit without the broken dependency |
| `JobHeldUser` | Job is manually held | Release hold or resubmit |

---

## Tips for This Project

- **Debug smoke test**: use `debug` queue with `--gres=gpu:2 --time=00:25:00 -n 8 --mem=60G` and `--max-steps 3`
- **Production training (A800)**: `i64m1tga800u` with 4–8 GPUs; set `CODI_TRAIN_NPROC_PER_NODE` to match `--gres=gpu:N`
- **Long runs**: `long_gpu` gives 14 days; still extendable once via web portal
- **Logs**: always `mkdir -p logs` before job runs; use `logs/slurm_%j_<tag>.{out,err}` naming
- **torchrun + Slurm**: set `--nproc_per_node` equal to `--gres=gpu:N`; do NOT use `--nnodes > 1` unless multi-node
