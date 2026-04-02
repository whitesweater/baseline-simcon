---
name: hpc4-npu-runtime-scheduler
description: Use when a task on HPC4 needs live NPU inspection, HCCL or rank-table checks, topology awareness, or a decision between single-card jobs, parallel single-card scheduling, and multi-card DDP on an AIStudio container.
---

# HPC4 NPU Runtime Scheduler

## Purpose

This skill is for the part that usually goes wrong after SSH succeeds:

- what cards are really available right now
- whether `torch_npu` can use them
- whether HCCL and rank-table state are healthy enough for DDP
- whether the safer plan is DDP or multiple independent single-card jobs

Use it after the task is already bound to one container.

## First Snapshot

Run the bundled snapshot script first:

```bash
bash scripts/collect_npu_state.sh
```

That snapshot should include:

- `hostname`, `whoami`, `pwd`
- `npu-smi info`
- `npu-smi info -m`
- `npu-smi info -t topo` when supported
- active Python processes on NPUs
- relevant env vars such as `ASCEND_RT_VISIBLE_DEVICES`, `RANK_TABLE_FILE`, `MASTER_*`, `HCCL_*`
- `/etc/hccl_conf.json` presence and a quick preview

## Scheduling Rules

### 1. Prefer single-card when one job fits

If one training or eval job fits on one card, use one card. Do not default to DDP.

### 2. Prefer parallel single-card jobs for independent work

If the jobs are independent, such as separate datasets or separate eval-only runs, spread them across multiple cards as separate processes instead of building DDP groups.

This is the default fallback when HCCL is unhealthy.

### 3. Use DDP only when it is actually needed

Use DDP only when:

- one job truly needs the memory or throughput of multiple cards
- or the user explicitly requests distributed training

Before a long DDP run, verify:

- `torch_npu` imports after sourcing Ascend
- basic NPU tensors work
- HCCL-related env vars are intentional
- the rank table is valid JSON if `/etc/hccl_conf.json` or `RANK_TABLE_FILE` is in play
- topology and visible-card mapping make sense for the chosen group

If the rank table is malformed, missing required fields, or contains bogus device IPs, stop treating DDP as ready.

## Decision Flow

### A. Health check

1. source Ascend env
2. run minimal `torch_npu` tensor ops
3. inspect `npu-smi` and active processes

### B. Card grouping

- for single-card jobs, assign explicit `ASCEND_RT_VISIBLE_DEVICES=<card>`
- for several independent jobs, choose non-overlapping card ids
- for DDP, choose a consistent card list and verify the HCCL path before launch

### C. Launch style

- short smoke test -> foreground
- long training -> background only after log file and pid file are explicit
- DDP smoke test before long DDP training -> always foreground first

## Practical Defaults

### Single-card launch

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

### DDP launch precheck

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export HCCL_CONNECT_TIMEOUT=1800
```

Then run a small foreground smoke test before the real job.

### Visible-device rule on AIStudio containers

- Treat `ASCEND_RT_VISIBLE_DEVICES` as the container's logical device list, not the raw physical IDs shown by `npu-smi`.
- If `torch.npu.device_count()` becomes `0` after setting `ASCEND_RT_VISIBLE_DEVICES=14,15` or similar raw IDs, switch back to logical IDs such as `0,1` and verify with a one-line tensor smoke test.
- On containers that expose physical chips like `0,1,2,3,8,9,14,15`, it is normal for the valid runtime IDs to still be only `0..7`.

### Rank-table workaround for `HcclGetRootInfo ... error code is 19`

- If a DDP smoke test reaches HCCL and then fails with:
  - `HcclGetRootInfo(&hcclID), error code is 19`
  - `Invalid_Ranktable_Configuration`
  - or a message saying the `ip in ranktable is not a valid ip address`
- inspect `/etc/hccl_conf.json` immediately.
- If that file contains placeholder values such as `"device_ip": " "`, do not keep using it for job launches.
- For single-node DDP, create a per-job rank table that contains only the selected devices and omit `device_ip` fields entirely.
- Then launch with `export RANK_TABLE_FILE=/path/to/job_rank_table.json`.

Minimal 2-card single-node rank table:

```json
{
  "version": "1.0",
  "server_count": "1",
  "server_list": [
    {
      "server_id": "103.224.146.33",
      "host_nic_ip": "reserve",
      "device": [
        {"device_id": "0", "rank_id": "0"},
        {"device_id": "1", "rank_id": "1"}
      ]
    }
  ],
  "status": "completed"
}
```

Launch pattern:

```bash
export RANK_TABLE_FILE=/data/user/user224/proj/<project>/logs/hccl_rank_table_2card_01.json
export ASCEND_RT_VISIBLE_DEVICES=0,1
export RANK_SIZE=2
export HCCL_CONNECT_TIMEOUT=300
export HCCL_WHITELIST_DISABLE=1
```

## Common Failure Patterns

### `torch_npu` works but DDP fails immediately

This usually means the container-level HCCL or rank-table state is broken, not that the model code is wrong.

### `npu-smi info -t ip` is unsupported

Do not invent device IPs. If the rank table needs them and the container cannot provide them, prefer single-card scheduling until the container runtime is fixed.

### Large eval OOM with otherwise healthy training

If the model accumulates predictions on NPU, move those accumulations to CPU first. Then retry single-card scheduling before escalating to DDP.
