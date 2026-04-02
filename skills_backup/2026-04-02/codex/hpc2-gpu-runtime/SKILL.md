---
name: hpc2-gpu-runtime
description: Use when work must run on HKUST-GZ HPC2 GPU development containers instead of the login node, especially when this workstation reaches campus resources through the validated remote VPN jump host `ubuntu@43.134.118.168`, needs the current Model Develop container list, must distinguish `运行`/`等待`/`退出` states, or must connect non-interactively to one running GPU container and verify runtime health.
---

# HPC2 GPU Runtime

## Overview

This is the end-to-end runtime path for HPC2 GPU development containers from the current workstation. Use it to list live containers through the remote VPN path, pick the right running container, connect to it, and verify that GPU work belongs there instead of on `hpc2login`.

## Critical Rules

### 1. Put compute on the right machine

- `hpc2login` is a login node, not the ready GPU runtime.
- Only containers in `运行` state count as ready.
- `等待` means resources have not been assigned yet, so there is no usable GPU endpoint.

### 2. Prefer the validated remote-VPN path from this workstation

Current validated chain:

- jump host: `ubuntu@43.134.118.168`
- campus SOCKS on jump host: `127.0.0.1:1080`
- local direct HTTPS or campus DNS should not be assumed to work

When the local workstation fails with portal SSL or DNS errors, route the portal query through that jump host instead of debugging local networking first.

### 3. Do not persist portal or container passwords

- Read portal credentials from environment variables such as `HPC2_USER` and `HPC2_PASS`.
- Only request sensitive container fields when an actual container connection is needed.
- Prefer environment variables or stdin over writing secrets into files or shell history.

### 4. Present container choices before binding the task

- If there are multiple running containers, summarize them first.
- If the user says "you choose", still show the options and then pick the best fit.
- After one container is chosen, keep the whole task on that container until the user asks to switch.

### 5. Verify the container immediately after login

Run:

```bash
hostname
whoami
pwd
nvidia-smi
```

Also confirm that the mounted project path is visible before editing or launching work.

## Workflow

### Stage 0. Verify the jump-host VPN proxy before touching the portal

Before querying the portal, treat the jump-host SOCKS service as a dependency that can disappear:

```bash
ssh ubuntu@43.134.118.168 'bash -lc '"'"'python3 -c "import socket; socket.create_connection((\"127.0.0.1\", 1080), 5).close()"'"'"''
```

If this fails, stop and report that the remote EasyConnect/SOCKS layer is down. Do not keep retrying portal login until the proxy is healthy again.

### Stage 1. List portal containers through the remote VPN path

Use:

```bash
python3 scripts/fetch_hpc2_gpu_containers_via_remote_vpn.py --status all
```

For machine-readable output:

```bash
python3 scripts/fetch_hpc2_gpu_containers_via_remote_vpn.py \
  --status all \
  --format json
```

This wrapper reuses the sibling `hpc2-gpu-containers` portal logic but executes it on the jump host with the validated SOCKS proxy.

### Stage 2. Decide whether the task belongs on `hpc2login` or on a GPU container

- Use `hpc-login-ssh` for login-node work such as Git operations, directory setup, and lightweight checks.
- Switch to a GPU container for model loading, training, inference, or any task that needs `nvidia-smi`.

### Stage 3. Resolve one running container and connect

If the container name is known:

```bash
python3 scripts/run_hpc2_gpu_container_cmd.py \
  --service-name a800_4_3
```

To run a non-interactive command instead of opening a shell:

```bash
python3 scripts/run_hpc2_gpu_container_cmd.py \
  --service-name a800_4_3 \
  --cmd 'hostname; whoami; pwd; nvidia-smi'
```

If there is exactly one running container and the user did not specify a name, the connector may use it automatically.

### Stage 4. Bind the task to the chosen container

Once connected:

- keep project work under `/hpc2hdd/home/yhao481/jhupload/proj/<project>`
- keep caches under `/hpc2hdd/home/yhao481/jhupload/cache` when the project expects shared storage
- avoid mixing work between multiple GPU containers unless the user explicitly asks for that

## Common Failure Modes

- Jump-host `127.0.0.1:1080` is down: stop early and say the remote EasyConnect/SOCKS service must be restored before portal or container access can work.
- Local portal requests fail with SSL EOF or DNS errors: use the remote-VPN wrapper instead of direct local HTTPS.
- A container is `等待`: do not treat it as ready, because it has no stable node or port yet.
- A container is `退出`: it may keep stale metadata, but it is not a live runtime target.
- The login node is reachable but has no GPU: this is expected; switch to a running container.

## Resources

- Portal listing wrapper: [scripts/fetch_hpc2_gpu_containers_via_remote_vpn.py](scripts/fetch_hpc2_gpu_containers_via_remote_vpn.py)
- Container connector: [scripts/run_hpc2_gpu_container_cmd.py](scripts/run_hpc2_gpu_container_cmd.py)
- Runtime notes: [references/remote-vpn-notes.md](references/remote-vpn-notes.md)
- Sibling portal skill: `../hpc2-gpu-containers/scripts/fetch_hpc2_gpu_containers.py`
