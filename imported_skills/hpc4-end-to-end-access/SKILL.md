---
name: hpc4-end-to-end-access
description: Use when the user wants the fastest reusable end-to-end path from a headless Linux workstation into HKUST-GZ HPC4, including Docker EasyConnect VPN, HPC4 login through the SOCKS5 proxy, AIStudio NPU pool refresh, NPU container selection, and optional in-container health checks.
---

# HPC4 End-To-End Access

## Overview

This skill orchestrates the entire access chain:

1. Bring up or reuse Docker EasyConnect VPN
2. Verify the local SOCKS5 proxy
3. Log in to HPC4 through that proxy
4. Query AIStudio for the running NPU pool
5. Present NPU container choices to the user
6. Connect to the selected NPU container and verify runtime health

In the current environment, the validated VPN access host is:

- `ubuntu@43.134.118.168`

Prefer the remote wrapper scripts in this skill from the current workstation. They SSH to the validated VPN host and then run the original per-stage helpers there.

## Why This Skill Exists

The full chain contains separable concerns that fail differently:

- VPN bootstrap
- VPN reachability validation
- HPC4 login authentication
- AIStudio token exchange and container discovery
- NPU container SSH
- NPU runtime verification

Keeping these as explicit stages makes retries cheap and diagnostics precise.

## Workflow

### Stage 1. VPN bootstrap

Use `hpc4-docker-vpn` first.

Prefer:

- [../hpc4-docker-vpn/scripts/ensure_easyconnect_container_remote.sh](../hpc4-docker-vpn/scripts/ensure_easyconnect_container_remote.sh)
- [../hpc4-docker-vpn/scripts/check_easyconnect_proxy_remote.sh](../hpc4-docker-vpn/scripts/check_easyconnect_proxy_remote.sh)

### Stage 2. HPC4 login through SOCKS5

The validated local-to-remote login helper is [scripts/hpc4_login_via_vpn_remote.sh](scripts/hpc4_login_via_vpn_remote.sh).

Credential source in this environment:

- `~/.profile`
- `hpc4user`
- `hpc4password`

Immediate verification commands:

```bash
hostname
whoami
pwd
```

### Stage 3. Refresh and list the NPU pool

Run [scripts/refresh_npu_pool_via_hpc4_remote.sh](scripts/refresh_npu_pool_via_hpc4_remote.sh).

This path is intentionally remote-executed through the HPC4 login node because the local workstation may not resolve `hpc4login.hpc.hkust-gz.edu.cn` directly outside the VPN proxy.

Default project in the validated environment:

- `project_id=670`

Output file:

- `~/npu_pool_summary.json`

### Stage 4. Present choices, do not auto-pick

Show every running NPU container from the pool file:

- name
- run id
- NPU count and series
- image
- SSH command

Never auto-select a container when multiple running choices exist.

### Stage 5. Connect to the selected NPU container

Use [scripts/connect_npu_from_summary_remote.sh](scripts/connect_npu_from_summary_remote.sh) with the chosen 1-based index.

After login, verify:

```bash
hostname
whoami
pwd
npu-smi info
```

## Current Validated Environment Facts

- Docker EasyConnect container name: `easyconnect`
- Local SOCKS5 proxy: `127.0.0.1:1080`
- HPC4 login alias path: SOCKS5 proxy plus password SSH
- Default AIStudio project id: `670`
- Known running container summary is persisted in `~/npu_pool_summary.json`

## Boundaries

- This skill assumes the HPC4 and VPN credentials already exist.
- It does not decide which project id to use if the user asks for a different AIStudio project.
- It should stop for explicit user choice before entering one NPU container among several running options.

## Resources

- Process reference: [references/hpc4_access_flow.md](references/hpc4_access_flow.md)
- HPC4 login helper: [scripts/hpc4_login_via_vpn.sh](scripts/hpc4_login_via_vpn.sh)
- Pool refresh helper: [scripts/refresh_npu_pool_via_hpc4.sh](scripts/refresh_npu_pool_via_hpc4.sh)
- Container connector: [scripts/connect_npu_from_summary.sh](scripts/connect_npu_from_summary.sh)
- Remote AIStudio query payload: [scripts/list_remote_npu_pool.py](scripts/list_remote_npu_pool.py)
- Remote login wrapper: [scripts/hpc4_login_via_vpn_remote.sh](scripts/hpc4_login_via_vpn_remote.sh)
- Remote pool refresh wrapper: [scripts/refresh_npu_pool_via_hpc4_remote.sh](scripts/refresh_npu_pool_via_hpc4_remote.sh)
- Remote container connector wrapper: [scripts/connect_npu_from_summary_remote.sh](scripts/connect_npu_from_summary_remote.sh)
