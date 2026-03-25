---
name: hpc4-docker-vpn
description: Use when a task needs campus VPN access for HPC4 from a headless Linux machine, especially when EasyConnect must run in Docker, expose a local SOCKS5 proxy on 127.0.0.1:1080, and be validated before any HPC4 SSH or AIStudio step.
---

# HPC4 Docker VPN

## Overview

This skill standardizes the headless VPN path for HKUST-GZ HPC4 access.

It uses the Docker EasyConnect CLI image to establish the VPN, then verifies that the local SOCKS5 proxy can reach both the campus VPN portal and the HPC4 login node.

In the current environment, the validated execution host is:

- `ubuntu@43.134.118.168`

That host already has the required tooling:

- `docker`
- `sshpass`
- `nc`
- `curl`

From this workstation, prefer the bundled remote wrapper scripts so the VPN steps run on that host instead of assuming local Docker access.

## When To Use

- The machine has no GUI, so the native EasyConnect app is not the preferred path.
- The task needs campus-network reachability before connecting to HPC4.
- `hagb/docker-easyconnect:cli` is the chosen VPN runtime.
- The task should reuse an existing `easyconnect` container when possible.

## Workflow

### 1. Reuse the existing container first

Prefer this order:

1. If `easyconnect` is already running, keep it.
2. If `easyconnect` exists but is stopped, start it.
3. Only recreate the container if it does not exist or its CLI options are wrong.

Use one of:

- local on the VPN host: [scripts/ensure_easyconnect_container.sh](scripts/ensure_easyconnect_container.sh)
- from this workstation via SSH: [scripts/ensure_easyconnect_container_remote.sh](scripts/ensure_easyconnect_container_remote.sh)

Required env vars when creating a new container:

- `EASYCONNECT_URL`
- `EASYCONNECT_USER`
- `EASYCONNECT_PASSWORD`

Optional env vars:

- `EASYCONNECT_CONTAINER_NAME` default `easyconnect`
- `EC_VER` default `7.6.7`

### 2. Treat the VPN as a local proxy service

The validated local interface is:

- SOCKS5: `127.0.0.1:1080`

Do not assume the host gets campus DNS or global routes directly. In this environment, the reliable path is host tools going through the SOCKS5 proxy.

### 3. Verify the VPN before touching HPC4

Run one of:

- local on the VPN host: [scripts/check_easyconnect_proxy.sh](scripts/check_easyconnect_proxy.sh)
- from this workstation via SSH: [scripts/check_easyconnect_proxy_remote.sh](scripts/check_easyconnect_proxy_remote.sh)

It verifies:

- the Docker container is running
- `https://remote.hkust-gz.edu.cn` is reachable through `socks5h://127.0.0.1:1080`
- `hpc4login.hpc.hkust-gz.edu.cn:22` is reachable through the same SOCKS proxy

### 4. Hand off to HPC4 access skills

Once the proxy is healthy, the next steps should use:

- `hpc-login-ssh` for the HPC4 login node
- `hpc4-aistudio-npu-pool` to query running NPU containers
- `hpc4-npu-container-ssh` to enter a selected NPU container
- `hpc4-end-to-end-access` when the user wants the whole chain as one workflow

## Boundaries

- This skill only covers the Docker EasyConnect VPN leg.
- It does not choose an NPU container.
- It does not mutate `.profile` with VPN credentials unless the user explicitly asks.

## Resources

- Start or reuse the container with [scripts/ensure_easyconnect_container.sh](scripts/ensure_easyconnect_container.sh)
- Validate proxy health with [scripts/check_easyconnect_proxy.sh](scripts/check_easyconnect_proxy.sh)
- SSH wrapper for container bootstrap: [scripts/ensure_easyconnect_container_remote.sh](scripts/ensure_easyconnect_container_remote.sh)
- SSH wrapper for proxy validation: [scripts/check_easyconnect_proxy_remote.sh](scripts/check_easyconnect_proxy_remote.sh)
