---
name: hpc2-gpu-containers
description: Use when a task needs the HPC2 web desktop Model Develop workflow, especially logging into https://hpc2login.hpc.hkust-gz.edu.cn/appform/desktop, following the RSA plus SSO portal login, and listing or summarizing current GPU development containers from the JHAI API.
---

# HPC2 GPU Containers

## Overview

Use this skill for the HPC2 web portal flow behind `模型开发`:

- verify that the AppForm desktop login still works
- map the desktop icon to the JHAI route `"/jhai/aiBase#devEnv"`
- read the current development-environment container list
- summarize status, node, image, CPU or GPU size, and GPU bindings

This workflow is read-only by default.

## Critical Rules

### 1. Do not persist credentials

- Never write the user's HPC2 password into files.
- Prefer `--password-env` or an interactive prompt over literal command history.

### 2. Do not surface sensitive response fields unless explicitly asked

The portal API can return SSH passwords, expanded environment variables, and other sensitive fields.

- Default to the bundled script's redacted output.
- Only use `--include-sensitive` when the user explicitly asks for those fields.

### 3. The login flow is not a plain form post

Follow the same sequence as the website:

1. `GET /appform/login`
2. `GET /appform/js/login/login.js` and extract `publicKey`
3. RSA-encrypt the password with that public key
4. `GET /appform/sso/main?uinfo=...`
5. `POST /appform/j_spring_security_check`
6. Reuse the same session for `/appform/desktop` and `/jhai/aiBase/`

### 4. The Model Develop list comes from one API

Use:

```bash
POST /jhai/dockerService/listByModule
{"moduleType": 1}
```

That endpoint returns the same development-environment container data shown under `模型开发`.

### 5. Prefer the normalized fields first

For summaries, prefer:

- `serviceName`
- `serviceType`
- `serviceStatus`
- `serviceNode`
- `cpuNum`
- `gpuNum`
- `resourceComboName`
- `imageName`
- `jobId`
- `gpuBinds`

## Default Workflow

If credentials are already available, run the bundled script:

```bash
export HPC2_USER="your_username"
export HPC2_PASS="your_password"

python3 scripts/fetch_hpc2_gpu_containers.py \
  --username "$HPC2_USER" \
  --password-env HPC2_PASS \
  --status running
```

For a structured response:

```bash
python3 scripts/fetch_hpc2_gpu_containers.py \
  --username "$HPC2_USER" \
  --password-env HPC2_PASS \
  --format json
```

## When The Route Changes

If the portal UI changes, rediscover from the desktop page first:

- locate the desktop app metadata for `模型开发`
- confirm it still points to `/jhai/aiBase#devEnv`
- if needed, inspect `/jhai/aiBase/static/js/devEnv-*.js` for the current API route

## Cross-Machine Install

When deploying this skill to another machine:

- install into the verified remote `~/.codex/skills/hpc2-gpu-containers`
- verify remote `HOME`, `CODEX_HOME`, and `pwd` before relying on `~`
- if WSL direct SSH cannot reach a VPN-only host but Windows can, use:
  - `C:\Windows\System32\OpenSSH\ssh.exe`
  - `C:\Windows\System32\OpenSSH\scp.exe`

## Resources

- `scripts/fetch_hpc2_gpu_containers.py`
