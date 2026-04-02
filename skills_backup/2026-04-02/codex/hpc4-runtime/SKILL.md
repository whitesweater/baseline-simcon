---
name: hpc4-runtime
description: Use when a task runs on HPC4 and needs the real runtime workflow between hpc4login and AIStudio Ascend containers, especially when refreshing the live NPU pool, showing running container choices, binding the task to one container, or keeping outputs writable under /data/user/user224/proj.
---

# HPC4 Runtime

## Overview

This is the single entry point for HPC4 runtime work. Use it for:

- deciding whether work belongs on `hpc4login` or on an AIStudio NPU container
- refreshing the live AIStudio pool
- presenting the running container choices
- binding the task to one chosen container
- running shared-directory work safely under `/data/user/user224/proj`

## Critical Rules

### 1. Project root guard

All edits and outputs must stay under:

```bash
/data/user/user224/proj/<project>
```

If the target path is elsewhere, stop before editing anything.

### 2. Put the task on the right machine

- `hpc4login` is for light CPU-side work, code edits, and short checks.
- AIStudio containers are the only ready NPU targets.
- Only containers in `运行中` state count as ready.

### 3. Refresh the pool before container choice when needed

Default pool file:

- `~/.codex/tmp/hpc4_runtime/npu_pool.json`

Prefer the current token first, then fall back to portal login when needed:

```bash
eval "$(python3 scripts/ensure_aistudio_token.py --shell)"

python3 scripts/refresh_npu_pool.py \
  --project-url 'https://hpc4login.hpc.hkust-gz.edu.cn/AIStudio/pagebox/aiarts/project/expertDevelop?proId=670&version=v2&external=false&pageNum=1&labId=1922884' \
  --output ~/.codex/tmp/hpc4_runtime/npu_pool.json
```

### 4. Always present the running choices

Show every running entry with:

- container name
- NPU count and model
- image name
- run id
- SSH command

Use:

```bash
python3 scripts/list_npu_pool.py \
  --pool-file ~/.codex/tmp/hpc4_runtime/npu_pool.json
```

If the user explicitly says "you choose", still summarize the options first, then pick the best-fit container and say why.

### 5. Bind the task to one container

After the user chooses a running container, keep the whole task bound to that container until the user asks to switch.

### 6. Connect non-interactively and verify immediately

Use the pool entry's decoded SSH fields:

- `ssh.user`
- `ssh.host`
- `ssh.port`
- `ssh.password`

Prefer `sshpass` or another non-interactive password flow when available.

After login, verify:

```bash
hostname
whoami
pwd
npu-smi info
```

Also confirm the mounted project path before running anything.

### 7. Keep shared outputs deletable by `user224`

Before creating shared directories from a root shell inside a container, use the owner-safe pattern:

```bash
OWNER_REF_DIR="/data/user/user224"
OWNER_UID="$(stat -c '%u' "$OWNER_REF_DIR")"
OWNER_GID="$(stat -c '%g' "$OWNER_REF_DIR")"

ensure_owner_dir() {
  install -d -m 755 -o "$OWNER_UID" -g "$OWNER_GID" "$1"
}

run_as_owner() {
  setpriv --reuid="$OWNER_UID" --regid="$OWNER_GID" --clear-groups \
    env HOME="$OWNER_REF_DIR" USER="user224" LOGNAME="user224" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="${PYTHONPATH:-}" \
    "$@"
}
```

### 8. No `nohup ... &` inside SSH heredocs

Do not launch background jobs with `nohup ... &` inside SSH heredocs. Run the command directly in the remote shell block instead.

### 9. Always source Ascend before NPU execution

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null \
  || source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null \
  || source /opt/conda/Ascend/ascend-toolkit/set_env.sh
```

## Resources

- `scripts/ensure_aistudio_token.py`
- `scripts/aistudio_auth.py`
- `scripts/refresh_npu_pool.py`
- `scripts/list_npu_pool.py`
