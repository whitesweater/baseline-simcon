---
name: hpc4-python-env
description: Use when working on HPC4 or Ascend containers and the task needs a project-local Python environment under /data/user/user224/proj, especially when defaulting to uv, rebuilding a broken .venv against the current container, validating torch_npu with real NPU ops, or falling back to Conda only when uv is not the right route.
---

# HPC4 Python Env

## Overview

This is the single entry point for Python environment setup on HPC4.

Default route:

- `uv`

Fallback route:

- Conda, only when the user explicitly wants it or the platform makes `uv` unsuitable

The environment is not treated as ready until real `torch_npu` ops succeed on the current container.

## Critical Rules

### 1. Project root guard

Work only under:

```bash
/data/user/user224/proj/<project>
```

If the project is elsewhere, stop before editing anything.

### 2. Shared layout

Use:

- `/data/user/user224/proj/.uv-cache` as the shared cache
- `/data/user/user224/proj/<project>/pyproject.toml`
- `/data/user/user224/proj/<project>/.venv`

Do not create one global `pyproject.toml` at `/data/user/user224/proj`.

### 3. Treat system Python as diagnostics unless the project env works

If a system or shared Python already imports `torch` and `torch_npu`, it is acceptable for:

- quick diagnostics
- one-off unblockers
- discovering the platform-supported version pair

But do not say the project environment is ready until the project-local `.venv` works.

## Workflow

### 1. Inspect the current container first

Collect:

- `python3 --version`
- `uv --version` if present
- `npu-smi info`
- existing `.venv/bin/python`
- `.venv/pyvenv.cfg`
- which `set_env.sh` path exists
- whether `libhccl.so` exists

Use:

```bash
bash scripts/check_ascend_runtime.sh
```

### 2. Check whether `.venv` is portable

```bash
ls -l .venv/bin/python .venv/bin/python3 2>/dev/null
sed -n '1,40p' .venv/pyvenv.cfg 2>/dev/null
```

If `.venv/bin/python` points to a missing path on the current container, treat the environment as broken and rebuild it.

### 3. Default to uv

If `uv` is missing, bootstrap the tool first. The final project env must still be:

```bash
/data/user/user224/proj/<project>/.venv
```

Then:

```bash
export UV_CACHE_DIR=/data/user/user224/proj/.uv-cache
mkdir -p "$UV_CACHE_DIR"

cd /data/user/user224/proj/<project>
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY no_proxy NO_PROXY

UV_LINK_MODE=copy uv sync --python "$(command -v python3)"
```

If the repo already has `pyproject.toml` or `uv.lock`, use them directly. Only infer dependencies manually when they are missing.

### 4. Align to the platform-supported torch pair

On Ascend, prefer the working platform pair over a random repo pin. Be explicit about the choice.

If another HPC4 project already proves a healthy pair on the same image, it is reasonable to reuse that same pair.

### 5. Source Ascend before validation

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null \
  || source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null \
  || source /opt/conda/Ascend/ascend-toolkit/set_env.sh

cd /data/user/user224/proj/<project>
source .venv/bin/activate
```

### 6. Validate real NPU ops

```bash
python - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
print("tensor:", torch.tensor([1.0, 2.0], device="npu:0"))
print("zeros:", torch.zeros(4, device="npu:0"))
print("arange:", torch.arange(4, device="npu:0"))
PY
```

If import fails because Ascend runtime libraries are missing, treat it as a runtime/image problem, not an application problem.

If import works but `tensor`, `zeros`, or `arange` fails, the environment is still broken.

### 7. Conda is a fallback, not the default

Use the Conda route only when:

- the user explicitly asks for Conda
- the current HPC4 image or dependency set makes `uv` unsuitable

If you need that route, read:

- `references/conda-route.md`

## Resources

- `scripts/check_ascend_runtime.sh`
- `scripts/prepare_shared_uv_project.sh`
- `references/ascend-runtime.md`
- `references/conda-route.md`
