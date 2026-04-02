---
name: runtime-env-rebuild-verify
description: Use when a project has been moved to a new machine or container, or when commands fail in ways that may come from the Python environment rather than the code, and you need a disciplined way to verify interpreter paths, rebuild the environment on the target runtime, and prove it is healthy before relaunching work.
---

# Runtime Env Rebuild Verify

## Overview

This skill separates environment health from project logic.

Use it when:

- a repo was migrated to another machine or container
- `python`, `torchrun`, or imports behave differently than expected
- the code path and interpreter path may point to different storage roots
- training failed and it is unclear whether the problem is environment or code

## Core Rules

### 1. Do not assume code path and env path are the same

A healthy runtime may use:

- code from one mounted path
- `.venv` from another mounted path

Verify first. Do not rewrite paths just to make them look symmetrical.

### 2. Do not copy `.venv` across machines by default

Rebuild the environment on the target runtime unless there is a proven project-specific exception.

### 3. Fix the environment before monkeypatching the repo

If an import or launcher is broken because of package compatibility, repair the environment first.

Do not leave source hacks that only hide a broken runtime.

## Workflow

### 1. Reconfirm the active runtime

Check:

```bash
hostname
whoami
pwd
```

Then verify the project root you intend to use.

### 2. Inspect the actual toolchain

Run:

```bash
which python
which pip
which torchrun
python -V
```

If the project uses `uv`, also check:

```bash
which uv
```

### 3. Distinguish project root from environment root

Record separately:

- code checkout path
- active interpreter path
- active launcher path

Only then decide whether there is a real mismatch or just a different but valid mount layout.

### 4. Rebuild on the target runtime when needed

Preferred order:

1. project-provided bootstrap script
2. `uv sync` for `uv`-managed projects
3. the project’s established fallback installer

Do the rebuild inside the runtime where the workload will actually run.

### 5. Re-verify key packages

After rebuild, check the packages that define the runtime:

- Python
- torch
- transformers
- accelerator-specific libraries
- launcher tooling such as `torchrun`

### 6. Run a smoke test before relaunching

Minimum smoke test:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

If the project is accelerator-specific, include one tiny real device operation when safe.

### 7. Relaunch only through canonical project entry points

After the environment is healthy, use the repo’s existing train or eval entry scripts.

Do not introduce a new wrapper unless the project truly lacks one.

## Common Failure Patterns

- `which python` points outside the intended runtime
- `torchrun` resolves to a different environment than `python`
- copied `.venv` contains binaries incompatible with the current container or driver stack
- package versions changed during migration and now break imports
- source patches were added to hide environment breakage instead of fixing the package set

## Recovery Order

Use this order:

1. verify runtime
2. verify interpreter and launcher
3. rebuild environment on target
4. smoke test imports and accelerator access
5. relaunch the canonical script

## Output Standard

When using this skill, report:

- code root
- environment root
- launcher path
- rebuild method used
- key package versions after rebuild
- smoke-test result
- whether the remaining issue is still environment-related or now a project-logic issue

## Resources

Use together with project migration and runtime skills when they exist.

In this repository, the closest concrete references are:

- `hpc-migration`
- `baseline-hpc2-codi-runtime`
