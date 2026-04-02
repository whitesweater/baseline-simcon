---
name: baseline-rebuttal-prepare
description: Prepare and verify the `baseline` / `CODI` rebuttal workspace before training or evaluation. Use when Codex needs to locate the repo root, source `CODI/config.env`, prepare or download supported backbone models and dataset caches with the existing GSM8K asset scripts, honor the download order local path then ModelScope then HF Mirror then Hugging Face with proxy, or confirm that new runs stay isolated under the rebuttal run root instead of historical `CODI/outputs` and `CODI/results`.
---

# Baseline Rebuttal Prepare

Use the existing project scripts and env files. Do not create parallel setup wrappers unless the user explicitly asks for new automation.

## Find The Repo

Locate the active checkout first.

- Prefer `git rev-parse --show-toplevel`.
- Confirm the root contains `PROJECT_GUIDE.md`, `CODI/config.env`, and `CODI/train_on_gsm8k_dataset/prepare_assets.sh`.
- If it exists, read `/root/.codex/memories/baseline-collaboration-memory.md` for durable collaboration and reporting preferences before acting on ambiguous repo tasks.
- If the current directory is not inside the repo, search for the checkout instead of assuming relative paths.

If the task is mainly about project facts, repo boundaries, method mapping, or handover rules, read the root `PROJECT_GUIDE.md` and stop there. Do not use this skill as a substitute for project documentation.

## Load The Existing Environment

From the repo root, reuse the checked-in env files:

```bash
source CODI/config.env
source CODI/train_on_gsm8k_dataset/env.sh
source "${CODI_VENV_PATH}"
```

Validate these before doing anything heavier:

- `CODI_VENV_PATH` exists.
- `CODI_RUN_ROOT` points to the rebuttal-stage run root, not historical output trees.
- `${CODI_MULTIMODEL_ROOT}` and its subdirectories are writable.

If you need the current default paths, model aliases, or dataset list, read [current-defaults.md](./references/current-defaults.md).

## Prepare Assets Through The Existing Entry Point

Use `CODI/train_on_gsm8k_dataset/prepare_assets.sh` as the single setup entry point.

- Use `bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama1b --force-datasets` for one backbone.
- Use `--models llama1b llama3b llama8b qwen3` when the user wants the full current set.
- Use `--skip-datasets` only when the user explicitly wants to avoid dataset warmup.

Let `prepare_assets.sh` keep ownership of:

- local model reuse and symlink creation
- manifest writing
- `icot` cache copy into the stage cache
- remote fallback order
- dataset warmup for the current evaluation set

Do not reimplement this flow in ad hoc shell snippets unless the checked-in script is broken and you have verified why.

## Honor Download Priority And Proxy Rules

Keep the current download order:

1. Reuse the local model path from `CODI/config.env` when it is already complete.
2. Try `ModelScope`.
3. Fall back to `HF Mirror`.
4. Use the original Hugging Face endpoint only with proxy enabled.

`prepare_assets.sh` already sources `/root/.bashrc` and toggles `proxy_on` / `proxy_off` while switching backends. Do not hardcode a new proxy port if the machine already documents the current Mihomo config.

Only add manual `HTTP_PROXY` / `HTTPS_PROXY` exports when you are in a non-interactive shell path that bypasses `/root/.bashrc` and you have confirmed that direct Hugging Face access is required.

## Validate The Workspace Before Training

Before launching training or evaluation, confirm:

- every requested model directory contains `config.json` and at least one weight file
- `${CODI_MULTIMODEL_ICOT_CACHE_DIR}/dataset_icot_0a5b3650760a22ea.pt` exists
- manifests exist under `${CODI_MULTIMODEL_MANIFEST_DIR}`
- new outputs, results, logs, caches, and model copies stay under `${CODI_RUN_ROOT}`
- runtime datasets are resolved from `CODI/local_datasets/` when local files are expected

If validation fails, fix the environment or rerun `prepare_assets.sh` instead of silently skipping checks.

## Hand Off To Direct Training Scripts

After preparation succeeds, use the direct entry points under `CODI/train_on_gsm8k_dataset/`.

- Default variant is `simcon`.
- Pass `--sircl` or `--variant simcon_sircl` only when the user explicitly wants the SIRCL plugin enabled.
- Keep all new run products in the rebuttal stage tree; do not redirect the current mainline back into `CODI/outputs` or `CODI/results`.

If you need the current exact run root, current stage tag, model keys, or default datasets, read [current-defaults.md](./references/current-defaults.md) and then verify against `CODI/config.env` plus `CODI/train_on_gsm8k_dataset/env.sh`.
