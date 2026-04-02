---
name: project-sync-rsync-layered
description: Use when code, models, caches, logs, or results must be synchronized across machines or clusters, especially when an existing project migration script may already exist and rsync should be applied in layers instead of copying the whole workspace blindly.
---

# Project Sync Rsync Layered

## Overview

This skill standardizes cross-machine project synchronization.

It is built on one rule:

> Sync different asset classes separately.

Do not treat source code, large models, caches, environments, logs, and checkpoints as one monolithic directory tree.

## First Rule: Reuse Existing Migration Scripts

Before inventing a new rsync command, check whether the repo already contains:

- migration scripts
- sync wrappers
- exclude lists
- verification scripts

If those exist, prefer them over ad hoc shell history.

Typical search:

```bash
find . -maxdepth 3 \( -iname '*sync*' -o -iname '*migrate*' -o -iname '*rsync*' -o -iname '*upload*' \)
```

Only fall back to manual rsync when:

- no project script exists
- the existing script does not match the task
- you have verified why the existing script is not appropriate

## Layered Sync Model

Split the workspace into layers:

### Layer 1. Source and small runtime assets

Usually include:

- source code
- shell scripts
- config files
- small local datasets
- manifests

Usually exclude:

- `.venv`
- logs
- outputs
- checkpoints
- large models
- cache directories

### Layer 2. Models and shared caches

Sync separately because they are:

- large
- slower to validate
- often reusable across projects

Examples:

- backbone model directories
- Hugging Face cache
- dataset cache
- ModelScope cache

### Layer 3. Outputs and checkpoints

Do not sync by default unless the task explicitly needs them.

Typical cases:

- pull back logs
- pull specific checkpoints
- archive completed outputs

### Layer 4. Environment

Do not sync `.venv` by default.

Prefer rebuilding on the target machine:

```bash
uv sync
```

or the project’s existing environment bootstrap process.

## Recommended Rsync Pattern

Use:

```bash
rsync -aH --info=progress2 --partial --append-verify
```

Add:

- `-e 'ssh -F <config>'` when a dedicated SSH config is needed
- `--exclude-from=<file>` when the project already ships an exclude list

Avoid `--delete` unless the user explicitly wants the destination to mirror the source.

## Safe Workflow

### 1. Verify source and destination roots

Confirm:

- source repo root
- destination host
- destination real path
- optional compatibility symlink path

### 2. Dry-run first for unfamiliar paths

When the sync shape is new or risky:

```bash
rsync -aHn --info=progress2 ...
```

### 3. Sync code first

Get the workspace into a known state before touching models or outputs.

### 4. Sync models and caches only when needed

Do not move hundreds of GB unless the target really needs them.

### 5. Rebuild or verify the target environment

After sync:

```bash
hostname
whoami
pwd
which python
python -V
```

### 6. Pull logs and outputs explicitly

Use targeted pull commands for:

- `logs/`
- one checkpoint directory
- one results directory

Do not re-run the whole migration script just to fetch a single log file.

## Common Mistakes

- copying `.venv` between incompatible machines
- syncing huge outputs by accident
- overwriting a shared destination with `--delete`
- syncing symlinks without verifying real model contents
- inventing a custom rsync while the repo already has a correct script

## Output Standard

When using this skill, summarize sync work in this format:

- what layer was synced
- source path
- destination path
- whether existing project scripts were reused
- whether environment rebuild is still required
