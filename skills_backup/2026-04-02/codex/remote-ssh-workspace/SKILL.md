---
name: remote-ssh-workspace
description: Connect to remote development machines over SSH, enforce per-host working-directory boundaries, and inspect experiment artifacts such as csv, logs, and checkpoints before summarizing results. Use when the user asks to connect to a remote server, dev machine, or SSH host, or when remote experiment data must be verified. If host or working directory is ambiguous, enumerate every plausible option and wait for the user to choose before acting.
---

# Remote SSH Workspace

## Overview

Use this skill to connect to registered remote machines, stay inside the user-approved workdir, and verify remote experiment artifacts before summarizing them.

## Workflow

### 1. Resolve the target host and workdir

- Read [references/host-registry.md](references/host-registry.md) for known hosts and allowed directories.
- If it exists, read `/root/.codex/memories/remote-dev-experiment-preferences.md` for persistent collaboration preferences and project conventions.
- If the selected workdir is `/data/yhao/baseline` and it exists, also read `/root/.codex/memories/baseline-collaboration-memory.md`.
- Match an explicit host alias or workdir exactly when the user provides one.
- If the user request maps to multiple plausible hosts or workdirs, list all plausible choices, explain the difference briefly, and stop until the user chooses.
- If the user only says "connect to the dev machine" and no unique mapping exists, do not connect yet.

### 2. Connect safely

- Prefer `ssh <alias>` using the user's SSH config.
- After connecting, verify the location with `pwd`.
- Immediately `cd` into the selected allowed workdir.
- Never read or modify files outside the approved workdir for that task.
- Default to read-only discovery commands until the user asks for edits or execution.
- If the requested workdir conflicts with the registry or prior user constraints, stop and ask.

### 3. Inspect artifacts before conclusions

- For experiment-result organization, verify source artifacts before writing summaries.
- Prioritize `results/*.csv`, `results/*.md`, logs, checkpoint directories, `trainer_state.json`, and `metrics.json`.
- Keep "best checkpoint" and "final checkpoint/final model" as separate views by default.
- Treat missing 1B artifacts as "not run yet" unless the remote files show otherwise.
- Use the baseline definition from memory: SimCoT with decoder enabled and no spectral regularization.
- If `gsm8k -> gsm8k` compares runs with materially different training steps, describe it as baseline leading but not fairly comparable.

### 4. Communicate ambiguity before acting

- Surface ambiguity before taking action: host, workdir, dataset, result slice, output format, or interpretation of missing artifacts.
- Once the user chooses, proceed without repeating the same clarification unless new ambiguity appears.

### 5. Keep the registry current

- When the user gives a new machine or workdir rule, update the host registry and the related memory doc after the task.
- Keep registry entries small and factual: alias, address, auth method, allowed workdirs, likely project paths, and notes.
