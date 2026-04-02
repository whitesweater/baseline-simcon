---
name: baseline-collaboration-retrospective
description: Use when working in `/data/yhao/baseline` and the user asks for reflection, retrospective, memory updates, workflow conventions, or to learn from Claude/Copilot instructions. This skill loads the durable collaboration memory for the repo, separates stable instructions from transient assistant history, and updates only the reusable lessons.
---

# Baseline Collaboration Retrospective

## Overview

Use this skill when the task is about improving how Codex works with the `baseline` repo rather than about a single experiment run.

## Workflow

### 1. Load the durable collaboration memory first

- Read `/root/.codex/memories/baseline-collaboration-memory.md` before proposing updates.
- Treat that file as the compact source of durable behavioral preferences for this repo.

### 2. Separate stable guidance from transient artifacts

When the task involves learning from other assistants or past work, keep these layers separate:

1. explicit instruction files such as `CLAUDE.md` and `.github/copilot-instructions.md`
2. explicit memory files
3. project skill docs
4. settings and permissions
5. transient artifacts such as session logs, telemetry, file-history, extension internals, and shell history

Only layers 1 through 4 are candidates for durable memory by default.

### 3. Keep retrospective updates small and reusable

- Add a new memory entry only when it changes future behavior across multiple tasks.
- If the lesson changes execution flow, patch the existing skill that owns that workflow instead of creating another baseline-specific SOP.
- If the lesson is only about user collaboration style, prefer updating the memory doc over expanding unrelated skills.

### 4. Use the project source of truth instead of re-documenting the repo

For project facts, method mapping, active entry points, and trusted outputs, defer to:

- `PROJECT_GUIDE.md`
- `NEWCOMER_HANDOVER.md`
- checked-in train and eval scripts
- existing Codex skills already tied to runtime, migration, reporting, or experiment status

Do not duplicate those facts into the memory doc unless they become a stable collaboration rule.

### 5. Finish with an explicit behavior change

A retrospective is not complete until one of these happened:

- the memory doc was updated with a durable new rule
- an existing skill was patched to follow the new rule
- both, when the lesson affects both collaboration style and execution flow
