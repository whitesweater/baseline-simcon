---
name: codi-gsm8k-multibackbone-experiments
description: Locate and operate the GSM8K rebuttal experiment matrix in /data/yhao/baseline across five methods (cot-sft, simcot, simcot+sircl, codi, codi+sircl) and three backbones (llama3-3b, qwen3-4b, qwen3-1.7b). Use when Codex needs the exact training wrapper or Coconut args, the manual test command, the expected report/result/weight paths, live checkpoint presence, or accurate current summary and report tables for any of the 15 combinations or for a cross-backbone status report.
---

# CODI GSM8K Multibackbone Experiments

## Overview

Use this skill to map a method and backbone to the exact train command, test command, artifact locations, and live observed artifact status for the GSM8K rebuttal workspace under `/data/yhao/baseline`.

This skill covers:
- `cot-sft`
- `simcot`
- `simcot+sircl`
- `codi`
- `codi+sircl`

Across:
- `llama3-3b`
- `qwen3-4b`
- `qwen3-1.7b`

## Quick Start

From any directory, run:

```bash
python3 /root/.codex/skills/codi-gsm8k-multibackbone-experiments/scripts/locate_experiment.py \
  --method "codi+sircl" \
  --backbone "qwen3-1.7b"
```

To print all 15 combinations:

```bash
python3 /root/.codex/skills/codi-gsm8k-multibackbone-experiments/scripts/locate_experiment.py --all
```

If you want a prewritten document instead of querying the script, read:
- `references/experiment-matrix.md`
- `references/reporting-workflow.md`

## Workflow

1. Normalize the user request to one canonical method and one canonical backbone.
2. If it exists, read `/root/.codex/memories/baseline-collaboration-memory.md` before turning repo artifacts into a user-facing summary.
3. If the user wants a current result, progress summary, or cross-backbone report, open the stage summary docs before scanning raw run directories.
4. Use the locator script first when you need exact commands, path roots, or live artifact presence.
5. If the method is `cot-sft`, use the `Coconut/` workflow.
6. Otherwise use the `CODI/train_on_gsm8k_dataset/` wrappers.
7. For non-`cot-sft` methods, treat `summary/comparison_matrix.csv` as the authoritative source for `Best avg` and `best checkpoint`.
8. Distinguish carefully between `latest checkpoint` and `best checkpoint(avg)`; do not collapse them into one field.
9. If the stage summary docs disagree with live `comparison_matrix.csv` or logs, trust the live artifacts and then patch the docs.

## Canonical Names

Method aliases:
- `cot-sft`: also accept `cot_sft`
- `simcot`: also accept `simcon`
- `simcot+sircl`: also accept `simcon_sircl`, `simcot_sircl`
- `codi+sircl`: also accept `codi_sircl`

Backbone aliases:
- `llama3-3b`: also accept `llama3b`, `llama-3.2-3b`
- `qwen3-4b`: also accept `qwen3`, `qwen3_4b`
- `qwen3-1.7b`: also accept `qwen3-1p7b`, `qwen3_1p7b`

## Artifact Rules

Stage-wide summary docs:
- `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
- `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
- `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
- `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
- `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`

For non-`cot-sft` methods:
- Weights live under `CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/.../checkpoint-*`
- Multi-dataset eval outputs live under `CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/...`
- The most important per-run report files are `summary/comparison_matrix.csv` and `summary/all_results.csv`

For `cot-sft`:
- Weights live under `Coconut/ckpts/...`
- Evaluation logs live under `Coconut/logs/`
- Successful multi-dataset batch eval may also write `multi_eval_*.json` under the checkpoint directory

## Reporting Priorities

When a user asks for "现在的结果", "训练进度", "汇总表", or "3 个 backbone 总结", use this order:

1. `THREE_BACKBONE_SUMMARY_20260330.md` for a ready-made 3-backbone summary.
2. `CURRENT_MULTIMODEL_RESULTS_20260329.md` for the 15-combination table and per-backbone reportable snapshots.
3. `EXPERIMENT_MASTER_SUMMARY.md` for ready-to-cite wording and condensed conclusions.
4. Per-run `comparison_matrix.csv` only when you need to verify or refresh one row.
5. Raw logs and checkpoint directories only when the summaries are missing, stale, or the user asks for the latest live status.

## Result Semantics

- `sweep complete`: `comparison_matrix.csv` exists and covers the checkpoint sweep.
- `partial sweep`: some checkpoints were evaluated and summarized, but the sweep is incomplete or interrupted.
- `no sweep`: checkpoints may exist, but there is no `comparison_matrix.csv`.
- `gsm8k only`: the run has a citeable GSM8K result, but no complete multi-dataset result set.
- `stopped early`: logs or artifacts show the run failed before producing a usable result set.

For `cot-sft`, GSM8K may be reportable even when multi-dataset evaluation failed; confirm from logs before claiming a broader sweep.

## Current Caveat

- `qwen3-1.7b + cot-sft` no longer lives in a purely planned state: a real run exists, but it stopped early and only left `checkpoint_1`.
- `qwen3-4b` is currently a weak implicit backbone story; avoid presenting it as a positive cross-backbone confirmation without re-checking live results.
- `qwen3-1.7b` currently has its strongest reportable line on `codi`, not on `simcot` or `cot-sft`.

## References

Read `references/experiment-matrix.md` when you want the full 15-experiment map in one document.
Read `references/reporting-workflow.md` when you want the fastest reliable path for producing a current result summary.
