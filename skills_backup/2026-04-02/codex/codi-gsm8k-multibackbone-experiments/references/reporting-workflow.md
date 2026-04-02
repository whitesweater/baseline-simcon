# Reporting Workflow

Use this reference when the user asks for current results, progress, or a summary table.

## Fast Path

1. Open `CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md` when the user wants a cross-backbone report.
2. Open `CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md` when the user wants the full 15-combination table or a per-backbone snapshot.
3. Open `CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md` when the user wants ready-to-cite conclusions.
4. Only drop to raw artifacts when the summaries are missing a row, look stale, or the user explicitly asks for the latest live state.

## Evidence Precedence

Use this precedence order when artifacts disagree:

1. Live `summary/comparison_matrix.csv`
2. Live training or eval logs
3. Current summary docs in `CODI/plots/rebuttal_20260328/`
4. Historical progress docs

## Per-Method Rules

### Non-`cot-sft`

- Read `summary/comparison_matrix.csv` first.
- Treat the first row as the current `best checkpoint(avg)` unless the file format changes.
- Report both:
  - latest checkpoint seen on disk
  - best checkpoint from `comparison_matrix.csv`

### `cot-sft`

- Look under `Coconut/ckpts/...` for checkpoint files or directories.
- Look under `Coconut/logs/` for usable GSM8K or batch-eval logs.
- Do not assume multi-dataset results exist just because training checkpoints exist.

## Status Labels

- `sweep complete`: `comparison_matrix.csv` exists and is usable.
- `partial sweep`: some evaluated checkpoints exist, but the sweep is incomplete or interrupted.
- `no sweep`: no `comparison_matrix.csv`.
- `gsm8k only`: only a citeable GSM8K result is available.
- `stopped early`: the run failed before producing a usable result set.

## Current Practical Snapshot

Use this only as a starting prior and always confirm from the current docs:

- `llama3-3b` is the strongest and most complete backbone story.
- `qwen3-1.7b` currently has a good `codi` line and a weaker `codi+sircl` line.
- `qwen3-4b` currently does not provide a strong implicit-learning confirmation.
