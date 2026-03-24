# Rebuttal Workspace Rules

This note records the workspace rules agreed on for the post-submission revision cycle.

## Git boundary

- The actual Git repository root is `/data/yhao/baseline`.
- `CODI/` is a subdirectory inside that repository.
- All code changes for the revision cycle should be done on dedicated branches created from the `baseline` repo root.

## Output isolation

Starting from 2026-03-25, new experiments should not write into the legacy `CODI/outputs` or `CODI/results` trees.

Recommended local machine layout:
- `CODI_RUN_ROOT=/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325`
- `CODI_SAVE_DIR=${CODI_RUN_ROOT}/outputs`
- `CODI_RESULT_DIR=${CODI_RUN_ROOT}/results`

This keeps rebuttal checkpoints, logs, metrics, and summaries separate from historical runs.

## Trusted historical artifacts

For reading historical evidence, prefer:
- `CODI/results_useful/`
- plotting / analysis scripts under `CODI/plots/` and the source files they read
- the final paper checkpoints under `CODI/final_use_model_codi_sim_sircl/`

## SemCoT handling

- `CODI/SemCoT/` is treated as an external reference repository.
- The active CODI code should not rely on the whole SemCoT repo at runtime.
- The dataset JSON files actually used by CODI are copied into `CODI/local_datasets/`.

## Commit policy

Commit:
- code
- scripts
- concise documentation / notes needed to reproduce the workflow

Do not commit:
- checkpoints
- logs
- results
- large generated artifacts
