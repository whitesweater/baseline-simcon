# Qwen3-0.6B CommonsenseQA comparison status — 2026-04-05

## Official comparison policy

- Use staged offline artifacts only.
- Use best-checkpoint `summary/all_results.csv`, not last-checkpoint-only comparisons.
- Exclude watcher-produced partial eval outputs from live concurrent runs.
- Do not count failed clean retries that produced no checkpoints.

## Current artifact-backed table

| Candidate | Artifact basis | Best checkpoint | Accuracy | Correct / Total | Status / note |
| --- | --- | --- | ---: | ---: | --- |
| Raw baseline | `CODI_rebuttal_runs/rebuttal_20260325/qwen_commonsense_fast_iteration_v1/results/baseline/qwen3_0p6b_raw_commonsense/summary.json` | – | 38.82% | 474 / 1221 | Reference floor |
| SIM-CoT recovered full run | `CODI_rebuttal_runs/rebuttal_20260325/qwen_commonsense_fast_iteration_v1/results/checkpoint_sweeps/qwen_commonsense_fast_iteration_v1_commonsense_qwen3_0p6b_simcon_worker4_0p6_full_recovered_eval/Qwen3-0.6B-Base/ep_10/lr_0.0005/seed_11/summary/all_results.csv` | `checkpoint-464` | 67.98% | 830 / 1221 | Strongest current offline result |
| SIM-CoT clean run v3 | `CODI_rebuttal_runs/rebuttal_20260325/qwen_commonsense_fast_iteration_v1/results/checkpoint_sweeps/qwen_commonsense_fast_iteration_v1_commonsense_qwen3_0p6b_simcon_phase2_clean_v3/Qwen3-0.6B-Base/ep_10/lr_0.0005/seed_11/summary/all_results.csv` | `checkpoint-232` / `checkpoint-290` (tie) | 65.19% | 795 / 1221 | Official post-train rows only; watcher-noise excluded |
| SIM-CoT + SIRCL | pending lane-B artifact | – | – | – | Wait for official sweep before fair head-to-head verdict |

## Notes that affect fairness

- `slurm_9669242_qwen_csqa_0p6b.err` and `slurm_9669252_qwen_csqa_0p6b.err` show `ChildFailedError` / DDP unused-parameter failures for `_phase2_clean_v1` and `_phase2_clean_v2`; both retries exited before saving checkpoints and should not enter the comparison.
- `slurm_9669287_qwen_csqa_0p6b.out` records the successful `_phase2_clean_v3` run and final sweep output root.
- `slurm_9669287_qwen_csqa_0p6b.err` shows watcher-side CUDA OOM during live concurrent evaluation. The retained official basis is the completed `summary/all_results.csv` with 6 post-train rows (`checkpoint-145/174/203/232/261/290`), not partial watcher artifacts from earlier checkpoints.
- Recovered SIM-CoT full-run sweep still peaks higher than the clean v3 rerun (67.98% vs 65.19%), so the backbone/recipe remains clearly above baseline even after excluding watcher noise.

## Transfer recommendation

- **If GSM8K smoke must start before SIM-CoT+SIRCL lands:** transfer **SIM-CoT first**. It already has two artifact-backed wins over the raw baseline, including one clean rerun and one stronger recovered sweep.
- **If a fair horizontal comparison is still the priority:** wait for the official SIM-CoT+SIRCL post-train sweep, then:
  - promote **both** methods to GSM8K smoke if SIRCL also clears baseline with a credible checkpoint band;
  - otherwise promote **SIM-CoT only** and keep SIRCL as a follow-up debugging lane.
