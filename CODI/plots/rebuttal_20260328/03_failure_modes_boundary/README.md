# 03 - Failure Modes Boundary

This folder packages the materials for rebuttal point 3:

> If the trajectory is already wrong, does that mean the centroid is wrong too, and SIRCL only helps when the original reasoning is already good?

## Included Here

- `analyze_sircl_failure_modes.py`
- `results_main/`
- `report.md`

## What This Tool Measures

- transition buckets such as recovered, regressed, both-correct, and both-wrong
- gain split by baseline-correct vs baseline-wrong
- geometry summaries for each group
- failure and recovery examples

## Main Rebuttal Takeaway

This is the strongest folder for answering the applicability-boundary question directly.

Start with `results_main/summary.md`, then use `transition_summary.csv` and `gain_by_baseline_bucket.csv` as the compact evidence.
