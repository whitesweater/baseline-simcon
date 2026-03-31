# 05 - Correctness-Stratified Trajectories

This folder packages the materials for rebuttal point 5:

> Why analyze only all-correct samples? What changes if we inspect failed, mixed, or flipped cases?

## Included Here

- `analyze_trajectory_by_correctness.py`
- `results_main/`
- `report.md`

## What This Tool Produces

- PCA or UMAP trajectory projections
- per-step `r_t` curves with confidence intervals
- token similarity heatmaps
- grouped CSV and JSON summaries

## Main Rebuttal Takeaway

Use this folder when you need visuals that go beyond the all-correct subset and show how geometry differs across correct, wrong, mixed, and SIRCL-flipped samples.
