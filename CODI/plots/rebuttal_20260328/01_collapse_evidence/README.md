# 01 - Collapse Evidence

This folder packages the materials for rebuttal point 1:

> What direct evidence shows that SIRCL prevents harmful collapse rather than simply shrinking trajectories?

## Included Here

- `analyze_latent_collapse.py`
- `results_main/`
- `results_simcon_t16/`
- `results_codi_t16/`
- `results_gsmhard_fallback/`
- `report.md`

## What This Tool Measures

- effective rank
- random similarity / anisotropy
- trajectory diversity
- radius quantiles such as P50, P90, and P99

## Main Rebuttal Takeaway

Use `results_main/` for the clean reviewer-facing comparison on `simcon` vs `simcon_sircl` at `T=16`.

Use the extra result folders when you want to show that the same diagnostic tooling was also run on the wider rebuttal sweep.
