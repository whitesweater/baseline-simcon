# CODI Plot And Analysis Scripts

## Main Layout

- `color_config.py` - shared plotting colors
- `plot_ablation.py`, `plot_gsm8k_comparison.py`, `plot_latent_sweep.py` - paper and sweep figures
- `analyze_latent_collapse.py` - collapse diagnostics added for rebuttal
- `analyze_centroid_reference.py` - centroid-reference probing and offline intervention
- `analyze_sircl_failure_modes.py` - baseline-wrong vs recovered/regressed boundary analysis
- `analyze_trajectory_by_correctness.py` - correctness-stratified trajectory geometry
- `plot_scaling_stability.py` - matched no-SIRCL vs +SIRCL scaling summary
- `results/` - generated figures, CSV, JSON, and markdown reports
- `rebuttal_20260328/` - single landing zone for the five rebuttal-only tools and their outputs

## Rebuttal Workspace

If you only care about the rebuttal materials, start here:

- `rebuttal_20260328/README.md`

That workspace groups the five new rebuttal analysis tools together with:

- the exact script used
- the corresponding result directory or directories
- the closest report or interpretation note
- a short local README for each rebuttal point

## Shared Color Config

The plotting scripts reuse the same palette from `color_config.py`.

Example:

```python
from color_config import COLOR_LIST, COLORS, LINE_COLOR

plt.bar(x, y, color=COLOR_LIST[0])
plt.plot(x, y, color=COLORS["purple"])
```

## Typical Usage

```bash
cd CODI/plots
python plot_ablation.py
```

```bash
cd CODI/plots
python plot_scaling_stability.py --preset latent_sweep_simcon --output-dir results/rebuttal_scaling_simcon_20260328
```
