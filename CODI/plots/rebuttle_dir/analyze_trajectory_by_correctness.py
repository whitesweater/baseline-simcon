#!/usr/bin/env python3
"""
Trajectory geometry analysis stratified by correctness and difficulty.

Addresses the reviewer question:
  "Why only analyze all-correct samples? Can you show that SIRCL changes
   trajectories at critical points (failure cases, mixed difficulty)?"

Produces three families of figures:
  1. PCA / UMAP trajectory projections — correct vs wrong, side-by-side
  2. Per-step r_t curves — mean ± 95 % bootstrap CI, per group
  3. Token pairwise cosine-similarity heatmaps — per group (collapse check)

Plus a CSV / JSON summary of per-group statistics.

Usage examples:
  # Compare simcon vs simcon_sircl at latent_16:
  python3 CODI/plots/analyze_trajectory_by_correctness.py \
      CODI/results/latent_sweep_gsm8k/latent_16/models/simcon/gsm8k/run_0 \
      CODI/results/latent_sweep_gsm8k/latent_16/models/simcon_sircl/gsm8k/run_0 \
      --labels simcon simcon_sircl \
      --output-dir CODI/plots/results/trajectory_correctness

  # Auto-discover all models under a root:
  python3 CODI/plots/analyze_trajectory_by_correctness.py \
      CODI/results/latent_sweep_gsm8k/latent_16/models \
      --output-dir CODI/plots/results/trajectory_correctness_all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
CODI_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CODI_ROOT.parent
if str(CODI_ROOT) not in sys.path:
    sys.path.insert(0, str(CODI_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "numpy is required. Please activate the CODI virtualenv first."
    ) from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except Exception:
    plt = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

try:
    from umap import UMAP
except ImportError:
    UMAP = None

try:
    from color_config import BAR_EDGE_COLOR, COLOR_LIST, GRID_ALPHA
except Exception:
    COLOR_LIST = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#fdd0a2"]
    BAR_EDGE_COLOR = "black"
    GRID_ALPHA = 0.5


# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────
EPS = 1e-12
DEFAULT_BOOTSTRAP = 500
DEFAULT_MAX_SAMPLES = 0  # 0 = use all
SEED = 42

# Groups for analysis
GROUP_CORRECT = "correct"
GROUP_WRONG = "wrong"
GROUP_ALL_CORRECT = "all_correct"
GROUP_ALL_WRONG = "all_wrong"
GROUP_SIRCL_FLIPS = "sircl_flips"  # wrong→correct when adding SIRCL


# ────────────────────────────────────────────────────────────────────
# Data loading helpers
# ────────────────────────────────────────────────────────────────────
@dataclass
class RunData:
    label: str
    run_dir: Path
    latents: np.ndarray          # (S, T, D)
    predictions: list
    ground_truth: list
    correct_mask: np.ndarray     # (S,) bool
    num_samples: int
    num_iterations: int
    embedding_dim: int


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_correct(pred, gt) -> bool:
    """Check whether a prediction matches ground truth."""
    try:
        p = float(pred)
        g = float(gt)
        if abs(p - g) < 1e-6:
            return True
        if abs(g) > EPS and abs(p - g) / abs(g) < 1e-4:
            return True
        return False
    except (ValueError, TypeError):
        return str(pred).strip() == str(gt).strip()


def load_run(run_dir: Path, label: str, max_samples: int = 0) -> RunData:
    """Load latents and predictions from a single run directory."""
    latents_path = run_dir / "latents.json"
    predictions_path = run_dir / "predictions.json"

    if not latents_path.exists():
        raise FileNotFoundError(f"No latents.json in {run_dir}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"No predictions.json in {run_dir}")

    lat_data = load_json(latents_path)
    pred_data = load_json(predictions_path)

    preds = pred_data["predictions"]
    gts = pred_data["ground_truth"]
    n = min(len(preds), len(gts), len(lat_data["latents"]))

    # Subsample if requested
    if max_samples > 0 and max_samples < n:
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(n, size=max_samples, replace=False))
        latents_list = [lat_data["latents"][i] for i in idx]
        preds = [preds[i] for i in idx]
        gts = [gts[i] for i in idx]
    else:
        latents_list = lat_data["latents"][:n]
        preds = preds[:n]
        gts = gts[:n]

    latents = np.asarray(latents_list, dtype=np.float32)
    correct_mask = np.array([is_correct(p, g) for p, g in zip(preds, gts)])

    return RunData(
        label=label,
        run_dir=run_dir,
        latents=latents,
        predictions=preds,
        ground_truth=gts,
        correct_mask=correct_mask,
        num_samples=latents.shape[0],
        num_iterations=latents.shape[1],
        embedding_dim=latents.shape[2],
    )


# ────────────────────────────────────────────────────────────────────
# Sample grouping
# ────────────────────────────────────────────────────────────────────
@dataclass
class GroupedIndices:
    """Indices for different correctness groups."""
    per_model_correct: dict = field(default_factory=dict)   # label -> idx array
    per_model_wrong: dict = field(default_factory=dict)     # label -> idx array
    all_correct: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    all_wrong: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    mixed: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    sircl_flips: dict = field(default_factory=dict)  # (base, sircl) -> idx array  (wrong→correct)
    sircl_regress: dict = field(default_factory=dict)  # (base, sircl) -> idx array (correct→wrong)


def build_groups(runs: list[RunData]) -> GroupedIndices:
    """Build correctness-based groups from multiple runs."""
    g = GroupedIndices()
    n = runs[0].num_samples  # all runs should have same sample count

    # Per-model
    for run in runs:
        g.per_model_correct[run.label] = np.where(run.correct_mask)[0]
        g.per_model_wrong[run.label] = np.where(~run.correct_mask)[0]

    # Cross-model
    all_masks = np.stack([run.correct_mask for run in runs], axis=0)  # (M, S)
    g.all_correct = np.where(all_masks.all(axis=0))[0]
    g.all_wrong = np.where(~all_masks.any(axis=0))[0]
    g.mixed = np.where(
        ~all_masks.all(axis=0) & all_masks.any(axis=0)
    )[0]

    # SIRCL flip detection: find (base, sircl) pairs
    labels = [run.label for run in runs]
    for i, base in enumerate(runs):
        for j, sircl in enumerate(runs):
            if i == j:
                continue
            # Heuristic: sircl model label contains base label + "_sircl"
            if sircl.label.replace(base.label, "").strip("_/ ") in ("sircl",):
                flips = np.where(~base.correct_mask & sircl.correct_mask)[0]
                g.sircl_flips[(base.label, sircl.label)] = flips
                regress = np.where(base.correct_mask & ~sircl.correct_mask)[0]
                g.sircl_regress[(base.label, sircl.label)] = regress

    return g


# ────────────────────────────────────────────────────────────────────
# Metrics: per-step r_t
# ────────────────────────────────────────────────────────────────────
def compute_per_step_radii(latents: np.ndarray) -> np.ndarray:
    """
    Compute r_t for each step: distance from each latent token to the
    sample center (Euclidean).

    Args:
        latents: (S, T, D) array
    Returns:
        radii: (S, T) array of distances to per-sample mean
    """
    centers = latents.mean(axis=1, keepdims=True)  # (S, 1, D)
    return np.linalg.norm(latents - centers, axis=-1).astype(np.float64)


def bootstrap_per_step_ci(
    radii: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap mean ± 95% CI for per-step r_t.

    Args:
        radii: (S, T) array
    Returns:
        mean (T,), ci_low (T,), ci_high (T,)
    """
    S, T = radii.shape
    if S == 0:
        nan = np.full(T, np.nan)
        return nan, nan, nan

    rng = np.random.default_rng(seed)
    boot_means = np.empty((n_boot, T), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, S, size=S)
        boot_means[b] = radii[idx].mean(axis=0)

    mean = radii.mean(axis=0)
    ci_low = np.percentile(boot_means, 2.5, axis=0)
    ci_high = np.percentile(boot_means, 97.5, axis=0)
    return mean, ci_low, ci_high


# ────────────────────────────────────────────────────────────────────
# Metrics: pairwise cosine similarity heatmap
# ────────────────────────────────────────────────────────────────────
def compute_cosine_sim_matrix(latents: np.ndarray) -> np.ndarray:
    """
    Compute average pairwise cosine similarity between latent iterations,
    averaged across samples.

    Args:
        latents: (S, T, D) array
    Returns:
        sim_matrix: (T, T) array — mean cosine similarity between step i and j
    """
    S, T, D = latents.shape
    if S == 0:
        return np.full((T, T), np.nan)

    # Normalize each vector
    norms = np.linalg.norm(latents, axis=-1, keepdims=True)
    norms = np.clip(norms, EPS, None)
    normed = latents / norms  # (S, T, D)

    # Per-sample: (S, T, T) cosine similarity
    # Use batched matrix multiply: normed @ normed^T for each sample
    sim_matrices = np.einsum("std,srd->str", normed, normed)  # (S, T, T)
    return sim_matrices.mean(axis=0)  # (T, T)


# ────────────────────────────────────────────────────────────────────
# Visualization 1: PCA / UMAP trajectory projections
# ────────────────────────────────────────────────────────────────────
def plot_trajectory_projections(
    runs: list[RunData],
    groups: GroupedIndices,
    output_dir: Path,
    method: str = "pca",
    max_trajectories: int = 100,
):
    """Plot 2D trajectory projections for correct vs wrong samples, side by side."""
    if plt is None:
        print("[Warn] matplotlib unavailable; skipping projection plots.")
        return

    if method == "pca" and PCA is None:
        print("[Warn] sklearn not available; skipping PCA projection.")
        return
    if method == "umap" and UMAP is None:
        print("[Warn] umap not available; skipping UMAP projection.")
        return

    for run in runs:
        correct_idx = groups.per_model_correct[run.label]
        wrong_idx = groups.per_model_wrong[run.label]

        # Subsample for clarity
        rng = np.random.default_rng(SEED)
        if len(correct_idx) > max_trajectories:
            correct_idx = rng.choice(correct_idx, max_trajectories, replace=False)
        if len(wrong_idx) > max_trajectories:
            wrong_idx = rng.choice(wrong_idx, max_trajectories, replace=False)

        if len(correct_idx) == 0 and len(wrong_idx) == 0:
            continue

        # Collect all latent vectors for fitting the projection
        all_idx = np.concatenate([correct_idx, wrong_idx])
        all_latents = run.latents[all_idx]  # (N, T, D)
        N, T, D = all_latents.shape
        flat = all_latents.reshape(-1, D).astype(np.float32)

        if method == "pca":
            reducer = PCA(n_components=2, random_state=SEED)
        else:
            reducer = UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)

        coords_2d = reducer.fit_transform(flat)  # (N*T, 2)
        coords_2d = coords_2d.reshape(N, T, 2)

        n_correct = len(correct_idx)
        correct_coords = coords_2d[:n_correct]
        wrong_coords = coords_2d[n_correct:]

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#FFFFFF")
        method_label = method.upper()

        cmap = plt.cm.viridis
        norm = plt.Normalize(0, T - 1)

        for ax, coords, title, count in [
            (axes[0], correct_coords, f"Correct (n={len(correct_idx)})", len(correct_idx)),
            (axes[1], wrong_coords, f"Wrong (n={len(wrong_idx)})", len(wrong_idx)),
        ]:
            ax.set_facecolor("#FAFAFA")
            if count == 0:
                ax.text(0.5, 0.5, "No samples", ha="center", va="center",
                        fontsize=14, transform=ax.transAxes)
                ax.set_title(title, fontsize=13, fontweight="bold")
                continue

            # Draw trajectories as lines with color gradient
            for traj in coords:
                for t in range(T - 1):
                    ax.plot(
                        [traj[t, 0], traj[t + 1, 0]],
                        [traj[t, 1], traj[t + 1, 1]],
                        color=cmap(norm(t)),
                        alpha=0.25,
                        linewidth=0.8,
                    )

            # Draw start and end points
            starts = coords[:, 0, :]
            ends = coords[:, -1, :]
            ax.scatter(starts[:, 0], starts[:, 1], c="blue", s=12, alpha=0.5,
                       zorder=5, label=f"t=0", marker="o", edgecolors="none")
            ax.scatter(ends[:, 0], ends[:, 1], c="red", s=18, alpha=0.6,
                       zorder=5, label=f"t={T-1}", marker="*", edgecolors="none")

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel(f"{method_label}-1", fontsize=11)
            ax.set_ylabel(f"{method_label}-2", fontsize=11)

        fig.suptitle(
            f"Trajectory Projection ({method_label}) — {run.label}",
            fontsize=15, fontweight="bold",
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
        cbar.set_label("Iteration step", fontsize=11)

        fig.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.08)
        out_path = output_dir / f"trajectory_{method}_{run.label}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # --- Cross-model: all-correct vs all-wrong vs SIRCL-flips ---
    if len(runs) >= 2:
        _plot_cross_model_projection(runs, groups, output_dir, method, max_trajectories)


def _plot_cross_model_projection(
    runs: list[RunData],
    groups: GroupedIndices,
    output_dir: Path,
    method: str,
    max_trajectories: int,
):
    """Compare all-correct / all-wrong / SIRCL-flips in a single projection."""
    if plt is None:
        return

    # Use the first SIRCL pair if available, otherwise first two runs
    sircl_pairs = list(groups.sircl_flips.keys())
    if sircl_pairs:
        base_label, sircl_label = sircl_pairs[0]
        base_run = next(r for r in runs if r.label == base_label)
        sircl_run = next(r for r in runs if r.label == sircl_label)
    else:
        base_run = runs[0]
        sircl_run = runs[1] if len(runs) > 1 else runs[0]

    group_map = {
        "all_correct": groups.all_correct,
        "all_wrong": groups.all_wrong,
    }
    if sircl_pairs:
        group_map["sircl_flips"] = groups.sircl_flips[sircl_pairs[0]]
        regress = groups.sircl_regress.get(sircl_pairs[0], np.array([], dtype=int))
        if len(regress) > 0:
            group_map["sircl_regress"] = regress

    rng = np.random.default_rng(SEED)
    for key in group_map:
        idx = group_map[key]
        if len(idx) > max_trajectories:
            group_map[key] = rng.choice(idx, max_trajectories, replace=False)

    # For each run in the pair, plot the three groups
    for run in [base_run, sircl_run]:
        all_idx = np.concatenate(list(group_map.values()))
        if len(all_idx) == 0:
            continue
        all_idx_unique = np.unique(all_idx)

        lat = run.latents[all_idx_unique]
        N, T, D = lat.shape
        flat = lat.reshape(-1, D).astype(np.float32)

        if method == "pca":
            if PCA is None:
                continue
            reducer = PCA(n_components=2, random_state=SEED)
        else:
            if UMAP is None:
                continue
            reducer = UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)

        coords_2d = reducer.fit_transform(flat).reshape(N, T, 2)
        idx_to_pos = {int(idx): pos for pos, idx in enumerate(all_idx_unique)}

        n_groups = len(group_map)
        fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 7), facecolor="#FFFFFF")
        if n_groups == 1:
            axes = [axes]

        group_colors = {
            "all_correct": "#2ca02c",
            "all_wrong": "#d62728",
            "sircl_flips": "#ff7f0e",
            "sircl_regress": "#9467bd",
        }
        group_display = {
            "all_correct": "All Correct",
            "all_wrong": "All Wrong",
            "sircl_flips": "SIRCL Flips (W→C)",
            "sircl_regress": "SIRCL Regress (C→W)",
        }

        method_label = method.upper()
        for ax, (gname, gidx) in zip(axes, group_map.items()):
            ax.set_facecolor("#FAFAFA")
            color = group_colors.get(gname, "#333333")
            n_samples = len(gidx)

            if n_samples == 0:
                ax.text(0.5, 0.5, "No samples", ha="center", va="center",
                        fontsize=14, transform=ax.transAxes)
            else:
                for sample_idx in gidx:
                    pos = idx_to_pos.get(int(sample_idx))
                    if pos is None:
                        continue
                    traj = coords_2d[pos]
                    for t in range(T - 1):
                        ax.plot(
                            [traj[t, 0], traj[t + 1, 0]],
                            [traj[t, 1], traj[t + 1, 1]],
                            color=color, alpha=0.3, linewidth=0.8,
                        )
                    ax.scatter(traj[0, 0], traj[0, 1], c="blue", s=10, alpha=0.4,
                               zorder=5, marker="o", edgecolors="none")
                    ax.scatter(traj[-1, 0], traj[-1, 1], c=color, s=16, alpha=0.6,
                               zorder=5, marker="*", edgecolors="none")

            display = group_display.get(gname, gname)
            ax.set_title(f"{display} (n={n_samples})", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel(f"{method_label}-1", fontsize=11)
            ax.set_ylabel(f"{method_label}-2", fontsize=11)

        fig.suptitle(
            f"Cross-Model Groups ({method_label}) — {run.label}",
            fontsize=15, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = output_dir / f"trajectory_{method}_groups_{run.label}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ────────────────────────────────────────────────────────────────────
# Visualization 2: Per-step r_t curves
# ────────────────────────────────────────────────────────────────────
def plot_per_step_rt(
    runs: list[RunData],
    groups: GroupedIndices,
    output_dir: Path,
    n_boot: int = DEFAULT_BOOTSTRAP,
):
    """Plot per-step r_t curves with mean ± 95% CI, comparing correct vs wrong."""
    if plt is None:
        print("[Warn] matplotlib unavailable; skipping r_t plots.")
        return

    # --- Per-model: correct vs wrong ---
    for run in runs:
        radii = compute_per_step_radii(run.latents)  # (S, T)
        T = run.num_iterations

        correct_idx = groups.per_model_correct[run.label]
        wrong_idx = groups.per_model_wrong[run.label]

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        steps = np.arange(T)

        for idx_arr, label, color, ls in [
            (correct_idx, f"Correct (n={len(correct_idx)})", "#2ca02c", "-"),
            (wrong_idx, f"Wrong (n={len(wrong_idx)})", "#d62728", "--"),
        ]:
            if len(idx_arr) == 0:
                continue
            mean, ci_lo, ci_hi = bootstrap_per_step_ci(radii[idx_arr], n_boot)
            ax.plot(steps, mean, color=color, linestyle=ls, linewidth=2.2, label=label)
            ax.fill_between(steps, ci_lo, ci_hi, color=color, alpha=0.15)

        ax.set_xlabel("Iteration step t", fontsize=13, fontweight="bold")
        ax.set_ylabel(r"$r_t$ (distance to sample center)", fontsize=13, fontweight="bold")
        ax.set_title(f"Per-Step Radius — {run.label}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        out_path = output_dir / f"rt_curve_{run.label}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # --- Cross-model overlay: base vs SIRCL for each group ---
    sircl_pairs = list(groups.sircl_flips.keys())
    if not sircl_pairs:
        # Fall back to comparing first two runs
        if len(runs) >= 2:
            sircl_pairs = [(runs[0].label, runs[1].label)]

    for base_label, sircl_label in sircl_pairs:
        base_run = next((r for r in runs if r.label == base_label), None)
        sircl_run = next((r for r in runs if r.label == sircl_label), None)
        if base_run is None or sircl_run is None:
            continue

        T = base_run.num_iterations
        steps = np.arange(T)

        group_configs = [
            ("all_correct", groups.all_correct, "All Correct"),
            ("all_wrong", groups.all_wrong, "All Wrong"),
            ("mixed", groups.mixed, "Mixed (disagreement)"),
        ]
        flip_idx = groups.sircl_flips.get((base_label, sircl_label), np.array([], dtype=int))
        if len(flip_idx) > 0:
            group_configs.append(("sircl_flips", flip_idx, "SIRCL Flips (W→C)"))
        regress_idx = groups.sircl_regress.get((base_label, sircl_label), np.array([], dtype=int))
        if len(regress_idx) > 0:
            group_configs.append(("sircl_regress", regress_idx, "SIRCL Regress (C→W)"))

        n_panels = len(group_configs)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5), facecolor="#FFFFFF")
        if n_panels == 1:
            axes = [axes]

        for ax, (gname, gidx, gtitle) in zip(axes, group_configs):
            ax.set_facecolor("#FFFFFF")
            if len(gidx) == 0:
                ax.text(0.5, 0.5, "No samples", ha="center", va="center",
                        fontsize=12, transform=ax.transAxes)
                ax.set_title(f"{gtitle} (n=0)", fontsize=12, fontweight="bold")
                continue

            for run, color, ls in [
                (base_run, "#1f77b4", "-"),
                (sircl_run, "#ff7f0e", "--"),
            ]:
                radii = compute_per_step_radii(run.latents)
                mean, ci_lo, ci_hi = bootstrap_per_step_ci(radii[gidx], n_boot)
                ax.plot(steps, mean, color=color, linestyle=ls, linewidth=2.0,
                        label=run.label)
                ax.fill_between(steps, ci_lo, ci_hi, color=color, alpha=0.12)

            ax.set_title(f"{gtitle} (n={len(gidx)})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Step t", fontsize=11)
            ax.set_ylabel(r"$r_t$", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(
            f"Per-Step r_t — {base_label} vs {sircl_label}",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = output_dir / f"rt_curve_compare_{base_label}_vs_{sircl_label}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ────────────────────────────────────────────────────────────────────
# Visualization 3: Token cosine-similarity heatmaps
# ────────────────────────────────────────────────────────────────────
def plot_similarity_heatmaps(
    runs: list[RunData],
    groups: GroupedIndices,
    output_dir: Path,
):
    """Plot per-group cosine similarity heatmaps between latent iterations."""
    if plt is None:
        print("[Warn] matplotlib unavailable; skipping heatmap plots.")
        return

    for run in runs:
        correct_idx = groups.per_model_correct[run.label]
        wrong_idx = groups.per_model_wrong[run.label]

        subsets = [
            ("correct", correct_idx),
            ("wrong", wrong_idx),
        ]
        # Add cross-model groups if available
        if len(groups.all_correct) > 0:
            subsets.append(("all_correct", groups.all_correct))
        if len(groups.all_wrong) > 0:
            subsets.append(("all_wrong", groups.all_wrong))
        for (bl, sl), fidx in groups.sircl_flips.items():
            if run.label in (bl, sl) and len(fidx) > 0:
                subsets.append(("sircl_flips", fidx))
        for (bl, sl), ridx in groups.sircl_regress.items():
            if run.label in (bl, sl) and len(ridx) > 0:
                subsets.append(("sircl_regress", ridx))

        n_sub = len(subsets)
        fig, axes = plt.subplots(1, n_sub, figsize=(6 * n_sub, 5), facecolor="#FFFFFF")
        if n_sub == 1:
            axes = [axes]

        for ax, (sname, sidx) in zip(axes, subsets):
            if len(sidx) == 0:
                ax.text(0.5, 0.5, "No samples", ha="center", va="center",
                        fontsize=12, transform=ax.transAxes)
                ax.set_title(f"{sname} (n=0)", fontsize=12, fontweight="bold")
                continue

            sim = compute_cosine_sim_matrix(run.latents[sidx])
            T = sim.shape[0]

            im = ax.imshow(sim, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, aspect="equal")
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xlabel("Iteration j", fontsize=10)
            ax.set_ylabel("Iteration i", fontsize=10)
            ax.set_title(f"{sname} (n={len(sidx)})", fontsize=12, fontweight="bold")

            # Only label every few ticks if T is large
            if T > 10:
                tick_step = max(1, T // 8)
                ax.set_xticks(range(0, T, tick_step))
                ax.set_yticks(range(0, T, tick_step))

        fig.suptitle(
            f"Cosine Similarity Heatmap — {run.label}",
            fontsize=14, fontweight="bold",
        )

        # Add shared colorbar
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Cosine similarity")

        out_path = output_dir / f"sim_heatmap_{run.label}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ────────────────────────────────────────────────────────────────────
# Summary statistics
# ────────────────────────────────────────────────────────────────────
def compute_group_summary(
    runs: list[RunData],
    groups: GroupedIndices,
    n_boot: int = DEFAULT_BOOTSTRAP,
) -> list[dict]:
    """Compute summary statistics for each (run, group) combination."""
    rows = []

    for run in runs:
        radii = compute_per_step_radii(run.latents)  # (S, T)

        group_map = {
            "correct": groups.per_model_correct[run.label],
            "wrong": groups.per_model_wrong[run.label],
            "all_correct": groups.all_correct,
            "all_wrong": groups.all_wrong,
            "mixed": groups.mixed,
        }
        for (bl, sl), fidx in groups.sircl_flips.items():
            if run.label in (bl, sl):
                group_map["sircl_flips"] = fidx
        for (bl, sl), ridx in groups.sircl_regress.items():
            if run.label in (bl, sl):
                group_map["sircl_regress"] = ridx

        for gname, gidx in group_map.items():
            n = len(gidx)
            row = {
                "model": run.label,
                "group": gname,
                "n_samples": n,
            }

            if n == 0:
                row.update({
                    "rt_mean": float("nan"),
                    "rt_mean_ci_lo": float("nan"),
                    "rt_mean_ci_hi": float("nan"),
                    "rt_final_mean": float("nan"),
                    "cos_sim_consecutive_mean": float("nan"),
                    "cos_sim_all_pairs_mean": float("nan"),
                    "diversity_l2_mean": float("nan"),
                    "diversity_cosine_mean": float("nan"),
                })
                rows.append(row)
                continue

            sub_radii = radii[gidx]
            sub_latents = run.latents[gidx]

            # Overall r_t stats
            mean_rt = float(sub_radii.mean())
            boot_means = []
            rng = np.random.default_rng(SEED)
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boot_means.append(sub_radii[idx].mean())
            boot_means = np.array(boot_means)
            row["rt_mean"] = mean_rt
            row["rt_mean_ci_lo"] = float(np.percentile(boot_means, 2.5))
            row["rt_mean_ci_hi"] = float(np.percentile(boot_means, 97.5))

            # Final-step r_t
            row["rt_final_mean"] = float(sub_radii[:, -1].mean())

            # Cosine similarity: consecutive steps
            sim = compute_cosine_sim_matrix(sub_latents)
            T = sim.shape[0]
            consec = [sim[t, t + 1] for t in range(T - 1)]
            row["cos_sim_consecutive_mean"] = float(np.mean(consec))

            # Cosine similarity: all pairs
            tri = np.triu_indices(T, k=1)
            row["cos_sim_all_pairs_mean"] = float(sim[tri].mean())

            # Pairwise diversity
            norms = np.linalg.norm(sub_latents, axis=-1, keepdims=True)
            norms = np.clip(norms, EPS, None)
            normed = sub_latents / norms
            # mean L2 pairwise within each sample
            l2_divs = []
            cos_divs = []
            tri_t = np.triu_indices(T, k=1)
            for s in range(min(n, 200)):
                gram = sub_latents[s].astype(np.float64) @ sub_latents[s].astype(np.float64).T
                sq = np.diag(gram)
                l2_sq = np.clip(sq[:, None] + sq[None, :] - 2 * gram, 0, None)
                l2_divs.append(float(np.sqrt(l2_sq[tri_t]).mean()))

                cg = normed[s].astype(np.float64) @ normed[s].astype(np.float64).T
                cg = np.clip(cg, -1, 1)
                cos_divs.append(float((1 - cg[tri_t]).mean()))

            row["diversity_l2_mean"] = float(np.mean(l2_divs))
            row["diversity_cosine_mean"] = float(np.mean(cos_divs))

            rows.append(row)

    return rows


# ────────────────────────────────────────────────────────────────────
# I/O
# ────────────────────────────────────────────────────────────────────
def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_json(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def print_summary(rows: list[dict]):
    print("\n" + "=" * 100)
    print("Trajectory-by-Correctness Summary")
    print("=" * 100)
    for row in rows:
        n = row["n_samples"]
        rt = row.get("rt_mean", float("nan"))
        ci_lo = row.get("rt_mean_ci_lo", float("nan"))
        ci_hi = row.get("rt_mean_ci_hi", float("nan"))
        cos_c = row.get("cos_sim_consecutive_mean", float("nan"))
        div_l2 = row.get("diversity_l2_mean", float("nan"))

        rt_str = f"{rt:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]" if not math.isnan(rt) else "NA"
        cos_str = f"{cos_c:.4f}" if not math.isnan(cos_c) else "NA"
        div_str = f"{div_l2:.4f}" if not math.isnan(div_l2) else "NA"

        print(f"  {row['model']:30s}  {row['group']:16s}  n={n:5d}  "
              f"r_t={rt_str}  cos_consec={cos_str}  div_L2={div_str}")
    print("=" * 100)


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────
def has_run_artifacts(path: Path) -> bool:
    return (path / "latents.json").exists() and (path / "predictions.json").exists()


def discover_runs(target: Path) -> list[Path]:
    if has_run_artifacts(target):
        return [target]
    return sorted(
        c for c in target.rglob("run_*")
        if c.is_dir() and has_run_artifacts(c)
    )


def infer_label(run_dir: Path) -> str:
    parts = run_dir.parts
    if "models" in parts:
        idx = max(i for i, p in enumerate(parts) if p == "models")
        model = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
        return model
    return run_dir.parent.parent.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "targets", nargs="+",
        help="Run directories or higher-level roots containing run_* dirs.",
    )
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels matching targets. Auto-generated if omitted.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("CODI/plots/results/trajectory_correctness"),
                        help="Output directory for plots and summaries.")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help="Max samples per run (0 = all).")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP,
                        help="Number of bootstrap iterations for CIs.")
    parser.add_argument("--projection", choices=("pca", "umap", "both"), default="pca",
                        help="Projection method for trajectory plots.")
    parser.add_argument("--max-trajectories", type=int, default=100,
                        help="Max trajectories per group in projection plots.")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Only output CSV/JSON, no figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover and load runs
    labels = args.labels or []
    runs: list[RunData] = []
    seen = set()

    for i, target_str in enumerate(args.targets):
        target = Path(target_str)
        base_label = labels[i] if i < len(labels) else None
        run_dirs = discover_runs(target)
        if not run_dirs:
            print(f"[Warn] No run artifacts found under {target}")
            continue

        for rd in run_dirs:
            resolved = rd.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            label = base_label if (base_label and len(run_dirs) == 1) else infer_label(rd)
            print(f"Loading: {label} <- {rd}")
            runs.append(load_run(rd, label, max_samples=args.max_samples))

    if not runs:
        print("No runs loaded. Exiting.")
        return

    # Verify sample alignment
    n0 = runs[0].num_samples
    for run in runs[1:]:
        if run.num_samples != n0:
            print(f"[Warn] Sample count mismatch: {run.label} has {run.num_samples}, "
                  f"expected {n0}. Cross-model grouping may be incorrect.")

    # Build correctness groups
    print("\nBuilding correctness groups...")
    groups = build_groups(runs)
    print(f"  all_correct: {len(groups.all_correct)}")
    print(f"  all_wrong:   {len(groups.all_wrong)}")
    print(f"  mixed:       {len(groups.mixed)}")
    for (bl, sl), fidx in groups.sircl_flips.items():
        print(f"  sircl_flips ({bl} -> {sl}): {len(fidx)}")
    for (bl, sl), ridx in groups.sircl_regress.items():
        print(f"  sircl_regress ({bl} -> {sl}): {len(ridx)}")

    # Compute summary stats
    print("\nComputing per-group statistics...")
    summary_rows = compute_group_summary(runs, groups, n_boot=args.bootstrap)
    write_csv(summary_rows, args.output_dir / "trajectory_correctness_summary.csv")
    write_json(summary_rows, args.output_dir / "trajectory_correctness_summary.json")
    print_summary(summary_rows)

    if args.skip_plots:
        print("\nPlots skipped (--skip-plots).")
        return

    # Plot 1: Trajectory projections
    print("\nGenerating trajectory projections...")
    methods = ["pca", "umap"] if args.projection == "both" else [args.projection]
    for method in methods:
        plot_trajectory_projections(runs, groups, args.output_dir, method=method,
                                   max_trajectories=args.max_trajectories)

    # Plot 2: Per-step r_t curves
    print("\nGenerating per-step r_t curves...")
    plot_per_step_rt(runs, groups, args.output_dir, n_boot=args.bootstrap)

    # Plot 3: Cosine similarity heatmaps
    print("\nGenerating similarity heatmaps...")
    plot_similarity_heatmaps(runs, groups, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
