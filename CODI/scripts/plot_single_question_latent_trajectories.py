#!/usr/bin/env python3
"""Plot single-question latent-token PCA trajectories for GSM8K.

This is the small, stable version of the trajectory plots in
``CODI/analyze_latent_visualization.py``.  It focuses on the figure style
needed for method comparison:

- SimCoT vs SimCoT+SIRCL
- CODI vs CODI+SIRCL

For each selected all-four-correct GSM8K sample it writes paired comparison
plots and a four-panel per-method plot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA as SklearnPCA
except Exception:  # pragma: no cover - fallback for minimal environments
    SklearnPCA = None


SCRIPT_DIR = Path(__file__).resolve().parent
CODI_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CODI_ROOT.parent
DEFAULT_STAGE_ROOT = (
    REPO_ROOT
    / "CODI_rebuttal_runs"
    / "rebuttal_20260325"
    / "multimodel_gsm8k_math500_aime_v1"
)
DEFAULT_RERUN_ROOT = DEFAULT_STAGE_ROOT / "results" / "latent_pca_rerun_latest"
DEFAULT_OUTPUT_DIR = CODI_ROOT / "plots" / "latent_token_pca_trajectories"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from plot_colors import MODEL_COLORS
except Exception:  # pragma: no cover
    MODEL_COLORS = {}


METHOD_ORDER = ("simcon", "sircl", "codi", "codi_sircl")
DISPLAY_NAMES = {
    "simcon": "SimCoT",
    "sircl": "SimCoT+SIRCL",
    "codi": "CODI",
    "codi_sircl": "CODI+SIRCL",
}
ALIASES = {
    "simcot": "simcon",
    "simcon": "simcon",
    "simcot_sircl": "sircl",
    "simcot+sircl": "sircl",
    "simcon_sircl": "sircl",
    "sircl": "sircl",
    "codi": "codi",
    "codi_sircl": "codi_sircl",
    "codi+sircl": "codi_sircl",
}
METHOD_COLORS = {
    "simcon": MODEL_COLORS.get("simcon", "#3498DB"),
    "sircl": MODEL_COLORS.get("simcon_sircl", "#1F618D"),
    "codi": MODEL_COLORS.get("codi", "#F39C12"),
    "codi_sircl": MODEL_COLORS.get("codi_sircl", "#D35400"),
}
METHOD_MARKERS = {
    "simcon": "o",
    "sircl": "s",
    "codi": "^",
    "codi_sircl": "D",
}
PAIR_ORDER = (
    ("simcon", "sircl", "SimCoT vs SimCoT+SIRCL"),
    ("codi", "codi_sircl", "CODI vs CODI+SIRCL"),
)


@dataclass
class RunData:
    method: str
    run_dir: Path
    predictions_path: Path
    latents_path: Path
    predictions: list[str]
    ground_truth: list[str]
    latents: np.ndarray
    correct_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun-root",
        type=Path,
        default=DEFAULT_RERUN_ROOT,
        help=(
            "Root containing method/checkpoint-*/models/*/gsm8k/run_0 outputs. "
            "Ignored for methods supplied with --run."
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "Explicit mapping METHOD=RUN_DIR. METHOD accepts simcon/simcot, "
            "sircl/simcot_sircl, codi, codi_sircl. Can be repeated."
        ),
    )
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoint-40000")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-indices",
        default="",
        help="Comma-separated GSM8K sample indices. If set, only all-correct entries are plotted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for trajectory figures and manifest.",
    )
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=None,
        help="Optional normalized layout root: models/<method>/gsm8k/run_0.",
    )
    parser.add_argument(
        "--copy-layout",
        action="store_true",
        help="Copy predictions/latents into --layout-dir instead of symlinking.",
    )
    return parser.parse_args()


def canonical_method(raw: str) -> str:
    key = raw.strip().lower().replace("-", "_")
    key = key.replace("+", "_")
    if key in ALIASES:
        return ALIASES[key]
    raise SystemExit(f"Unknown method alias: {raw}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_answer(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except Exception:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return f"{number:.12g}"


def compute_correct_mask(predictions: Iterable[object], ground_truth: Iterable[object]) -> np.ndarray:
    return np.asarray(
        [normalize_answer(pred) == normalize_answer(gold) for pred, gold in zip(predictions, ground_truth)],
        dtype=bool,
    )


def parse_explicit_runs(specs: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--run must be METHOD=RUN_DIR, got: {spec}")
        method_raw, path_raw = spec.split("=", 1)
        method = canonical_method(method_raw)
        path = Path(path_raw).expanduser()
        if path.name == "latents.json":
            path = path.parent
        if not (path / "latents.json").exists():
            raise SystemExit(f"Missing latents.json for {method}: {path}")
        if not (path / "predictions.json").exists():
            raise SystemExit(f"Missing predictions.json for {method}: {path}")
        runs[method] = path
    return runs


def candidate_run_dirs(rerun_root: Path, method: str, checkpoint: str, dataset: str, run_id: int) -> list[Path]:
    patterns = [
        rerun_root / method / checkpoint / "models" / "*" / dataset / f"run_{run_id}",
        rerun_root / "models" / method / dataset / f"run_{run_id}",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in pattern.parent.glob(pattern.name) if path.is_dir())
    if not matches and rerun_root.exists():
        for latents_path in rerun_root.rglob("latents.json"):
            run_dir = latents_path.parent
            rel_parts = run_dir.relative_to(rerun_root).parts
            if rel_parts and rel_parts[0] == method and dataset in rel_parts:
                matches.append(run_dir)
    usable = [path for path in matches if (path / "latents.json").exists() and (path / "predictions.json").exists()]
    return sorted(set(usable), key=lambda item: str(item))


def resolve_run_dirs(args: argparse.Namespace) -> dict[str, Path]:
    explicit = parse_explicit_runs(args.run)
    resolved: dict[str, Path] = dict(explicit)
    for method in METHOD_ORDER:
        if method in resolved:
            continue
        candidates = candidate_run_dirs(args.rerun_root, method, args.checkpoint, args.dataset, args.run_id)
        if candidates:
            resolved[method] = candidates[-1]
    missing = [DISPLAY_NAMES[m] for m in METHOD_ORDER if m not in resolved]
    if missing:
        raise SystemExit(
            "Missing complete run dirs for: "
            + ", ".join(missing)
            + f"\nLooked under: {args.rerun_root}"
        )
    return resolved


def load_run(method: str, run_dir: Path) -> RunData:
    predictions_path = run_dir / "predictions.json"
    latents_path = run_dir / "latents.json"
    pred_data = load_json(predictions_path)
    latent_data = load_json(latents_path)
    predictions = pred_data["predictions"]
    ground_truth = pred_data.get("ground_truth") or pred_data.get("ground_truths")
    if ground_truth is None:
        raise ValueError(f"No ground truth key in {predictions_path}")
    latents = np.asarray(latent_data["latents"], dtype=np.float32)
    if latents.ndim != 3:
        raise ValueError(f"Expected [samples, tokens, dim] latents in {latents_path}, got {latents.shape}")
    usable = min(len(predictions), len(ground_truth), latents.shape[0])
    predictions = predictions[:usable]
    ground_truth = ground_truth[:usable]
    latents = latents[:usable]
    return RunData(
        method=method,
        run_dir=run_dir,
        predictions_path=predictions_path,
        latents_path=latents_path,
        predictions=predictions,
        ground_truth=ground_truth,
        latents=latents,
        correct_mask=compute_correct_mask(predictions, ground_truth),
    )


def pca_project(matrix: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if SklearnPCA is not None:
        pca = SklearnPCA(n_components=n_components)
        coords = pca.fit_transform(matrix)
        explained = pca.explained_variance_ratio_
        return coords.astype(np.float32), explained.astype(np.float32)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T
    coords = centered @ components
    denom = np.sum(singular_values**2)
    explained = (singular_values[:n_components] ** 2 / denom) if denom > 0 else np.zeros(n_components)
    return coords.astype(np.float32), explained.astype(np.float32)


def select_samples(all_correct_indices: np.ndarray, args: argparse.Namespace) -> list[int]:
    if args.sample_indices.strip():
        wanted = [int(item.strip()) for item in args.sample_indices.split(",") if item.strip()]
        allowed = set(int(idx) for idx in all_correct_indices)
        selected = [idx for idx in wanted if idx in allowed]
        if not selected:
            raise SystemExit("None of --sample-indices are all-four-correct samples.")
        return selected
    count = min(args.num_samples, len(all_correct_indices))
    if count <= 0:
        raise SystemExit("No all-four-correct samples available to plot.")
    rng = np.random.default_rng(args.seed)
    return sorted(rng.choice(all_correct_indices, size=count, replace=False).astype(int).tolist())


def path_metrics(latents: np.ndarray) -> tuple[float, float]:
    path_length = float(np.linalg.norm(latents[1:] - latents[:-1], axis=1).sum())
    dist_to_final = float(np.linalg.norm(latents[:-1] - latents[-1], axis=1).mean())
    return path_length, dist_to_final


def plot_one_trajectory(ax: plt.Axes, points: np.ndarray, method: str, latents: np.ndarray, label_suffix: str = "") -> None:
    color = METHOD_COLORS[method]
    marker = METHOD_MARKERS[method]
    label = DISPLAY_NAMES[method] + label_suffix
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2.4, alpha=0.88, label=label)
    if len(points) > 1:
        deltas = points[1:] - points[:-1]
        ax.quiver(
            points[:-1, 0],
            points[:-1, 1],
            deltas[:, 0],
            deltas[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            width=0.004,
            alpha=0.45,
        )
    ax.scatter(
        points[:-1, 0],
        points[:-1, 1],
        s=70,
        marker=marker,
        color=color,
        edgecolors="white",
        linewidths=1.2,
        zorder=4,
    )
    ax.scatter(
        points[-1, 0],
        points[-1, 1],
        s=180,
        marker="*",
        color=color,
        edgecolors="#222222",
        linewidths=0.9,
        zorder=5,
    )
    for step, point in enumerate(points, start=1):
        ax.annotate(
            str(step),
            (point[0], point[1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
        )
    path_length, dist_to_final = path_metrics(latents)
    ax.text(
        0.02,
        0.03,
        f"path={path_length:.2f}\nmean dist to final={dist_to_final:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
        ha="left",
        va="bottom",
    )


def decorate_axis(ax: plt.Axes, explained: np.ndarray, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", color="#222222")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.axhline(0, color="#BBBBBB", linewidth=0.7)
    ax.axvline(0, color="#BBBBBB", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_shared_limits(axes: Iterable[plt.Axes], point_groups: Iterable[np.ndarray]) -> None:
    all_points = np.vstack(list(point_groups))
    x_pad = max(1e-6, float(np.ptp(all_points[:, 0])) * 0.12)
    y_pad = max(1e-6, float(np.ptp(all_points[:, 1])) * 0.12)
    xlim = (float(all_points[:, 0].min() - x_pad), float(all_points[:, 0].max() + x_pad))
    ylim = (float(all_points[:, 1].min() - y_pad), float(all_points[:, 1].max() + y_pad))
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


def plot_pair_comparison(sample_idx: int, runs: dict[str, RunData], output_dir: Path) -> tuple[Path, Path]:
    pair_dir = output_dir / "pair_comparisons"
    pair_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor="#FFFFFF")
    relative_fig, relative_axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor="#FFFFFF")

    for ax, rel_ax, (base_method, sircl_method, title) in zip(axes, relative_axes, PAIR_ORDER):
        base_latents = runs[base_method].latents[sample_idx]
        sircl_latents = runs[sircl_method].latents[sample_idx]
        combined = np.vstack([base_latents, sircl_latents])
        coords, explained = pca_project(combined, n_components=2)
        steps = base_latents.shape[0]
        base_points = coords[:steps]
        sircl_points = coords[steps:]

        plot_one_trajectory(ax, base_points, base_method, base_latents)
        plot_one_trajectory(ax, sircl_points, sircl_method, sircl_latents)
        decorate_axis(ax, explained, title)
        ax.legend(loc="best", fontsize=9, frameon=True)
        set_shared_limits([ax], [base_points, sircl_points])

        base_rel = base_points - base_points[-1]
        sircl_rel = sircl_points - sircl_points[-1]
        plot_one_trajectory(rel_ax, base_rel, base_method, base_latents)
        plot_one_trajectory(rel_ax, sircl_rel, sircl_method, sircl_latents)
        decorate_axis(rel_ax, explained, title + " (centered on final token)")
        rel_ax.scatter([0], [0], s=60, marker="x", color="#222222", zorder=6)
        rel_ax.legend(loc="best", fontsize=9, frameon=True)
        set_shared_limits([rel_ax], [base_rel, sircl_rel])

    fig.suptitle(f"GSM8K sample #{sample_idx}: latent token PCA trajectory", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = pair_dir / f"trajectory_q{sample_idx}_pca_pairs.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)

    relative_fig.suptitle(
        f"GSM8K sample #{sample_idx}: relative latent trajectory to final token",
        fontsize=14,
        fontweight="bold",
    )
    relative_fig.tight_layout(rect=(0, 0, 1, 0.94))
    rel_path = pair_dir / f"trajectory_q{sample_idx}_pca_pairs_relative.png"
    relative_fig.savefig(rel_path, dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(relative_fig)

    return out_path, rel_path


def plot_individual_panels(sample_idx: int, runs: dict[str, RunData], output_dir: Path) -> Path:
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    latents_by_method = {method: runs[method].latents[sample_idx] for method in METHOD_ORDER}
    combined = np.vstack([latents_by_method[method] for method in METHOD_ORDER])
    coords, explained = pca_project(combined, n_components=2)
    steps = latents_by_method[METHOD_ORDER[0]].shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2), facecolor="#FFFFFF")
    axes_flat = axes.flatten()
    point_groups = []
    for method_idx, method in enumerate(METHOD_ORDER):
        points = coords[method_idx * steps : (method_idx + 1) * steps]
        point_groups.append(points)
        ax = axes_flat[method_idx]
        plot_one_trajectory(ax, points, method, latents_by_method[method])
        decorate_axis(ax, explained, DISPLAY_NAMES[method])
    set_shared_limits(axes_flat, point_groups)
    fig.suptitle(
        f"GSM8K sample #{sample_idx}: four-method PCA fit on all latent tokens",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = individual_dir / f"trajectory_q{sample_idx}_pca_separate.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    return out_path


def write_layout(layout_dir: Path, runs: dict[str, RunData], copy_files: bool) -> None:
    import shutil

    layout_dir.mkdir(parents=True, exist_ok=True)
    for method, run in runs.items():
        target_dir = layout_dir / "models" / method / "gsm8k" / "run_0"
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename, source in {
            "predictions.json": run.predictions_path,
            "latents.json": run.latents_path,
        }.items():
            target = target_dir / filename
            if target.exists() or target.is_symlink():
                target.unlink()
            if copy_files:
                shutil.copy2(source, target)
            else:
                target.symlink_to(source.resolve())


def write_manifest(
    output_dir: Path,
    runs: dict[str, RunData],
    all_correct_indices: np.ndarray,
    selected_samples: list[int],
    figure_paths: list[Path],
) -> Path:
    manifest = {
        "methods": {
            method: {
                "display_name": DISPLAY_NAMES[method],
                "run_dir": str(run.run_dir),
                "predictions_path": str(run.predictions_path),
                "latents_path": str(run.latents_path),
                "latents_shape": list(run.latents.shape),
                "correct": int(run.correct_mask.sum()),
                "total": int(len(run.correct_mask)),
            }
            for method, run in runs.items()
        },
        "all_four_correct_count": int(len(all_correct_indices)),
        "selected_samples": selected_samples,
        "figures": [str(path) for path in figure_paths],
    }
    path = output_dir / "single_question_latent_trajectory_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    args = parse_args()
    run_dirs = resolve_run_dirs(args)
    runs = {method: load_run(method, run_dirs[method]) for method in METHOD_ORDER}

    min_len = min(len(run.correct_mask) for run in runs.values())
    all_correct = np.ones(min_len, dtype=bool)
    for run in runs.values():
        all_correct &= run.correct_mask[:min_len]
    all_correct_indices = np.where(all_correct)[0]
    selected_samples = select_samples(all_correct_indices, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.layout_dir is not None:
        write_layout(args.layout_dir, runs, copy_files=args.copy_layout)

    figure_paths: list[Path] = []
    for sample_idx in selected_samples:
        pair_path, relative_path = plot_pair_comparison(sample_idx, runs, args.output_dir)
        individual_path = plot_individual_panels(sample_idx, runs, args.output_dir)
        figure_paths.extend([pair_path, relative_path, individual_path])

    manifest_path = write_manifest(args.output_dir, runs, all_correct_indices, selected_samples, figure_paths)
    print(f"[Loaded] methods: {', '.join(DISPLAY_NAMES[m] for m in METHOD_ORDER)}")
    for method in METHOD_ORDER:
        run = runs[method]
        print(
            f"[Stats] {DISPLAY_NAMES[method]} correct={int(run.correct_mask.sum())}/{len(run.correct_mask)} "
            f"latents={tuple(run.latents.shape)}"
        )
    print(f"[Stats] all-four-correct={len(all_correct_indices)}")
    print(f"[Saved] figures={len(figure_paths)} output_dir={args.output_dir}")
    print(f"[Saved] manifest={manifest_path}")


if __name__ == "__main__":
    main()
