#!/usr/bin/env python3
"""
Analyze latent-token collapse from saved CODI inference artifacts.

This script is designed for rebuttal-style analysis on existing `latents.json`
and `trajectory_stats.json` files. It computes low-cost collapse indicators that
can be run directly on saved checkpoints / evaluation outputs:

1. Effective rank (Roy & Vetterli): exp(H(normalized singular values))
2. RandomSim / anisotropy: mean cosine similarity of distinct random normalized
   latent-token pairs, computed in closed form from the summed token directions
3. Trajectory diversity within each sample:
   - mean pairwise L2 distance
   - mean pairwise cosine distance
4. Radius-tail summary:
   - r_t = distance from each latent token to the sample center
   - report P50 / P90 / P99 of all r_t values

Bootstrap is performed at the sample level and reports mean +/- bootstrap SE.

If a run only has `trajectory_stats.json` but no `latents.json`, the script
still exports a partial summary from the aggregated trajectory statistics and
marks the run as incomplete.

Examples
--------
Analyze a direct run directory:
    python3 CODI/plots/analyze_latent_collapse.py \
        CODI/results/latent_sweep_gsm8k/latent_16/models/simcon/gsm8k/run_0 \
        CODI/results/latent_sweep_gsm8k/latent_16/models/simcon_sircl/gsm8k/run_0 \
        --labels simcon simcon_sircl \
        --output-dir CODI/plots/results/latent_collapse_latent16

Analyze a broader experiment root (all matching run_* dirs beneath it):
    python3 CODI/plots/analyze_latent_collapse.py \
        CODI/results/latent_sweep_gsm8k/latent_16/models \
        --datasets gsm8k \
        --output-dir CODI/plots/results/latent_collapse_scan
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
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
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
    raise SystemExit(
        "numpy is required for latent collapse analysis. "
        "Please source the CODI virtualenv first, for example: "
        "`source /data/yhao/baseline/.venv/bin/activate`."
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None

try:
    from color_config import BAR_EDGE_COLOR, COLOR_LIST, GRID_ALPHA
except Exception:  # pragma: no cover - fall back to safe defaults
    COLOR_LIST = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#fdd0a2"]
    BAR_EDGE_COLOR = "black"
    GRID_ALPHA = 0.5


EPS = 1e-12
DEFAULT_BOOTSTRAP_ITERS = 400
DEFAULT_MAX_SAMPLES = 200
DEFAULT_PERCENTILES = (50, 90, 99)
PLOT_METRICS = (
    "effective_rank",
    "randomsim",
    "trajectory_l2_mean",
    "trajectory_cosine_distance_mean",
    "radius_p90",
    "radius_p99",
)


@dataclass
class RunArtifacts:
    run_dir: Path
    latents_path: Optional[Path]
    trajectory_stats_path: Optional[Path]
    metrics_path: Optional[Path]
    dataset: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="+",
        help=(
            "Run directories, artifact files, or higher-level roots. "
            "If a directory does not directly contain latents.json / "
            "trajectory_stats.json, the script recursively discovers run_* dirs."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching the top-level targets. Auto labels are used otherwise.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset filter, e.g. --datasets gsm8k gsm-hard",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("CODI/plots/results/latent_collapse"),
        help="Directory for summary CSV/JSON and optional plots.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Maximum number of samples per run. Use 0 or a negative number for all samples.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERS,
        help="Number of sample-level bootstrap iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample subsampling and bootstrap.",
    )
    parser.add_argument(
        "--center-before-svd",
        action="store_true",
        help="Center each sample matrix before computing effective rank.",
    )
    parser.add_argument(
        "--drop-last-latent",
        action="store_true",
        help=(
            "Drop the last saved latent step before analysis. "
            "Useful when saved iterations are T+1 and you want metrics aligned with configured T."
        ),
    )
    parser.add_argument(
        "--space-type",
        choices=("euclidean", "hyperbolic"),
        default="euclidean",
        help="Geometry used to define sample centers and radii.",
    )
    parser.add_argument(
        "--curvature",
        type=float,
        default=-1.0,
        help="Curvature used when --space-type hyperbolic.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Disable matplotlib summary figures.",
    )
    return parser.parse_args()


def infer_label_from_run_dir(run_dir: Path) -> str:
    parts = run_dir.parts
    dataset = run_dir.parent.name
    run_name = run_dir.name

    if "models" in parts:
        idx = max(i for i, part in enumerate(parts) if part == "models")
        model = parts[idx + 1] if idx + 1 < len(parts) else run_dir.parent.parent.name
        prefix = None
        for part in reversed(parts[:idx]):
            if part.startswith("latent_"):
                prefix = part
                break
        if prefix:
            return f"{prefix}/{model}/{dataset}/{run_name}"
        return f"{model}/{dataset}/{run_name}"

    if len(parts) >= 4:
        return "/".join(parts[-4:])
    return str(run_dir)


def has_run_artifacts(path: Path) -> bool:
    return (path / "latents.json").exists() or (path / "trajectory_stats.json").exists()


def discover_run_dirs(target: Path, dataset_filter: Optional[set[str]]) -> list[Path]:
    path = target
    if path.is_file():
        path = path.parent

    if has_run_artifacts(path):
        run_dirs = [path]
    else:
        run_dirs = sorted(
            candidate
            for candidate in path.rglob("run_*")
            if candidate.is_dir() and has_run_artifacts(candidate)
        )

    if dataset_filter:
        filtered = []
        for run_dir in run_dirs:
            dataset = run_dir.parent.name
            if dataset in dataset_filter:
                filtered.append(run_dir)
        run_dirs = filtered

    return run_dirs


def resolve_artifacts(args: argparse.Namespace) -> list[RunArtifacts]:
    dataset_filter = set(args.datasets) if args.datasets else None
    labels = args.labels or []
    if labels and len(labels) != len(args.targets):
        raise ValueError("--labels must match the number of top-level targets.")

    artifacts: list[RunArtifacts] = []
    seen: set[Path] = set()

    for idx, target_text in enumerate(args.targets):
        target = Path(target_text)
        base_label = labels[idx] if idx < len(labels) else None
        run_dirs = discover_run_dirs(target, dataset_filter)
        if not run_dirs:
            raise FileNotFoundError(
                f"No run artifacts found under target: {target}. "
                "Expected latents.json or trajectory_stats.json."
            )

        multiple_runs = len(run_dirs) > 1
        for run_dir in run_dirs:
            resolved = run_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            dataset = run_dir.parent.name
            auto_label = infer_label_from_run_dir(run_dir)
            label = auto_label
            if base_label:
                if multiple_runs:
                    label = f"{base_label}/{dataset}/{run_dir.name}"
                else:
                    label = base_label

            artifacts.append(
                RunArtifacts(
                    run_dir=run_dir,
                    latents_path=(run_dir / "latents.json") if (run_dir / "latents.json").exists() else None,
                    trajectory_stats_path=(
                        run_dir / "trajectory_stats.json"
                    )
                    if (run_dir / "trajectory_stats.json").exists()
                    else None,
                    metrics_path=(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else None,
                    dataset=dataset,
                    label=label,
                )
            )

    return artifacts


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metrics(metrics_path: Optional[Path]) -> dict:
    if metrics_path is None:
        return {}
    try:
        return load_json(metrics_path)
    except Exception:
        return {}


def choose_sample_indices(total_samples: int, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples <= 0 or max_samples >= total_samples:
        return np.arange(total_samples, dtype=np.int64)
    return np.sort(rng.choice(total_samples, size=max_samples, replace=False))


def load_selected_latents(
    latents_path: Path,
    max_samples: int,
    rng: np.random.Generator,
    drop_last_latent: bool,
) -> tuple[np.ndarray, dict]:
    data = load_json(latents_path)
    latents_list = data["latents"]
    total_samples = len(latents_list)
    sample_indices = choose_sample_indices(total_samples, max_samples, rng)
    selected = np.asarray([latents_list[i] for i in sample_indices], dtype=np.float32)
    if drop_last_latent and selected.ndim == 3 and selected.shape[1] > 1:
        selected = selected[:, :-1, :]
    metadata = {
        "description": data.get("description"),
        "num_samples_total": total_samples,
        "num_iterations_total": int(data.get("num_iterations", selected.shape[1] if selected.size else 0)),
        "embedding_dim": int(data.get("embedding_dim", selected.shape[2] if selected.size else 0)),
        "sample_indices": sample_indices,
        "drop_last_latent": bool(drop_last_latent),
    }
    return selected, metadata


def safe_l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.clip(norms, EPS, None)
    return x / norms


def effective_rank(matrix: np.ndarray, center_before_svd: bool) -> float:
    x = matrix.astype(np.float64, copy=False)
    if center_before_svd:
        x = x - x.mean(axis=0, keepdims=True)
    # T is typically tiny (<= 32) while D is large, so using the Gram matrix
    # is much faster than a full SVD on the T x D latent matrix.
    gram = x @ x.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    singular_values = np.sqrt(eigenvalues[::-1])
    total = singular_values.sum()
    if total <= EPS:
        return 0.0
    probs = singular_values / total
    probs = probs[probs > EPS]
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def compute_effective_rank_array(latents: np.ndarray, center_before_svd: bool) -> np.ndarray:
    return np.asarray(
        [effective_rank(sample, center_before_svd=center_before_svd) for sample in latents],
        dtype=np.float64,
    )


def compute_pairwise_diversity(latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_samples, num_tokens, _ = latents.shape
    if num_tokens <= 1:
        zeros = np.zeros(num_samples, dtype=np.float64)
        return zeros, zeros

    tri = np.triu_indices(num_tokens, k=1)
    pairwise_l2 = np.empty(num_samples, dtype=np.float64)
    pairwise_cosine_distance = np.empty(num_samples, dtype=np.float64)

    normalized = safe_l2_normalize(latents.astype(np.float64, copy=False))
    for idx in range(num_samples):
        sample = latents[idx].astype(np.float64, copy=False)
        gram = sample @ sample.T
        sq_norms = np.clip(np.diag(gram), 0.0, None)
        l2_sq = np.clip(sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram, 0.0, None)
        l2 = np.sqrt(l2_sq)
        pairwise_l2[idx] = float(l2[tri].mean())

        cosine = normalized[idx] @ normalized[idx].T
        cosine = np.clip(cosine, -1.0, 1.0)
        pairwise_cosine_distance[idx] = float((1.0 - cosine[tri]).mean())

    return pairwise_l2, pairwise_cosine_distance


def compute_radii(
    latents: np.ndarray,
    space_type: str,
    curvature: float,
) -> np.ndarray:
    if latents.size == 0:
        return np.zeros((0, 0), dtype=np.float64)

    if space_type == "euclidean":
        centers = latents.mean(axis=1, keepdims=True)
        return np.linalg.norm(latents - centers, axis=-1).astype(np.float64)

    import torch

    from src.trajectory_consistency import TrajectoryConsistencyCore

    core = TrajectoryConsistencyCore(space_type=space_type, curvature=curvature)
    x = torch.from_numpy(latents.astype(np.float32, copy=False))
    _, dist = core.center_and_dist(x)
    return dist.detach().cpu().numpy().astype(np.float64)


def compute_randomsim(normalized_latents: np.ndarray) -> float:
    tokens = normalized_latents.reshape(-1, normalized_latents.shape[-1])
    num_tokens = tokens.shape[0]
    if num_tokens <= 1:
        return 0.0

    sum_vector = tokens.sum(axis=0)
    total_pairwise = float(np.dot(sum_vector, sum_vector) - num_tokens)
    return total_pairwise / float(num_tokens * (num_tokens - 1))


def bootstrap_mean(values: np.ndarray, bootstrap_indices: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean": float("nan"), "bootstrap_se": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}

    if bootstrap_indices.size == 0:
        point = float(values.mean())
        return {"mean": point, "bootstrap_se": float("nan"), "ci95_low": point, "ci95_high": point}

    boot = values[bootstrap_indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "bootstrap_se": float(boot.std(ddof=1) if boot.size > 1 else 0.0),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
    }


def bootstrap_quantile(radii: np.ndarray, bootstrap_indices: np.ndarray, percentile: float) -> dict:
    flat = radii.reshape(-1)
    if flat.size == 0:
        return {"value": float("nan"), "bootstrap_se": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}

    if bootstrap_indices.size == 0:
        point = float(np.percentile(flat, percentile))
        return {"value": point, "bootstrap_se": float("nan"), "ci95_low": point, "ci95_high": point}

    boot = np.empty(bootstrap_indices.shape[0], dtype=np.float64)
    for idx, sample_ids in enumerate(bootstrap_indices):
        boot[idx] = float(np.percentile(radii[sample_ids].reshape(-1), percentile))

    return {
        "value": float(np.percentile(flat, percentile)),
        "bootstrap_se": float(boot.std(ddof=1) if boot.size > 1 else 0.0),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
    }


def bootstrap_randomsim(normalized_latents: np.ndarray, bootstrap_indices: np.ndarray) -> dict:
    if normalized_latents.size == 0:
        return {"mean": float("nan"), "bootstrap_se": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}

    point_estimate = compute_randomsim(normalized_latents)
    if bootstrap_indices.size == 0:
        return {
            "mean": float(point_estimate),
            "bootstrap_se": float("nan"),
            "ci95_low": float(point_estimate),
            "ci95_high": float(point_estimate),
        }

    boot = np.empty(bootstrap_indices.shape[0], dtype=np.float64)
    for idx, sample_ids in enumerate(bootstrap_indices):
        boot[idx] = compute_randomsim(normalized_latents[sample_ids])

    return {
        "mean": float(point_estimate),
        "bootstrap_se": float(boot.std(ddof=1) if boot.size > 1 else 0.0),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
    }


def maybe_load_trajectory_fallback(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    try:
        data = load_json(path)
    except Exception:
        return {}

    fallback = {}
    for key in ("threshold", "radius_max", "radius_mean", "radius_mean_std", "violation_rate_mean", "violation_rate_max", "num_batches", "total_samples"):
        if key in data:
            fallback[f"fallback_{key}"] = data[key]
    return fallback


def build_bootstrap_indices(num_samples: int, bootstrap_iters: int, rng: np.random.Generator) -> np.ndarray:
    if num_samples <= 0 or bootstrap_iters <= 0:
        return np.zeros((0, 0), dtype=np.int64)
    return rng.integers(0, num_samples, size=(bootstrap_iters, num_samples), endpoint=False)


def analyze_run(
    artifacts: RunArtifacts,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict:
    metrics = load_metrics(artifacts.metrics_path)
    summary = {
        "label": artifacts.label,
        "dataset": artifacts.dataset,
        "run_dir": str(artifacts.run_dir),
        "latents_path": str(artifacts.latents_path) if artifacts.latents_path else "",
        "trajectory_stats_path": str(artifacts.trajectory_stats_path) if artifacts.trajectory_stats_path else "",
        "metrics_path": str(artifacts.metrics_path) if artifacts.metrics_path else "",
        "accuracy": metrics.get("accuracy"),
        "avg_output_tokens": metrics.get("avg_output_tokens"),
        "has_latents": bool(artifacts.latents_path),
    }
    summary.update(maybe_load_trajectory_fallback(artifacts.trajectory_stats_path))

    if artifacts.latents_path is None:
        summary["status"] = "trajectory_only"
        summary["note"] = "Missing latents.json; only fallback trajectory stats are available."
        return summary

    selected_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
    latents, metadata = load_selected_latents(
        artifacts.latents_path,
        max_samples=args.max_samples,
        rng=selected_rng,
        drop_last_latent=args.drop_last_latent,
    )
    latents = latents.astype(np.float64, copy=False)

    num_samples, num_tokens, embedding_dim = latents.shape
    bootstrap_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
    bootstrap_indices = build_bootstrap_indices(num_samples, args.bootstrap_iters, bootstrap_rng)
    normalized_latents = safe_l2_normalize(latents)

    erank = compute_effective_rank_array(latents, center_before_svd=args.center_before_svd)
    pairwise_l2, pairwise_cosine_distance = compute_pairwise_diversity(latents)
    radii = compute_radii(latents, space_type=args.space_type, curvature=args.curvature)
    randomsim_stats = bootstrap_randomsim(normalized_latents, bootstrap_indices)

    erank_stats = bootstrap_mean(erank, bootstrap_indices)
    erank_ratio_stats = bootstrap_mean(erank / max(num_tokens, 1), bootstrap_indices)
    l2_stats = bootstrap_mean(pairwise_l2, bootstrap_indices)
    cosine_stats = bootstrap_mean(pairwise_cosine_distance, bootstrap_indices)

    summary.update(
        {
            "status": "complete",
            "description": metadata.get("description"),
            "num_samples_total": metadata["num_samples_total"],
            "num_samples_selected": num_samples,
            "num_tokens": num_tokens,
            "embedding_dim": embedding_dim,
            "center_before_svd": args.center_before_svd,
            "drop_last_latent": args.drop_last_latent,
            "space_type": args.space_type,
            "curvature": args.curvature,
            "effective_rank": erank_stats["mean"],
            "effective_rank_se": erank_stats["bootstrap_se"],
            "effective_rank_ci95_low": erank_stats["ci95_low"],
            "effective_rank_ci95_high": erank_stats["ci95_high"],
            "effective_rank_over_tokens": erank_ratio_stats["mean"],
            "effective_rank_over_tokens_se": erank_ratio_stats["bootstrap_se"],
            "effective_rank_over_tokens_ci95_low": erank_ratio_stats["ci95_low"],
            "effective_rank_over_tokens_ci95_high": erank_ratio_stats["ci95_high"],
            "randomsim": randomsim_stats["mean"],
            "randomsim_se": randomsim_stats["bootstrap_se"],
            "randomsim_ci95_low": randomsim_stats["ci95_low"],
            "randomsim_ci95_high": randomsim_stats["ci95_high"],
            "trajectory_l2_mean": l2_stats["mean"],
            "trajectory_l2_mean_se": l2_stats["bootstrap_se"],
            "trajectory_l2_mean_ci95_low": l2_stats["ci95_low"],
            "trajectory_l2_mean_ci95_high": l2_stats["ci95_high"],
            "trajectory_cosine_distance_mean": cosine_stats["mean"],
            "trajectory_cosine_distance_mean_se": cosine_stats["bootstrap_se"],
            "trajectory_cosine_distance_mean_ci95_low": cosine_stats["ci95_low"],
            "trajectory_cosine_distance_mean_ci95_high": cosine_stats["ci95_high"],
            "radius_mean": float(radii.mean()),
            "radius_std": float(radii.std(ddof=1)) if radii.size > 1 else 0.0,
            "radius_max": float(radii.max()) if radii.size else float("nan"),
        }
    )

    for percentile in DEFAULT_PERCENTILES:
        stats = bootstrap_quantile(radii, bootstrap_indices, percentile=percentile)
        key = f"radius_p{percentile}"
        summary[key] = stats["value"]
        summary[f"{key}_se"] = stats["bootstrap_se"]
        summary[f"{key}_ci95_low"] = stats["ci95_low"]
        summary[f"{key}_ci95_high"] = stats["ci95_high"]

    return summary


def write_summary_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_json(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def format_mean_se(mean: Optional[float], se: Optional[float], digits: int = 4) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "NA"
    if se is None or (isinstance(se, float) and math.isnan(se)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {se:.{digits}f}"


def print_console_summary(rows: list[dict]) -> None:
    print("=" * 120)
    print("Latent Collapse Summary")
    print("=" * 120)
    for row in rows:
        print(f"- {row['label']} [{row['dataset']}]")
        print(f"  status: {row.get('status', 'unknown')}")
        if row.get("accuracy") is not None:
            print(f"  accuracy: {row['accuracy']:.4f}")
        if row.get("status") != "complete":
            note = row.get("note")
            if note:
                print(f"  note: {note}")
            if row.get("fallback_radius_mean") is not None:
                print(
                    "  fallback radius mean/max/viol: "
                    f"{row.get('fallback_radius_mean'):.4f} / "
                    f"{row.get('fallback_radius_max'):.4f} / "
                    f"{row.get('fallback_violation_rate_mean'):.4f}"
                )
            continue

        print(f"  samples selected/total: {row['num_samples_selected']} / {row['num_samples_total']}")
        print(f"  latent tokens: {row['num_tokens']}")
        print(f"  effective rank: {format_mean_se(row['effective_rank'], row['effective_rank_se'])}")
        print(f"  randomsim: {format_mean_se(row['randomsim'], row['randomsim_se'])}")
        print(
            "  trajectory diversity (L2 / cosine distance): "
            f"{format_mean_se(row['trajectory_l2_mean'], row['trajectory_l2_mean_se'])} / "
            f"{format_mean_se(row['trajectory_cosine_distance_mean'], row['trajectory_cosine_distance_mean_se'])}"
        )
        print(
            "  radius quantiles p50 / p90 / p99: "
            f"{format_mean_se(row['radius_p50'], row['radius_p50_se'])} / "
            f"{format_mean_se(row['radius_p90'], row['radius_p90_se'])} / "
            f"{format_mean_se(row['radius_p99'], row['radius_p99_se'])}"
        )
    print("=" * 120)


def plot_metric_bars(dataset_rows: list[dict], output_path: Path) -> None:
    if plt is None or not dataset_rows:
        return

    complete_rows = [row for row in dataset_rows if row.get("status") == "complete"]
    if not complete_rows:
        return

    labels = [row["label"] for row in complete_rows]
    x = np.arange(len(labels))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#FFFFFF")
    axes = axes.ravel()

    for ax, metric in zip(axes, PLOT_METRICS):
        values = [row.get(metric, float("nan")) for row in complete_rows]
        se_values = [row.get(f"{metric}_se", float("nan")) for row in complete_rows]
        colors = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(len(values))]
        ax.bar(
            x,
            values,
            color=colors,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.0,
            alpha=0.9,
            yerr=se_values,
            capsize=4,
        )
        ax.set_title(metric.replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Latent Collapse Summary: {complete_rows[0]['dataset']}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def generate_plots(rows: list[dict], output_dir: Path) -> None:
    if plt is None:
        print("[Warn] matplotlib is unavailable; skip plots.")
        return

    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        plot_metric_bars(dataset_rows, output_dir / f"{dataset}_collapse_summary.png")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(args.seed)
    artifacts = resolve_artifacts(args)

    rows = [analyze_run(artifact, args, master_rng) for artifact in artifacts]
    rows.sort(key=lambda row: (row.get("dataset", ""), row.get("label", "")))

    write_summary_csv(rows, args.output_dir / "latent_collapse_summary.csv")
    write_summary_json(rows, args.output_dir / "latent_collapse_summary.json")
    if not args.skip_plots:
        generate_plots(rows, args.output_dir)
    print_console_summary(rows)

    print(f"Saved CSV:  {args.output_dir / 'latent_collapse_summary.csv'}")
    print(f"Saved JSON: {args.output_dir / 'latent_collapse_summary.json'}")
    if not args.skip_plots:
        print(f"Saved plots under: {args.output_dir}")


if __name__ == "__main__":
    main()
