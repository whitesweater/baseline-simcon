#!/usr/bin/env python3
"""
面向 rebuttal 的 scaling/stability/latent-collapse 联合分析脚本。

目标：
1. 从现有推理产物里读取 accuracy / trajectory / latent token 信息
2. 可选地从 trainer_state.json 读取训练期 loss 波动指标
3. 按 T 和是否使用 SIRCL 聚合，输出 CSV / Markdown / 图表

默认支持两种工作流：

1) 直接复用已有 latent sweep（方便先验证脚本）
   python CODI/plots/plot_scaling_stability.py \
       --preset latent_sweep_simcon \
       --t-values 6 16 \
       --output-dir CODI/plots/results/rebuttal_scaling_simcon

2) 用 manifest 接入 rebuttal 期间新补的 no-SIRCL / +SIRCL 结果
   python CODI/plots/plot_scaling_stability.py \
       --manifest path/to/scaling_manifest.csv \
       --output-dir CODI/plots/results/rebuttal_scaling_final

manifest CSV 列建议如下：
    t,condition,family,label,run_dir,trainer_state

其中：
- t: int，实验对应的 T
- condition: no-SIRCL / +SIRCL
- family: 可选，例如 Sim-CoT / CODI
- label: 可选，图例展示名
- run_dir: 指向包含 metrics.json / latents.json / trajectory_stats.json 的 run_* 目录
- trainer_state: 可选，若提供则额外计算训练 loss 波动指标
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from color_config import BAR_EDGE_COLOR, COLOR_LIST, GRID_ALPHA


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "CODI" / "plots" / "results" / "scaling_stability"
PRESET_LATENT_SWEEP = REPO_ROOT / "CODI" / "results" / "latent_sweep_gsm8k"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV manifest for formal rebuttal runs.",
    )
    parser.add_argument(
        "--preset",
        choices=("latent_sweep_simcon", "latent_sweep_codi"),
        default=None,
        help="Quick-start preset over existing latent sweep results.",
    )
    parser.add_argument(
        "--t-values",
        type=int,
        nargs="*",
        default=None,
        help="Optional T filter, e.g. --t-values 6 16 32.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV / markdown / plots.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum latent samples to analyze per run; <=0 means all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for latent subsampling.",
    )
    parser.add_argument(
        "--keep-last-latent",
        action="store_true",
        help="By default the last latent is removed so num_iterations-1 aligns with T.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=100,
        help="Smoothing window for training-loss curves.",
    )
    args = parser.parse_args()

    if args.manifest is None and args.preset is None:
        parser.error("Please provide either --manifest or --preset.")

    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def infer_condition(text: str) -> str:
    lower = text.lower()
    return "+SIRCL" if "sircl" in lower else "no-SIRCL"


def infer_family(text: str) -> str:
    lower = text.lower()
    if "simcon" in lower:
        return "Sim-CoT"
    if "codi" in lower:
        return "CODI"
    if "decoder-trajectory" in lower or "euclidean" in lower:
        return "Trajectory"
    return "Unknown"


def infer_t(text: str) -> int | None:
    patterns = (
        r"latent[_-](\d+)",
        r"(\d+)long",
        r"(?:^|[_-])t(\d+)(?:$|[_-])",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def infer_label(spec: dict[str, Any]) -> str:
    family = spec.get("family") or infer_family(str(spec["run_dir"]))
    condition = spec.get("condition") or infer_condition(str(spec["run_dir"]))
    t_value = spec.get("t")
    if t_value is None:
        t_value = infer_t(str(spec["run_dir"]))
    if t_value is None:
        return f"{family} {condition}"
    return f"T={t_value} {condition}"


def build_specs_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("run_dir"):
                continue
            spec: dict[str, Any] = {
                "run_dir": resolve_path(row["run_dir"]),
                "trainer_state": resolve_path(row["trainer_state"]) if row.get("trainer_state") else None,
                "t": int(row["t"]) if row.get("t") else None,
                "condition": row.get("condition") or None,
                "family": row.get("family") or None,
                "label": row.get("label") or None,
            }
            specs.append(spec)
    return specs


def build_specs_from_preset(preset: str, t_values: set[int] | None) -> list[dict[str, Any]]:
    if preset == "latent_sweep_simcon":
        model_names = ["simcon", "simcon_sircl"]
        family = "Sim-CoT"
    elif preset == "latent_sweep_codi":
        model_names = ["codi", "codi_sircl"]
        family = "CODI"
    else:
        raise ValueError(f"Unknown preset: {preset}")

    specs: list[dict[str, Any]] = []
    for latent_dir in sorted(PRESET_LATENT_SWEEP.glob("latent_*")):
        if not latent_dir.is_dir():
            continue
        t_value = infer_t(latent_dir.name)
        if t_value is None:
            continue
        if t_values and t_value not in t_values:
            continue

        for model_name in model_names:
            run_dir = latent_dir / "models" / model_name / "gsm8k" / "run_0"
            if not run_dir.exists():
                continue
            condition = "+SIRCL" if "sircl" in model_name else "no-SIRCL"
            specs.append(
                {
                    "run_dir": run_dir.resolve(),
                    "trainer_state": None,
                    "t": t_value,
                    "condition": condition,
                    "family": family,
                    "label": f"T={t_value} {condition}",
                }
            )
    return specs


def ensure_run_spec(spec: dict[str, Any], remove_last_latent: bool) -> dict[str, Any]:
    run_dir = Path(spec["run_dir"])
    metrics_path = run_dir / "metrics.json"
    trajectory_path = run_dir / "trajectory_stats.json"
    latents_path = run_dir / "latents.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    if not latents_path.exists():
        raise FileNotFoundError(f"latents.json not found: {latents_path}")

    spec = dict(spec)
    spec["metrics_path"] = metrics_path
    spec["trajectory_path"] = trajectory_path if trajectory_path.exists() else None
    spec["latents_path"] = latents_path

    if spec.get("t") is None:
        t_value = infer_t(str(run_dir))
        if t_value is None:
            lat_meta = load_json(latents_path)
            inferred = int(lat_meta.get("num_iterations", 0))
            if remove_last_latent and inferred > 0:
                inferred -= 1
            t_value = inferred if inferred > 0 else None
        spec["t"] = t_value

    if spec.get("condition") is None:
        spec["condition"] = infer_condition(str(run_dir))
    if spec.get("family") is None:
        spec["family"] = infer_family(str(run_dir))
    if spec.get("label") is None:
        spec["label"] = infer_label(spec)

    return spec


def sample_latents(latents: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or latents.shape[0] <= max_samples:
        return latents
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(latents.shape[0], size=max_samples, replace=False))
    return latents[indices]


def effective_rank(matrix: np.ndarray) -> float:
    x = np.asarray(matrix, dtype=np.float64)
    # T is usually tiny compared with hidden dim, so use the smaller Gram matrix.
    gram = x @ x.T if x.shape[0] <= x.shape[1] else x.T @ x
    eigvals = np.linalg.eigvalsh(gram)
    singular_values = np.sqrt(np.clip(eigvals, a_min=0.0, a_max=None))
    total = singular_values.sum()
    if total <= 1e-12:
        return 0.0
    probs = singular_values / total
    probs = probs[probs > 1e-12]
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(np.exp(entropy))


def compute_randomsim(latents: np.ndarray) -> float:
    norms = np.linalg.norm(latents, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = latents / norms
    flat = normalized.reshape(-1, normalized.shape[-1])
    if flat.shape[0] <= 1:
        return 0.0
    sum_vector = flat.sum(axis=0)
    total_pairwise = float(np.dot(sum_vector, sum_vector) - flat.shape[0])
    return total_pairwise / float(flat.shape[0] * (flat.shape[0] - 1))


def compute_latent_metrics(
    latents_path: Path,
    max_samples: int,
    seed: int,
    remove_last_latent: bool,
) -> dict[str, float]:
    data = load_json(latents_path)
    latents = np.asarray(data["latents"], dtype=np.float64)
    latents = sample_latents(latents, max_samples=max_samples, seed=seed)

    if remove_last_latent and latents.shape[1] > 1:
        latents = latents[:, :-1, :]

    num_samples, num_tokens, embedding_dim = latents.shape

    effective_ranks = np.asarray([effective_rank(sample) for sample in latents], dtype=np.float64)

    if num_tokens > 1:
        prev_tokens = latents[:, :-1, :]
        next_tokens = latents[:, 1:, :]

        prev_norm = np.linalg.norm(prev_tokens, axis=-1)
        next_norm = np.linalg.norm(next_tokens, axis=-1)
        denom = np.clip(prev_norm * next_norm, 1e-12, None)
        adjacent_cos = np.sum(prev_tokens * next_tokens, axis=-1) / denom
        step_l2 = np.linalg.norm(next_tokens - prev_tokens, axis=-1)
    else:
        adjacent_cos = np.zeros((num_samples, 1), dtype=np.float64)
        step_l2 = np.zeros((num_samples, 1), dtype=np.float64)

    return {
        "num_samples_selected": float(num_samples),
        "num_tokens_used": float(num_tokens),
        "embedding_dim": float(embedding_dim),
        "effective_rank_mean": float(np.mean(effective_ranks)),
        "effective_rank_std": float(np.std(effective_ranks)),
        "effective_rank_ratio_mean": float(np.mean(effective_ranks / max(num_tokens, 1))),
        "effective_rank_ratio_std": float(np.std(effective_ranks / max(num_tokens, 1))),
        "adjacent_cosine_mean": float(np.mean(adjacent_cos)),
        "adjacent_cosine_std": float(np.std(adjacent_cos)),
        "step_l2_mean": float(np.mean(step_l2)),
        "step_l2_std": float(np.std(step_l2)),
        "randomsim": float(compute_randomsim(latents)),
    }


def compute_trajectory_metrics(trajectory_path: Path | None) -> dict[str, float]:
    if trajectory_path is None or not trajectory_path.exists():
        return {
            "radius_mean": math.nan,
            "radius_mean_std": math.nan,
            "radius_max": math.nan,
            "violation_rate_mean": math.nan,
            "violation_rate_max": math.nan,
        }

    data = load_json(trajectory_path)
    return {
        "radius_mean": float(data.get("radius_mean", math.nan)),
        "radius_mean_std": float(data.get("radius_mean_std", math.nan)),
        "radius_max": float(data.get("radius_max", math.nan)),
        "violation_rate_mean": float(data.get("violation_rate_mean", math.nan)),
        "violation_rate_max": float(data.get("violation_rate_max", math.nan)),
    }


def compute_loss_metrics(trainer_state_path: Path | None) -> dict[str, float]:
    if trainer_state_path is None or not trainer_state_path.exists():
        return {
            "loss_points": 0.0,
            "loss_mean": math.nan,
            "loss_std": math.nan,
            "loss_cv": math.nan,
            "tail_loss_mean": math.nan,
            "tail_loss_std": math.nan,
            "loss_step_diff_mean": math.nan,
            "loss_step_diff_p95": math.nan,
        }

    data = load_json(trainer_state_path)
    history = data.get("log_history", [])
    losses = np.asarray([item["loss"] for item in history if "loss" in item], dtype=np.float64)
    if losses.size == 0:
        return {
            "loss_points": 0.0,
            "loss_mean": math.nan,
            "loss_std": math.nan,
            "loss_cv": math.nan,
            "tail_loss_mean": math.nan,
            "tail_loss_std": math.nan,
            "loss_step_diff_mean": math.nan,
            "loss_step_diff_p95": math.nan,
        }

    tail_start = int(losses.size * 0.8)
    tail = losses[tail_start:] if tail_start < losses.size else losses
    diffs = np.abs(np.diff(losses)) if losses.size > 1 else np.zeros(1, dtype=np.float64)

    return {
        "loss_points": float(losses.size),
        "loss_mean": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "loss_cv": float(np.std(losses) / max(abs(np.mean(losses)), 1e-12)),
        "tail_loss_mean": float(np.mean(tail)),
        "tail_loss_std": float(np.std(tail)),
        "loss_step_diff_mean": float(np.mean(diffs)),
        "loss_step_diff_p95": float(np.percentile(diffs, 95)),
    }


def load_loss_curve(trainer_state_path: Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if trainer_state_path is None or not trainer_state_path.exists():
        return None

    data = load_json(trainer_state_path)
    history = data.get("log_history", [])
    steps = []
    losses = []
    for item in history:
        if "loss" in item and "step" in item:
            steps.append(item["step"])
            losses.append(item["loss"])
    if not losses:
        return None
    return np.asarray(steps, dtype=np.float64), np.asarray(losses, dtype=np.float64)


def analyze_run(spec: dict[str, Any], args: argparse.Namespace, index: int) -> dict[str, Any]:
    metrics = load_json(spec["metrics_path"])
    row: dict[str, Any] = {
        "t": int(spec["t"]) if spec.get("t") is not None else None,
        "condition": spec["condition"],
        "family": spec["family"],
        "label": spec["label"],
        "run_dir": str(spec["run_dir"]),
        "trainer_state": str(spec["trainer_state"]) if spec.get("trainer_state") else "",
        "accuracy": float(metrics.get("accuracy", math.nan)),
        "accuracy_pct": float(metrics.get("accuracy", math.nan) * 100.0),
        "avg_output_tokens": float(metrics.get("avg_output_tokens", math.nan)),
        "total_samples": float(metrics.get("total_samples", math.nan)),
        "correct": float(metrics.get("correct", math.nan)),
    }
    row.update(
        compute_latent_metrics(
            spec["latents_path"],
            max_samples=args.max_samples,
            seed=args.seed + index,
            remove_last_latent=not args.keep_last_latent,
        )
    )
    row.update(compute_trajectory_metrics(spec["trajectory_path"]))
    row.update(compute_loss_metrics(spec.get("trainer_state")))
    return row


def summarize_groups(run_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col for col in run_df.columns
        if pd.api.types.is_numeric_dtype(run_df[col]) and col not in {"t"}
    ]
    group_cols = ["family", "t", "condition"]
    grouped = run_df.groupby(group_cols, dropna=False)

    pieces = []
    for (family, t_value, condition), part in grouped:
        row: dict[str, Any] = {
            "family": family,
            "t": int(t_value),
            "condition": condition,
            "num_runs": int(len(part)),
        }
        for col in numeric_cols:
            values = part[col].astype(float)
            row[col] = float(values.mean()) if len(values) else math.nan
            row[f"{col}_run_std"] = float(values.std(ddof=0)) if len(values) else math.nan
        pieces.append(row)

    summary_df = pd.DataFrame(pieces)
    return summary_df.sort_values(["family", "t", "condition"]).reset_index(drop=True)


def build_delta_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    delta_rows = []
    metrics = [
        "accuracy_pct",
        "effective_rank_ratio_mean",
        "randomsim",
        "adjacent_cosine_mean",
        "step_l2_mean",
        "radius_mean",
        "violation_rate_mean",
        "tail_loss_std",
        "loss_step_diff_p95",
    ]

    for family in sorted(summary_df["family"].unique()):
        family_df = summary_df[summary_df["family"] == family]
        for t_value in sorted(family_df["t"].unique()):
            part = family_df[family_df["t"] == t_value]
            if {"+SIRCL", "no-SIRCL"} - set(part["condition"].tolist()):
                continue
            sircl_row = part[part["condition"] == "+SIRCL"].iloc[0]
            base_row = part[part["condition"] == "no-SIRCL"].iloc[0]

            delta = {
                "family": family,
                "t": int(t_value),
            }
            for metric in metrics:
                if metric in sircl_row.index and metric in base_row.index:
                    delta[f"{metric}_delta_sircl_minus_base"] = float(sircl_row[metric] - base_row[metric])
            delta_rows.append(delta)

    if not delta_rows:
        return pd.DataFrame()
    return pd.DataFrame(delta_rows).sort_values(["family", "t"]).reset_index(drop=True)


def choose_metric_panels(summary_df: pd.DataFrame) -> list[tuple[str, str]]:
    panels = [
        ("accuracy_pct", "Accuracy (%)"),
        ("effective_rank_ratio_mean", "Effective Rank / T"),
        ("randomsim", "RandomSim"),
        ("adjacent_cosine_mean", "Adjacent Cosine"),
        ("radius_mean", "Radius Mean"),
    ]

    if summary_df["tail_loss_std"].notna().any():
        panels.append(("tail_loss_std", "Tail Loss Std"))
    else:
        panels.append(("violation_rate_mean", "Violation Rate"))
    return panels


def plot_group_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        return

    metric_panels = choose_metric_panels(summary_df)
    color_map = {
        "+SIRCL": COLOR_LIST[0],
        "no-SIRCL": COLOR_LIST[1],
    }

    for family in sorted(summary_df["family"].unique()):
        family_df = summary_df[summary_df["family"] == family].copy()
        t_values = sorted(family_df["t"].unique())
        conditions = [cond for cond in ["no-SIRCL", "+SIRCL"] if cond in set(family_df["condition"])]
        if not conditions:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#FFFFFF")
        axes = axes.ravel()
        x = np.arange(len(t_values))
        width = 0.35 if len(conditions) > 1 else 0.55

        for ax, (metric_key, title) in zip(axes, metric_panels):
            for idx, condition in enumerate(conditions):
                cond_df = family_df[family_df["condition"] == condition]
                cond_df = cond_df.set_index("t").reindex(t_values)
                values = cond_df[metric_key].to_numpy(dtype=float)
                offsets = x + (idx - (len(conditions) - 1) / 2.0) * width
                bars = ax.bar(
                    offsets,
                    values,
                    width=width,
                    color=color_map.get(condition, COLOR_LIST[idx % len(COLOR_LIST)]),
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=1.0,
                    label=condition,
                    alpha=0.9,
                )
                for bar, value in zip(bars, values):
                    if np.isnan(value):
                        continue
                    ax.annotate(
                        f"{value:.3f}" if abs(value) < 10 else f"{value:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="#333333",
                    )

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([str(t) for t in t_values], fontsize=11, fontweight="bold")
            ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].legend(frameon=False, fontsize=11)
        fig.suptitle(f"{family}: Scaling Accuracy / Stability / Collapse", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = output_dir / f"{family.lower().replace(' ', '_')}_scaling_summary.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)


def smooth_curve(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def plot_training_curves(specs: list[dict[str, Any]], output_dir: Path, smooth_window: int) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        grouped.setdefault(spec["family"], []).append(spec)

    for family, family_specs in grouped.items():
        curves = []
        for spec in family_specs:
            curve = load_loss_curve(spec.get("trainer_state"))
            if curve is None:
                continue
            curves.append((spec, curve))
        if not curves:
            continue

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#FFFFFF")
        for idx, (spec, (steps, losses)) in enumerate(sorted(curves, key=lambda item: (item[0]["t"], item[0]["condition"]))):
            color = COLOR_LIST[idx % len(COLOR_LIST)]
            smoothed = smooth_curve(losses, smooth_window)
            ax.plot(steps, smoothed, linewidth=2.0, color=color, label=spec["label"], alpha=0.95)

        ax.set_xlabel("Training Step", fontsize=12, fontweight="bold")
        ax.set_ylabel("Smoothed Loss", fontsize=12, fontweight="bold")
        ax.set_title(f"{family}: Training Loss Curves", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()
        output_path = output_dir / f"{family.lower().replace(' ', '_')}_training_loss.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)


def format_number(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def write_markdown_tables(summary_df: pd.DataFrame, delta_df: pd.DataFrame, output_dir: Path) -> None:
    for family in sorted(summary_df["family"].unique()):
        family_df = summary_df[summary_df["family"] == family].copy()
        lines = [
            f"# {family} Scaling Summary",
            "",
            "| T | Condition | Accuracy (%) | EffRank/T | RandomSim | Adj Cos | Radius Mean | Tail Loss Std |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in family_df.sort_values(["t", "condition"]).iterrows():
            lines.append(
                "| "
                f"{int(row['t'])} | "
                f"{row['condition']} | "
                f"{row['accuracy_pct']:.2f} | "
                f"{row['effective_rank_ratio_mean']:.4f} | "
                f"{row['randomsim']:.4f} | "
                f"{row['adjacent_cosine_mean']:.4f} | "
                f"{row['radius_mean']:.4f} | "
                f"{format_number(row['tail_loss_std'], digits=4)} |"
            )

        family_delta = delta_df[delta_df["family"] == family] if not delta_df.empty else pd.DataFrame()
        if not family_delta.empty:
            lines.extend(
                [
                    "",
                    "## Delta (+SIRCL - no-SIRCL)",
                    "",
                    "| T | Delta Acc (%) | Delta EffRank/T | Delta RandomSim | Delta Tail Loss Std |",
                    "|---:|---:|---:|---:|---:|",
                ]
            )
            for _, row in family_delta.sort_values("t").iterrows():
                lines.append(
                    "| "
                    f"{int(row['t'])} | "
                    f"{format_number(row.get('accuracy_pct_delta_sircl_minus_base', math.nan), digits=2)} | "
                    f"{format_number(row.get('effective_rank_ratio_mean_delta_sircl_minus_base', math.nan), digits=4)} | "
                    f"{format_number(row.get('randomsim_delta_sircl_minus_base', math.nan), digits=4)} | "
                    f"{format_number(row.get('tail_loss_std_delta_sircl_minus_base', math.nan), digits=4)} |"
                )

        output_path = output_dir / f"{family.lower().replace(' ', '_')}_summary.md"
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t_filter = set(args.t_values) if args.t_values else None
    if args.manifest is not None:
        raw_specs = build_specs_from_manifest(resolve_path(args.manifest))
    else:
        raw_specs = build_specs_from_preset(args.preset, t_values=t_filter)

    specs = [ensure_run_spec(spec, remove_last_latent=not args.keep_last_latent) for spec in raw_specs]
    if t_filter:
        specs = [spec for spec in specs if spec.get("t") in t_filter]

    if not specs:
        raise RuntimeError("No valid run specs found for analysis.")

    run_rows = [analyze_run(spec, args, index=i) for i, spec in enumerate(specs)]
    run_df = pd.DataFrame(run_rows).sort_values(["family", "t", "condition", "label"]).reset_index(drop=True)
    summary_df = summarize_groups(run_df)
    delta_df = build_delta_summary(summary_df)

    run_csv = args.output_dir / "run_level_summary.csv"
    summary_csv = args.output_dir / "grouped_summary.csv"
    delta_csv = args.output_dir / "delta_summary.csv"

    run_df.to_csv(run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    if not delta_df.empty:
        delta_df.to_csv(delta_csv, index=False)

    write_markdown_tables(summary_df, delta_df, args.output_dir)
    plot_group_summary(summary_df, args.output_dir)
    plot_training_curves(specs, args.output_dir, smooth_window=args.smooth_window)

    print("=" * 80)
    print("Scaling / Stability / Collapse analysis finished")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    if not delta_df.empty:
        print("\nDelta summary (+SIRCL - no-SIRCL):")
        print(delta_df.to_string(index=False))
    print(f"\nSaved run-level CSV: {run_csv}")
    print(f"Saved grouped CSV:   {summary_csv}")
    if not delta_df.empty:
        print(f"Saved delta CSV:     {delta_csv}")
    print(f"Saved plots / md to: {args.output_dir}")


if __name__ == "__main__":
    main()
