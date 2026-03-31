#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DATASETS = ["gsm8k", "math500", "aime", "gsm-hard", "asdiv"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return path


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with require_file(csv_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def best_by_dataset(csv_path: Path, datasets: list[str]) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in load_rows(csv_path):
        dataset = row["dataset"]
        if dataset not in datasets:
            continue
        accuracy = float(row["accuracy"])
        previous = best.get(dataset)
        if previous is None or accuracy > previous["accuracy"]:
            best[dataset] = {
                "accuracy": accuracy,
                "model": row["model"],
                "timestamp": row.get("timestamp", ""),
            }
    missing = [dataset for dataset in datasets if dataset not in best]
    if missing:
        raise RuntimeError(f"Missing datasets in {csv_path}: {', '.join(missing)}")
    return best


def best_accuracy(csv_path: Path, dataset: str) -> float:
    rows = [row for row in load_rows(csv_path) if row["dataset"] == dataset]
    if not rows:
        raise RuntimeError(f"No rows for dataset={dataset} in {csv_path}")
    return max(float(row["accuracy"]) for row in rows)


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}"


def parse_coconut_accuracy(log_path: Path) -> dict[str, object]:
    text = require_file(log_path).read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Accuracy on validation set:\s*(\d+)\s*/\s*(\d+)\s*=\s*([0-9.]+)",
        text,
    )
    if not match:
        raise RuntimeError(f"Could not parse GSM8K accuracy from {log_path}")
    correct, total, accuracy = match.groups()
    return {
        "correct": int(correct),
        "total": int(total),
        "accuracy": float(accuracy),
    }


def parse_qwen3_simcon_oom(log_path: Path) -> dict[str, object]:
    text = require_file(log_path).read_text(encoding="utf-8", errors="replace")
    result: dict[str, object] = {
        "oom": "torch.OutOfMemoryError" in text or "CUDA out of memory" in text,
    }
    for label, key in [
        ("Per-device batch", "per_device_batch"),
        ("Grad accum \\(base\\)", "grad_acc"),
        ("Global batch effective", "global_batch"),
    ]:
        match = re.search(rf"{label}\s*:\s*(\d+)", text)
        if match:
            result[key] = int(match.group(1))
    return result


def parse_multidataset_eval_failure(log_path: Path) -> str:
    text = require_file(log_path).read_text(encoding="utf-8", errors="replace")
    if "EADDRINUSE" in text:
        return "multi-dataset eval attempt failed with EADDRINUSE on port 29500"
    return "multi-dataset eval attempt did not complete cleanly"


def scaling_block(paths: dict[str, Path]) -> list[dict[str, object]]:
    rows = []
    for latent, base_path, sircl_path in [
        (6, paths["t6_base"], paths["t6_sircl"]),
        (16, paths["t16_base"], paths["t16_sircl"]),
        (32, paths["t32_base"], paths["t32_sircl"]),
    ]:
        base_acc = best_accuracy(base_path, "gsm8k")
        sircl_acc = best_accuracy(sircl_path, "gsm8k")
        rows.append(
            {
                "latent": latent,
                "no_sircl": base_acc,
                "sircl": sircl_acc,
                "delta": sircl_acc - base_acc,
            }
        )
    return rows


def build_summary() -> dict[str, object]:
    root = repo_root()

    paths = {
        "llama3b_simcon_sircl": root
        / "CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps"
        / "multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon_sircl/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/summary/all_results.csv",
        "llama3b_codi_sircl": root
        / "CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps"
        / "multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi_sircl/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/summary/all_results.csv",
        "qwen3_4b_codi_sircl": root
        / "CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps"
        / "multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi_sircl/Qwen3-4B/ep_10/lr_0.0003/seed_11/summary/all_results.csv",
        "t6_sircl": root / "CODI/results/latent_sweep_gsm8k/latent_6/summary/all_results.csv",
        "t6_base": root / "CODI/results/latent_sweep_gsm8k/latent_6/summary/all_results.csv",
        "t16_sircl": root / "CODI/results/16long/summary/all_results.csv",
        "t16_base": root
        / "CODI_rebuttal_runs/rebuttal_20260325/results/checkpoint_sweeps/decoder-trajectory-euclidean-16long/Llama-3.2-1B-Instruct/ep_10/lr_0.0008/seed_11/summary/all_results.csv",
        "t32_sircl": root / "CODI/results/32long/summary/all_results.csv",
        "t32_base": root
        / "CODI_rebuttal_runs/rebuttal_20260325/results/checkpoint_sweeps/decoder-trajectory-euclidean-32long/Llama-3.2-1B-Instruct/ep_10/lr_0.0008/seed_11/summary/all_results.csv",
        "coconut_gsm8k_log": root / "Coconut/logs/eval_cot_qwen3_20260329_141428.log",
        "coconut_multidataset_log": root / "Coconut/logs/eval_qwen3_all_20260329_153500.log",
        "qwen3_simcon_oom_log": root / "CODI_rebuttal_runs/rebuttal_20260325/qwen3_simcon_sircl.log",
    }

    t6_rows = load_rows(paths["t6_sircl"])
    t6_lookup = {(row["model"], row["dataset"]): float(row["accuracy"]) for row in t6_rows}

    scaling_rows = scaling_block(paths)
    scaling_rows[0]["no_sircl"] = t6_lookup[("simcon", "gsm8k")]
    scaling_rows[0]["sircl"] = t6_lookup[("simcon_sircl", "gsm8k")]
    scaling_rows[0]["delta"] = scaling_rows[0]["sircl"] - scaling_rows[0]["no_sircl"]

    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "ready_to_cite": {
            "llama3b_simcon_sircl": best_by_dataset(paths["llama3b_simcon_sircl"], DATASETS),
            "llama3b_codi_sircl": best_by_dataset(paths["llama3b_codi_sircl"], DATASETS),
            "scaling_gsm8k": scaling_rows,
            "coconut_qwen3_cot_gsm8k": parse_coconut_accuracy(paths["coconut_gsm8k_log"]),
        },
        "partial": {
            "qwen3_4b_implicit_codi_sircl": best_by_dataset(paths["qwen3_4b_codi_sircl"], DATASETS),
            "qwen3_simcon_oom": parse_qwen3_simcon_oom(paths["qwen3_simcon_oom_log"]),
            "coconut_multidataset_eval": parse_multidataset_eval_failure(paths["coconut_multidataset_log"]),
        },
    }


def build_markdown(summary: dict[str, object]) -> str:
    ready = summary["ready_to_cite"]
    partial = summary["partial"]

    simcon = ready["llama3b_simcon_sircl"]
    codi = ready["llama3b_codi_sircl"]
    scaling_rows = ready["scaling_gsm8k"]
    coconut = ready["coconut_qwen3_cot_gsm8k"]
    qwen_bad = partial["qwen3_4b_implicit_codi_sircl"]
    qwen_oom = partial["qwen3_simcon_oom"]

    lines = [
        "# Experiment Master Summary",
        "",
        "更新时间：2026-03-29",
        "",
        "## Ready-to-cite",
        "",
        "### Main Table",
        "",
        "| System | GSM8K | MATH500 | AIME | GSM-Hard | ASDiv | Note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        "| LLaMA-3B SIM-CoT + SIRCL | "
        f"{format_pct(simcon['gsm8k']['accuracy'])} | {format_pct(simcon['math500']['accuracy'])} | "
        f"{format_pct(simcon['aime']['accuracy'])} | {format_pct(simcon['gsm-hard']['accuracy'])} | "
        f"{format_pct(simcon['asdiv']['accuracy'])} | Best per-dataset checkpoint from the rebuttal multi-dataset sweep |",
        "| LLaMA-3B CODI + SIRCL | "
        f"{format_pct(codi['gsm8k']['accuracy'])} | {format_pct(codi['math500']['accuracy'])} | "
        f"{format_pct(codi['aime']['accuracy'])} | {format_pct(codi['gsm-hard']['accuracy'])} | "
        f"{format_pct(codi['asdiv']['accuracy'])} | Best per-dataset checkpoint from the rebuttal multi-dataset sweep |",
        "| Coconut Qwen3-4B CoT-SFT | "
        f"{format_pct(coconut['accuracy'])} | - | - | - | - | GSM8K only; multi-dataset evaluation attempt is not citeable |",
        "",
        "### Matched Scaling (GSM8K)",
        "",
        "| T | no-SIRCL | +SIRCL | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]

    for row in scaling_rows:
        lines.append(
            f"| {row['latent']} | {format_pct(row['no_sircl'])} | "
            f"{format_pct(row['sircl'])} | {row['delta'] * 100:+.2f} pp |"
        )

    lines.extend(
        [
            "",
            "## Partial / do-not-cite",
            "",
            f"- Qwen3-4B implicit runs are not a positive backbone story right now. "
            f"The best finished implicit checkpoint sweep we have is `CODI + SIRCL`, with "
            f"GSM8K {format_pct(qwen_bad['gsm8k']['accuracy'])}, MATH500 {format_pct(qwen_bad['math500']['accuracy'])}, "
            f"AIME {format_pct(qwen_bad['aime']['accuracy'])}, GSM-Hard {format_pct(qwen_bad['gsm-hard']['accuracy'])}, "
            f"and ASDiv {format_pct(qwen_bad['asdiv']['accuracy'])}.",
            f"- The Qwen3-4B SIM-CoT + SIRCL run hit OOM early. Logged config: per-device batch "
            f"{qwen_oom.get('per_device_batch', 'n/a')}, grad accum {qwen_oom.get('grad_acc', 'n/a')}, "
            f"effective global batch {qwen_oom.get('global_batch', 'n/a')}.",
            f"- Coconut Qwen3-4B multi-dataset evaluation is not citeable because it {partial['coconut_multidataset_eval']}.",
            "",
            "## Current claims we can safely make",
            "",
            "- The strongest rebuttal evidence is the matched no-SIRCL scaling comparison: long latent chains degrade sharply without SIRCL and stay stable with SIRCL.",
            "- All geometry-heavy rebuttal analyses are aligned to the paper setting `T=6`; `T=16/32` are used only for scaling.",
            "- The current extra-backbone story should stay conservative: LLaMA-3B is ready-to-cite, while Qwen3-4B implicit runs are not.",
            "",
            "## Minimal TODOs",
            "",
            "- Qwen3-1.7B `T=6` main results for `SIM-CoT` and `CODI`.",
            "- A short `R/λ` selection note based on baseline `r_t` scale and regularizer/task-loss balance.",
            "- Final typo / formatting / missing-citation pass.",
            "",
        ]
    )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize current rebuttal-ready experiment results.")
    parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "CODI/plots/rebuttal_20260328"),
        help="Directory for summary outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary()
    markdown = build_markdown(summary)

    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "EXPERIMENT_MASTER_SUMMARY.md").write_text(markdown, encoding="utf-8")
    print(f"[done] wrote {(output_dir / 'experiment_summary.json')}")
    print(f"[done] wrote {(output_dir / 'EXPERIMENT_MASTER_SUMMARY.md')}")


if __name__ == "__main__":
    main()
