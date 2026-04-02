#!/usr/bin/env python3
"""Locate train/eval commands and artifact paths for GSM8K multibackbone runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


BASELINE_ROOT = Path("/data/yhao/baseline")
RUN_ROOT = BASELINE_ROOT / "CODI_rebuttal_runs" / "rebuttal_20260325"
MULTIMODEL_TAG = "multimodel_gsm8k_math500_aime_v1"
MULTIMODEL_ROOT = RUN_ROOT / MULTIMODEL_TAG
COCONUT_ROOT = BASELINE_ROOT / "Coconut"
PLOTS_ROOT = BASELINE_ROOT / "CODI" / "plots" / "rebuttal_20260328"

POST_TRAIN_DATASETS = "gsm8k math500 aime svamp gsm-hard asdiv"


METHOD_ALIASES = {
    "cot-sft": "cot-sft",
    "cot_sft": "cot-sft",
    "simcot": "simcot",
    "simcon": "simcot",
    "simcot+sircl": "simcot+sircl",
    "simcot_sircl": "simcot+sircl",
    "simcon+sircl": "simcot+sircl",
    "simcon_sircl": "simcot+sircl",
    "codi": "codi",
    "codi+sircl": "codi+sircl",
    "codi_sircl": "codi+sircl",
}

BACKBONE_ALIASES = {
    "llama3-3b": "llama3-3b",
    "llama3b": "llama3-3b",
    "llama-3.2-3b": "llama3-3b",
    "qwen3-4b": "qwen3-4b",
    "qwen3": "qwen3-4b",
    "qwen3_4b": "qwen3-4b",
    "qwen3-1.7b": "qwen3-1.7b",
    "qwen3-1p7b": "qwen3-1.7b",
    "qwen3_1p7b": "qwen3-1.7b",
}


@dataclass
class ExperimentInfo:
    method: str
    backbone: str
    status: str
    train_files: list[str]
    train_command: str
    test_files: list[str]
    test_command: str
    batch_eval_command: str | None
    report_paths: list[str]
    result_paths: list[str]
    weight_paths: list[str]
    notes: str
    observed_status: str
    observed_latest_checkpoint: str | None
    observed_best_checkpoint: str | None
    observed_checkpoint_count: int


def normalize_method(value: str) -> str:
    key = value.strip().lower()
    if key not in METHOD_ALIASES:
        raise SystemExit(f"Unsupported method: {value}")
    return METHOD_ALIASES[key]


def normalize_backbone(value: str) -> str:
    key = value.strip().lower()
    if key not in BACKBONE_ALIASES:
        raise SystemExit(f"Unsupported backbone: {value}")
    return BACKBONE_ALIASES[key]


def stage_reports() -> list[str]:
    return [
        str(PLOTS_ROOT / "THREE_BACKBONE_SUMMARY_20260330.md"),
        str(PLOTS_ROOT / "CURRENT_MULTIMODEL_RESULTS_20260329.md"),
        str(PLOTS_ROOT / "CURRENT_MULTIMODEL_RESULTS_20260329.csv"),
        str(PLOTS_ROOT / "EXPERIMENT_MASTER_SUMMARY.md"),
        str(RUN_ROOT / "EXPERIMENT_PROGRESS_REPORT_20260328.md"),
    ]


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    suffix = name.split("-", 1)[-1] if "-" in name else name.split("_", 1)[-1]
    try:
        return (int(suffix), name)
    except ValueError:
        return (-1, name)


def discover_checkpoints(root: Path) -> list[Path]:
    if not root.exists():
        return []
    matches = [
        path
        for path in root.iterdir()
        if path.name.startswith("checkpoint-") or path.name.startswith("checkpoint_")
    ]
    return sorted(matches, key=checkpoint_sort_key)


def best_checkpoint_from_comparison_matrix(path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)
    if not first_row:
        return None
    model_name = first_row.get("model", "")
    if "checkpoint-" in model_name:
        return "checkpoint-" + model_name.rsplit("checkpoint-", 1)[1]
    if "checkpoint_" in model_name:
        return "checkpoint_" + model_name.rsplit("checkpoint_", 1)[1]
    return None


def summarize_observed_artifacts(checkpoint_root: Path, comparison_matrix: Path) -> tuple[str, str | None, str | None, int]:
    checkpoints = discover_checkpoints(checkpoint_root)
    latest = checkpoints[-1].name if checkpoints else None
    best = best_checkpoint_from_comparison_matrix(comparison_matrix)
    count = len(checkpoints)
    if comparison_matrix.exists():
        status = "summary present"
    elif count:
        status = "checkpoints present, no comparison_matrix.csv"
    else:
        status = "no checkpoints observed"
    return status, latest, best, count


def model_meta(backbone: str) -> dict[str, object]:
    common_root = MULTIMODEL_ROOT / "models"
    if backbone == "llama3-3b":
        return {
            "model_name": "Llama-3.2-3B-Instruct",
            "model_path": str(common_root / "Llama-3.2-3B-Instruct"),
            "expt_prefix": f"{MULTIMODEL_TAG}_gsm8k_llama3b",
            "simcon_script": str(BASELINE_ROOT / "CODI" / "train_on_gsm8k_dataset" / "train_llama3b.sh"),
            "codi_script": str(BASELINE_ROOT / "CODI" / "train_on_gsm8k_dataset" / "train_llama3b_codi.sh"),
            "simcon_epochs": 10,
            "codi_epochs": 8,
            "eval_batch_size": 8,
            "prj_dim": 3072,
        }
    if backbone == "qwen3-4b":
        return {
            "model_name": "Qwen3-4B",
            "model_path": str(common_root / "Qwen3-4B"),
            "expt_prefix": f"{MULTIMODEL_TAG}_gsm8k_qwen3_4b",
            "simcon_script": str(BASELINE_ROOT / "CODI" / "train_on_gsm8k_dataset" / "train_qwen3.sh"),
            "codi_script": str(BASELINE_ROOT / "CODI" / "train_on_gsm8k_dataset" / "train_qwen3_codi.sh"),
            "simcon_epochs": 8,
            "codi_epochs": 8,
            "eval_batch_size": 8,
            "prj_dim": 2560,
        }
    if backbone == "qwen3-1.7b":
        return {
            "model_name": "Qwen3-1.7B",
            "model_path": str(common_root / "Qwen3-1.7B"),
            "expt_prefix": f"{MULTIMODEL_TAG}_gsm8k_qwen3_1p7b",
            "simcon_script": str(BASELINE_ROOT / "CODI" / "train_on_gsm8k_dataset" / "train_qwen3_1p7b.sh"),
            "codi_script": str(BASELINE_ROOT / "CODI" / "train_on_gsm8k_dataset" / "train_qwen3_1p7b_codi.sh"),
            "simcon_epochs": 10,
            "codi_epochs": 8,
            "eval_batch_size": 16,
            "prj_dim": 2048,
        }
    raise SystemExit(f"Unsupported backbone: {backbone}")


def codi_experiment(method: str, backbone: str) -> ExperimentInfo:
    meta = model_meta(backbone)
    if method in {"simcot", "simcot+sircl"}:
        family = "simcon"
        variant = "simcon" if method == "simcot" else "simcon_sircl"
        train_script = str(meta["simcon_script"])
        epochs = int(meta["simcon_epochs"])
    else:
        family = "codi"
        variant = "codi" if method == "codi" else "codi_sircl"
        train_script = str(meta["codi_script"])
        epochs = int(meta["codi_epochs"])

    lr = "0.0003"
    seed = "11"
    model_name = str(meta["model_name"])
    model_path = str(meta["model_path"])
    expt_name = f"{meta['expt_prefix']}_{variant}"
    checkpoint_root = MULTIMODEL_ROOT / "outputs" / expt_name / model_name / f"ep_{epochs}" / f"lr_{lr}" / f"seed_{seed}"
    sweep_root = MULTIMODEL_ROOT / "results" / "checkpoint_sweeps" / expt_name / model_name / f"ep_{epochs}" / f"lr_{lr}" / f"seed_{seed}"

    train_command = f"bash {train_script}"
    if variant.endswith("_sircl"):
        train_command += " --sircl"

    eval_batch_size = int(meta["eval_batch_size"])
    prj_dim = int(meta["prj_dim"])
    test_command = (
        f"cd {BASELINE_ROOT / 'CODI'} && "
        f"python test_multi_dataset.py "
        f"--model_name_or_path \"{model_path}\" "
        f"--ckpt_dir \"{checkpoint_root}/checkpoint-<step>\" "
        f"--datasets \"{POST_TRAIN_DATASETS}\" "
        f"--num_runs 1 "
        f"--result_dir \"{sweep_root}\" "
        f"--seed 11 "
        f"--model_max_length 512 "
        f"--bf16 "
        f"--lora_r 128 "
        f"--lora_alpha 32 "
        f"--lora_init "
        f"--batch_size {eval_batch_size} "
        f"--greedy True "
        f"--num_latent 6 "
        f"--use_prj True "
        f"--prj_dim {prj_dim} "
        f"--prj_no_ln False "
        f"--prj_dropout 0.0 "
        f"--inf_latent_iterations 6 "
        f"--remove_eos True "
        f"--use_lora True"
    )

    report_paths = [
        str(sweep_root / "summary" / "comparison_matrix.csv"),
        str(sweep_root / "summary" / "all_results.csv"),
        *stage_reports(),
    ]
    result_paths = [
        str(sweep_root),
        str(sweep_root / "datasets"),
        str(sweep_root / "models"),
    ]
    weight_paths = [
        str(checkpoint_root),
        str(checkpoint_root / "checkpoint-*"),
    ]

    notes = (
        "Train wrappers already run post-train multi-dataset evaluation automatically unless "
        "CODI_POST_TRAIN_EVAL disables it."
    )

    if backbone == "qwen3-1.7b":
        notes += " qwen3-1.7b now has real artifacts in the main run root; verify live summaries before describing a line as complete."
    if backbone == "llama3-3b" and method == "simcot":
        notes += (
            " The current reportable SIM-CoT sweep for llama3-3b lives in the deeper offline-side run "
            "under rebuttal_20260325/multimodel_gsm8k_math500_aime_v1_simcon_20260327_offline; "
            "the main-root observation shown here may be shallower."
        )
        report_paths.append(
            str(
                RUN_ROOT
                / "multimodel_gsm8k_math500_aime_v1_simcon_20260327_offline"
                / "results"
                / "checkpoint_sweeps"
                / "multimodel_gsm8k_math500_aime_v1_simcon_20260327_offline_gsm8k_llama3b_simcon"
                / "Llama-3.2-3B-Instruct"
                / "ep_10"
                / "lr_0.0003"
                / "seed_11"
                / "summary"
                / "comparison_matrix.csv"
            )
        )
    if backbone == "llama3-3b" and method == "codi":
        notes += (
            " The current reportable llama3-3b CODI row is a partial-sweep story tracked in the stage summary docs; "
            "do not infer reportable completeness from the main-root observation alone."
        )

    observed_status, observed_latest_checkpoint, observed_best_checkpoint, observed_checkpoint_count = summarize_observed_artifacts(
        checkpoint_root,
        sweep_root / "summary" / "comparison_matrix.csv",
    )

    return ExperimentInfo(
        method=method,
        backbone=backbone,
        status="supported",
        train_files=[train_script],
        train_command=train_command,
        test_files=[str(BASELINE_ROOT / "CODI" / "test_multi_dataset.py")],
        test_command=test_command,
        batch_eval_command=None,
        report_paths=report_paths,
        result_paths=result_paths,
        weight_paths=weight_paths,
        notes=notes,
        observed_status=observed_status,
        observed_latest_checkpoint=observed_latest_checkpoint,
        observed_best_checkpoint=observed_best_checkpoint,
        observed_checkpoint_count=observed_checkpoint_count,
    )


def cot_sft_experiment(backbone: str) -> ExperimentInfo:
    if backbone == "llama3-3b":
        train_files = [
            str(COCONUT_ROOT / "args" / "gsm_cot_llama3.yaml"),
            str(COCONUT_ROOT / "args" / "gsm_cot_llama3_eval.yaml"),
            str(COCONUT_ROOT / "scripts" / "batch_eval_cot_sft.sh"),
        ]
        return ExperimentInfo(
            method="cot-sft",
            backbone=backbone,
            status="supported",
            train_files=train_files,
            train_command=f"cd {COCONUT_ROOT} && torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot_llama3.yaml",
            test_files=[str(COCONUT_ROOT / "args" / "gsm_cot_llama3_eval.yaml")],
            test_command=f"cd {COCONUT_ROOT} && python run.py args/gsm_cot_llama3_eval.yaml",
            batch_eval_command=f"cd {COCONUT_ROOT} && bash scripts/batch_eval_cot_sft.sh",
            report_paths=[
                str(COCONUT_ROOT / "logs" / "eval_cot_llama3_*.log"),
                str(COCONUT_ROOT / "logs" / "eval_llama3_all_*.log"),
                *stage_reports(),
            ],
            result_paths=[
                str(COCONUT_ROOT / "ckpts" / "gsm-cot-llama3" / "multi_eval_*.json"),
                str(COCONUT_ROOT / "logs"),
            ],
            weight_paths=[
                str(COCONUT_ROOT / "ckpts" / "gsm-cot-llama3"),
                str(COCONUT_ROOT / "ckpts" / "gsm-cot-llama3" / "checkpoint_*"),
            ],
            notes="Current checked-in train and eval args are ready. Batch multi-dataset eval uses scripts/batch_eval_cot_sft.sh.",
            observed_status="checkpoints or eval outputs must be confirmed from Coconut artifacts",
            observed_latest_checkpoint=None,
            observed_best_checkpoint=None,
            observed_checkpoint_count=0,
        )

    if backbone == "qwen3-4b":
        train_files = [
            str(COCONUT_ROOT / "scripts" / "train_cot_qwen3.sh"),
            str(COCONUT_ROOT / "scripts" / "eval_cot_qwen3.sh"),
            str(COCONUT_ROOT / "args" / "gsm_cot_qwen3.yaml"),
            str(COCONUT_ROOT / "args" / "gsm_cot_qwen3_eval.yaml"),
            str(COCONUT_ROOT / "scripts" / "batch_eval_cot_sft.sh"),
        ]
        return ExperimentInfo(
            method="cot-sft",
            backbone=backbone,
            status="supported",
            train_files=train_files,
            train_command=f"cd {COCONUT_ROOT} && bash scripts/train_cot_qwen3.sh 4",
            test_files=[str(COCONUT_ROOT / "scripts" / "eval_cot_qwen3.sh")],
            test_command=f"cd {COCONUT_ROOT} && bash scripts/eval_cot_qwen3.sh 4",
            batch_eval_command=f"cd {COCONUT_ROOT} && bash scripts/batch_eval_cot_sft.sh",
            report_paths=[
                str(COCONUT_ROOT / "logs" / "eval_cot_qwen3_*.log"),
                str(COCONUT_ROOT / "logs" / "eval_qwen3_all_*.log"),
                *stage_reports(),
            ],
            result_paths=[
                str(COCONUT_ROOT / "ckpts" / "gsm-qwen3-cot-sft" / "multi_eval_*.json"),
                str(COCONUT_ROOT / "logs"),
            ],
            weight_paths=[
                str(COCONUT_ROOT / "ckpts" / "gsm-qwen3-cot-sft"),
                str(COCONUT_ROOT / "ckpts" / "gsm-qwen3-cot-sft" / "checkpoint_*"),
            ],
            notes="Current checked-in train and eval wrappers target Qwen3-4B directly.",
            observed_status="checkpoints or eval outputs must be confirmed from Coconut artifacts",
            observed_latest_checkpoint=None,
            observed_best_checkpoint=None,
            observed_checkpoint_count=0,
        )

    if backbone == "qwen3-1.7b":
        checkpoint_root = COCONUT_ROOT / "ckpts" / "gsm-qwen3-1p7b-cot-sft"
        observed_status, observed_latest_checkpoint, observed_best_checkpoint, observed_checkpoint_count = summarize_observed_artifacts(
            checkpoint_root,
            checkpoint_root / "summary" / "comparison_matrix.csv",
        )
        train_files = [
            str(COCONUT_ROOT / "args" / "gsm_cot_qwen3_1p7b.yaml"),
            str(COCONUT_ROOT / "args" / "gsm_cot_qwen3_1p7b_eval.yaml"),
            str(COCONUT_ROOT / "scripts" / "train_cot_qwen3_1p7b.sh"),
            str(COCONUT_ROOT / "scripts" / "eval_cot_qwen3_1p7b.sh"),
            str(COCONUT_ROOT / "scripts" / "batch_eval_cot_sft.sh"),
        ]
        return ExperimentInfo(
            method="cot-sft",
            backbone=backbone,
            status="supported",
            train_files=train_files,
            train_command=f"cd {COCONUT_ROOT} && bash scripts/train_cot_qwen3_1p7b.sh 4",
            test_files=[str(COCONUT_ROOT / "scripts" / "eval_cot_qwen3_1p7b.sh")],
            test_command=f"cd {COCONUT_ROOT} && bash scripts/eval_cot_qwen3_1p7b.sh 4",
            batch_eval_command=f"cd {COCONUT_ROOT} && bash scripts/batch_eval_cot_sft.sh",
            report_paths=[
                str(COCONUT_ROOT / "logs" / "eval_qwen3_1p7b_multi_*.log"),
                *stage_reports(),
            ],
            result_paths=[
                str(COCONUT_ROOT / "ckpts" / "gsm-qwen3-1p7b-cot-sft" / "multi_eval_*.json"),
                str(COCONUT_ROOT / "logs"),
            ],
            weight_paths=[
                str(checkpoint_root),
                str(checkpoint_root / "checkpoint_*"),
            ],
            notes=(
                "Train, single-dataset eval, and batch multi-dataset eval wrappers are now checked in. "
                "A real run already exists and stopped early; confirm logs before claiming a reportable result."
            ),
            observed_status=observed_status,
            observed_latest_checkpoint=observed_latest_checkpoint,
            observed_best_checkpoint=observed_best_checkpoint,
            observed_checkpoint_count=observed_checkpoint_count,
        )

    raise SystemExit(f"Unsupported backbone for cot-sft: {backbone}")


def experiment_info(method: str, backbone: str) -> ExperimentInfo:
    if method == "cot-sft":
        return cot_sft_experiment(backbone)
    return codi_experiment(method, backbone)


def render_text(info: ExperimentInfo) -> str:
    lines = [
        f"Experiment: {info.method} / {info.backbone}",
        f"Status: {info.status}",
        "Train files:",
        *[f"  - {value}" for value in info.train_files],
        f"Train command: {info.train_command}",
        "Test files:",
        *[f"  - {value}" for value in info.test_files],
        f"Test command: {info.test_command}",
    ]
    if info.batch_eval_command:
        lines.append(f"Batch eval command: {info.batch_eval_command}")
    lines.extend(
        [
            "Report paths:",
            *[f"  - {value}" for value in info.report_paths],
            "Result paths:",
            *[f"  - {value}" for value in info.result_paths],
            "Weight paths:",
            *[f"  - {value}" for value in info.weight_paths],
            "Observed artifacts:",
            f"  - status: {info.observed_status}",
            f"  - checkpoint count: {info.observed_checkpoint_count}",
            f"  - latest checkpoint: {info.observed_latest_checkpoint or '-'}",
            f"  - best checkpoint from summary: {info.observed_best_checkpoint or '-'}",
            f"Notes: {info.notes}",
        ]
    )
    return "\n".join(lines)


def render_markdown_block(info: ExperimentInfo) -> str:
    lines = [
        f"### `{info.method}` + `{info.backbone}`",
        "",
        f"- Status: `{info.status}`",
        "- Train files:",
        *[f"  - `{value}`" for value in info.train_files],
        f"- Train command: `{info.train_command}`",
        "- Test files:",
        *[f"  - `{value}`" for value in info.test_files],
        f"- Test command: `{info.test_command}`",
    ]
    if info.batch_eval_command:
        lines.append(f"- Batch eval command: `{info.batch_eval_command}`")
    lines.extend(
        [
            "- Report paths:",
            *[f"  - `{value}`" for value in info.report_paths],
            "- Result paths:",
            *[f"  - `{value}`" for value in info.result_paths],
            "- Weight paths:",
            *[f"  - `{value}`" for value in info.weight_paths],
            "- Observed artifacts:",
            f"  - status: `{info.observed_status}`",
            f"  - checkpoint count: `{info.observed_checkpoint_count}`",
            f"  - latest checkpoint: `{info.observed_latest_checkpoint or '-'}`",
            f"  - best checkpoint from summary: `{info.observed_best_checkpoint or '-'}`",
            f"- Notes: {info.notes}",
            "",
        ]
    )
    return "\n".join(lines)


def render_markdown_all(items: list[ExperimentInfo]) -> str:
    lines = [
        "# Experiment Matrix",
        "",
        "Generated from `scripts/locate_experiment.py`.",
        "",
        "This file captures the 15 GSM8K experiment combinations across five methods and three backbones.",
        "",
    ]
    for backbone in ("llama3-3b", "qwen3-4b", "qwen3-1.7b"):
        lines.append(f"## `{backbone}`")
        lines.append("")
        for item in items:
            if item.backbone == backbone:
                lines.append(render_markdown_block(item))
    return "\n".join(lines).rstrip() + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", help="Method name")
    parser.add_argument("--backbone", help="Backbone name")
    parser.add_argument("--all", action="store_true", help="Print all combinations")
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Output format",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.all:
        items = [
            experiment_info(method, backbone)
            for backbone in ("llama3-3b", "qwen3-4b", "qwen3-1.7b")
            for method in ("cot-sft", "simcot", "simcot+sircl", "codi", "codi+sircl")
        ]
        if args.format == "json":
            print(json.dumps([asdict(item) for item in items], indent=2, ensure_ascii=False))
            return
        if args.format == "markdown":
            print(render_markdown_all(items), end="")
            return
        print("\n\n".join(render_text(item) for item in items))
        return

    if not args.method or not args.backbone:
        parser.error("Either pass --all, or pass both --method and --backbone.")

    info = experiment_info(normalize_method(args.method), normalize_backbone(args.backbone))
    if args.format == "json":
        print(json.dumps(asdict(info), indent=2, ensure_ascii=False))
    elif args.format == "markdown":
        print(render_markdown_block(info))
    else:
        print(render_text(info))


if __name__ == "__main__":
    main()
