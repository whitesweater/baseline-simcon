#!/usr/bin/env python3
"""
Analyze the centroid as a trajectory-level global reference / geometric anchor.

This script works in two layers:

1. Offline analysis from saved `latents.json` + `predictions.json`
   - cosine(mu, z1)
   - variance / length robustness summaries
   - shuffled / wrong-centroid diagnostic anchors

2. Optional model-backed analysis
   - cosine(mu, mean question token representation)
   - inference-time centroid injection to measure accuracy drop

Examples
--------
Offline only:
    python3 CODI/plots/analyze_centroid_reference.py \
      --dataset gsm8k \
      --run base=/path/to/base/gsm8k/run_0 \
      --run sircl=/path/to/sircl/gsm8k/run_0 \
      --output-dir CODI/plots/results/centroid_reference_llama3b

With model-backed probing + intervention:
    python3 CODI/plots/analyze_centroid_reference.py \
      --dataset gsm8k \
      --run base=/path/to/base/gsm8k/run_0 \
      --run sircl=/path/to/sircl/gsm8k/run_0 \
      --auto-model-probe \
      --run-intervention \
      --intervention-samples 256 \
      --output-dir CODI/plots/results/centroid_reference_llama3b
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from color_config import BAR_ALPHA, BAR_EDGE_COLOR, COLOR_LIST, GRID_ALPHA, LINE_COLOR  # noqa: E402
from src.model import ModelArguments, TrainingArguments  # noqa: E402
from test_multi_dataset import (  # noqa: E402
    MultiDatasetEvaluator,
    answers_match,
    canonicalize_dataset_name,
    compute_accuracy,
    extract_answer,
    load_dataset_by_name,
    prepare_questions_and_answers,
)


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {value}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return numerator / np.clip(denominator, 1e-8, None)


def fixed_point_free_permutation(num_items: int) -> np.ndarray:
    if num_items <= 1:
        return np.arange(num_items)
    return np.roll(np.arange(num_items), -1)


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return 0.0
    x_series = pd.Series(np.asarray(x, dtype=float))
    y_series = pd.Series(np.asarray(y, dtype=float))
    value = x_series.corr(y_series, method="spearman")
    return 0.0 if pd.isna(value) else float(value)


def safe_linear_slope(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return 0.0
    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=float)
    if np.allclose(x_np, x_np[0]):
        return 0.0
    slope, _ = np.polyfit(x_np, y_np, 1)
    return float(slope)


def dataframe_to_markdown(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows_"

    rendered_rows: List[List[str]] = []
    for _, row in df.iterrows():
        rendered = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                rendered.append(format(float(value), float_fmt))
            else:
                rendered.append(str(value))
        rendered_rows.append(rendered)

    headers = [str(col) for col in df.columns.tolist()]
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rendered_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_labeled_paths(values: Sequence[str], default_label_prefix: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for index, raw_value in enumerate(values):
        if "=" in raw_value:
            label, path = raw_value.split("=", 1)
        else:
            label = f"{default_label_prefix}{index}"
            path = raw_value
        label = label.strip()
        path = path.strip()
        if not label:
            raise ValueError(f"Invalid labeled path: {raw_value}")
        result[label] = path
    return result


def resolve_stage_root(run_dir: Path) -> Optional[Path]:
    parts = list(run_dir.resolve().parts)
    if "results" not in parts:
        return None
    results_idx = max(idx for idx, part in enumerate(parts) if part == "results")
    if results_idx == 0:
        return None
    return Path(*parts[:results_idx])


def infer_model_name_hint(run_dir: Path) -> Optional[str]:
    parts = list(run_dir.resolve().parts)
    if "checkpoint_sweeps" in parts:
        idx = parts.index("checkpoint_sweeps")
        if len(parts) > idx + 2:
            return parts[idx + 2]
    return None


def infer_checkpoint_dir(run_dir: Path) -> Optional[Path]:
    parts = list(run_dir.resolve().parts)
    if "checkpoint_sweeps" not in parts:
        return None

    idx = parts.index("checkpoint_sweeps")
    if len(parts) <= idx + 7:
        return None

    results_idx = idx - 1
    if results_idx < 0 or parts[results_idx] != "results":
        return None

    stage_root = Path(*parts[:results_idx])
    expt_name = parts[idx + 1]
    model_name = parts[idx + 2]
    epoch_dir = parts[idx + 3]
    lr_dir = parts[idx + 4]
    seed_dir = parts[idx + 5]
    model_dir_name = parts[idx + 7]
    checkpoint_match = re.search(r"_checkpoint-(\d+)$", model_dir_name)
    if not checkpoint_match:
        return None
    checkpoint_step = checkpoint_match.group(1)
    candidate = (
        stage_root
        / "outputs"
        / expt_name
        / model_name
        / epoch_dir
        / lr_dir
        / seed_dir
        / f"checkpoint-{checkpoint_step}"
    )
    return candidate if candidate.exists() else None


def infer_model_path_from_manifest(run_dir: Path) -> Optional[str]:
    hint = infer_model_name_hint(run_dir)
    stage_root = resolve_stage_root(run_dir)
    if hint is None or stage_root is None or not stage_root.exists():
        return None

    matches = sorted(stage_root.rglob(f"{hint}.manifest.json"))
    if not matches:
        return None

    manifest = load_json(matches[0])
    model_path = manifest.get("path")
    if not model_path:
        return None
    return str(model_path)


@dataclass
class RunArtifacts:
    label: str
    run_dir: Path
    latents: np.ndarray
    predictions: List[object]
    ground_truth: List[object]
    metrics: dict
    raw_outputs: List[str]
    correct_mask: np.ndarray
    centroids: np.ndarray
    z1: np.ndarray
    final_latent: np.ndarray
    question_mean_repr: Optional[np.ndarray] = None
    question_token_lengths: Optional[np.ndarray] = None
    question_repr_cos: Optional[np.ndarray] = None
    model_name_or_path: Optional[str] = None
    ckpt_dir: Optional[Path] = None
    model_name_hint: Optional[str] = None


class ModelProbeRuntime:
    def __init__(self, label: str, model_name_or_path: str, ckpt_dir: Path, args: argparse.Namespace, num_iterations: int):
        self.label = label
        self.model_name_or_path = model_name_or_path
        self.ckpt_dir = ckpt_dir
        self.args = args
        self.num_iterations = num_iterations
        self.evaluator: Optional[MultiDatasetEvaluator] = None

    def load(self) -> MultiDatasetEvaluator:
        if self.evaluator is not None:
            return self.evaluator

        model_args = ModelArguments(
            model_name_or_path=self.model_name_or_path,
            ckpt_dir=str(self.ckpt_dir),
            lora_init=True,
            lora_r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            train=False,
        )
        training_args = TrainingArguments(
            output_dir=str(self.args.output_dir / "_runtime" / self.label),
            model_max_length=self.args.model_max_length,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            bf16=torch.cuda.is_available(),
            use_prj=self.args.use_prj,
            prj_dim=self.args.prj_dim,
            prj_no_ln=self.args.prj_no_ln,
            prj_dropout=self.args.prj_dropout,
            num_latent=max(self.num_iterations - 1, 1),
            inf_latent_iterations=max(self.num_iterations - 1, 1),
            remove_eos=self.args.remove_eos,
            use_lora=True,
            greedy=self.args.greedy,
            report_to=[],
        )
        self.evaluator = MultiDatasetEvaluator(model_args, training_args)
        self.evaluator.load_model()
        return self.evaluator

    @torch.no_grad()
    def compute_question_mean_representations(self, questions: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        evaluator = self.load()
        tokenizer = evaluator.tokenizer
        model = evaluator.model

        representations: List[np.ndarray] = []
        token_lengths: List[np.ndarray] = []

        for start in range(0, len(questions), self.args.eval_batch_size):
            batch_questions = list(questions[start:start + self.args.eval_batch_size])
            batch = tokenizer(
                batch_questions,
                return_tensors="pt",
                padding="longest",
            )
            original_mask = batch["attention_mask"]
            original_length = batch["input_ids"].size(1)

            if evaluator.training_args.remove_eos:
                bot_tensor = torch.tensor(
                    [model.bot_id],
                    dtype=torch.long,
                ).expand(batch["input_ids"].size(0), 1)
            else:
                bot_tensor = torch.tensor(
                    [tokenizer.eos_token_id, model.bot_id],
                    dtype=torch.long,
                ).expand(batch["input_ids"].size(0), 2)

            input_ids = torch.cat((batch["input_ids"], bot_tensor), dim=1).to("cuda")
            attention_mask = torch.cat(
                (batch["attention_mask"], torch.ones_like(bot_tensor)),
                dim=1,
            ).to("cuda")

            outputs = model.codi(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.hidden_states[-1][:, :original_length, :]
            if evaluator.training_args.use_prj:
                hidden_states = model.prj(hidden_states)

            active_mask = original_mask.to(hidden_states.device).unsqueeze(-1).bool()
            summed = (hidden_states * active_mask).sum(dim=1)
            denom = active_mask.sum(dim=1).clamp(min=1)
            mean_repr = summed / denom

            representations.append(mean_repr.float().cpu().numpy())
            token_lengths.append(original_mask.sum(dim=1).cpu().numpy())

        return (
            np.concatenate(representations, axis=0).astype(np.float32),
            np.concatenate(token_lengths, axis=0).astype(np.int32),
        )

    @torch.no_grad()
    def run_generation(
        self,
        questions: Sequence[str],
        answers: Sequence[object],
        answer_type: str,
        injected_latents: Optional[np.ndarray] = None,
    ) -> dict:
        evaluator = self.load()
        tokenizer = evaluator.tokenizer
        model = evaluator.model

        predictions: List[object] = []
        raw_outputs: List[str] = []
        output_lengths: List[int] = []

        latent_dtype = next(model.parameters()).dtype

        for start in range(0, len(questions), self.args.intervention_batch_size):
            end = start + self.args.intervention_batch_size
            batch_questions = list(questions[start:end])
            batch = tokenizer(
                batch_questions,
                return_tensors="pt",
                padding="longest",
            )
            if evaluator.training_args.remove_eos:
                bot_tensor = torch.tensor(
                    [model.bot_id],
                    dtype=torch.long,
                ).expand(batch["input_ids"].size(0), 1)
            else:
                bot_tensor = torch.tensor(
                    [tokenizer.eos_token_id, model.bot_id],
                    dtype=torch.long,
                ).expand(batch["input_ids"].size(0), 2)
            batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
            batch["attention_mask"] = torch.cat(
                (batch["attention_mask"], torch.ones_like(bot_tensor)),
                dim=1,
            )
            batch = {key: value.to("cuda") for key, value in batch.items()}

            outputs = model.codi(
                input_ids=batch["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                attention_mask=batch["attention_mask"],
            )
            past_key_values = outputs.past_key_values

            if injected_latents is None:
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if evaluator.training_args.use_prj:
                    latent_embd = model.prj(latent_embd)
            else:
                batch_latents = torch.as_tensor(
                    injected_latents[start:end],
                    dtype=latent_dtype,
                    device="cuda",
                )
                latent_embd = batch_latents.unsqueeze(1)

            for _ in range(evaluator.training_args.inf_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if evaluator.training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            if evaluator.training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device="cuda")
                ).unsqueeze(0).expand(batch["input_ids"].size(0), -1, -1)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor(
                        [model.eot_id, tokenizer.eos_token_id],
                        dtype=torch.long,
                        device="cuda",
                    )
                ).unsqueeze(0).expand(batch["input_ids"].size(0), -1, -1)

            output = eot_emb
            finished = torch.zeros(batch["input_ids"].size(0), dtype=torch.bool, device="cuda")
            pred_tokens = [[] for _ in range(batch["input_ids"].size(0))]

            for _ in range(self.args.max_new_tokens):
                out = model.codi(
                    inputs_embeds=output,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]

                if evaluator.training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1)
                else:
                    logits = logits / self.args.temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                for batch_idx in range(batch["input_ids"].size(0)):
                    if not finished[batch_idx]:
                        token_id = int(next_token_ids[batch_idx].item())
                        pred_tokens[batch_idx].append(token_id)
                        if token_id == tokenizer.eos_token_id:
                            finished[batch_idx] = True

                if finished.all():
                    break
                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1)

            for local_idx, tokens in enumerate(pred_tokens):
                decoded = tokenizer.decode(tokens, skip_special_tokens=True)
                question_text = batch_questions[local_idx]
                pred = extract_answer(decoded, answer_type, question=question_text)
                predictions.append(pred)
                raw_outputs.append(decoded)
                output_lengths.append(len(tokens))

        accuracy = compute_accuracy(list(answers), predictions)
        return {
            "predictions": predictions,
            "raw_outputs": raw_outputs,
            "accuracy": accuracy,
            "avg_output_tokens": float(np.mean(output_lengths)) if output_lengths else 0.0,
            "num_samples": len(predictions),
        }

    def close(self) -> None:
        self.evaluator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_run_artifacts(label: str, run_dir: Path) -> RunArtifacts:
    latents_path = run_dir / "latents.json"
    predictions_path = run_dir / "predictions.json"
    metrics_path = run_dir / "metrics.json"
    raw_outputs_path = run_dir / "raw_outputs.json"

    if not latents_path.exists():
        raise FileNotFoundError(f"Missing latents.json for {label}: {latents_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions.json for {label}: {predictions_path}")

    latents_payload = load_json(latents_path)
    predictions_payload = load_json(predictions_path)
    metrics_payload = load_json(metrics_path) if metrics_path.exists() else {}
    raw_outputs_payload = load_json(raw_outputs_path) if raw_outputs_path.exists() else {}

    latents = np.asarray(latents_payload["latents"], dtype=np.float32)
    predictions = list(predictions_payload["predictions"])
    ground_truth = list(predictions_payload["ground_truth"])
    raw_outputs = list(raw_outputs_payload.get("raw_outputs", []))

    if latents.shape[0] != len(predictions):
        raise ValueError(
            f"Latent/prediction size mismatch for {label}: "
            f"{latents.shape[0]} vs {len(predictions)}"
        )

    correct_mask = np.asarray(
        [answers_match(pred, gold) for pred, gold in zip(predictions, ground_truth)],
        dtype=bool,
    )
    centroids = latents.mean(axis=1)

    return RunArtifacts(
        label=label,
        run_dir=run_dir,
        latents=latents,
        predictions=predictions,
        ground_truth=ground_truth,
        metrics=metrics_payload,
        raw_outputs=raw_outputs,
        correct_mask=correct_mask,
        centroids=centroids,
        z1=latents[:, 0, :],
        final_latent=latents[:, -1, :],
        model_name_hint=infer_model_name_hint(run_dir),
        ckpt_dir=infer_checkpoint_dir(run_dir),
        model_name_or_path=infer_model_path_from_manifest(run_dir),
    )


def build_sample_frame(
    run: RunArtifacts,
    questions: Sequence[str],
    question_word_lengths: np.ndarray,
    question_char_lengths: np.ndarray,
) -> pd.DataFrame:
    centroid = run.centroids
    latents = run.latents
    dists = np.linalg.norm(latents - centroid[:, None, :], axis=-1)

    frame = pd.DataFrame(
        {
            "run": run.label,
            "sample_idx": np.arange(latents.shape[0]),
            "question": list(questions),
            "prediction": [str(item) for item in run.predictions],
            "ground_truth": [str(item) for item in run.ground_truth],
            "correct": run.correct_mask.astype(bool),
            "question_word_length": question_word_lengths,
            "question_char_length": question_char_lengths,
            "radius_mean": dists.mean(axis=1),
            "radius_std": dists.std(axis=1),
            "radius_max": dists.max(axis=1),
            "trajectory_path_length": np.linalg.norm(np.diff(latents, axis=1), axis=-1).sum(axis=1),
            "cos_mu_z1": cosine_similarity_rows(centroid, run.z1),
            "cos_mu_final": cosine_similarity_rows(centroid, run.final_latent),
        }
    )

    if run.question_token_lengths is not None:
        frame["question_token_length"] = run.question_token_lengths

    if run.question_repr_cos is not None:
        frame["cos_mu_question_mean"] = run.question_repr_cos

    return frame


def summarize_probe_metrics(frame: pd.DataFrame, length_col: str) -> pd.DataFrame:
    rows = []
    for run_label, group in frame.groupby("run", sort=False):
        row = {
            "run": run_label,
            "num_samples": int(len(group)),
            "accuracy": float(group["correct"].mean()),
            "radius_mean_mean": float(group["radius_mean"].mean()),
            "radius_mean_std": float(group["radius_mean"].std(ddof=0)),
            "cos_mu_z1_mean": float(group["cos_mu_z1"].mean()),
            "cos_mu_z1_std": float(group["cos_mu_z1"].std(ddof=0)),
            "cos_mu_z1_abs_spearman_len": abs(safe_spearman(group[length_col], group["cos_mu_z1"])),
            "cos_mu_z1_slope_len": safe_linear_slope(group[length_col], group["cos_mu_z1"]),
        }
        if "cos_mu_question_mean" in group.columns:
            row.update(
                {
                    "cos_mu_question_mean_mean": float(group["cos_mu_question_mean"].mean()),
                    "cos_mu_question_mean_std": float(group["cos_mu_question_mean"].std(ddof=0)),
                    "cos_mu_question_mean_abs_spearman_len": abs(
                        safe_spearman(group[length_col], group["cos_mu_question_mean"])
                    ),
                    "cos_mu_question_mean_slope_len": safe_linear_slope(
                        group[length_col], group["cos_mu_question_mean"]
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_length_bins(lengths: Sequence[int], num_bins: int) -> pd.Categorical:
    series = pd.Series(np.asarray(lengths, dtype=int))
    distinct = int(series.nunique())
    quantiles = max(1, min(num_bins, distinct))
    if quantiles == 1:
        labels = pd.Series(["all"] * len(series))
        return pd.Categorical(labels, categories=["all"], ordered=True)
    return pd.qcut(series, q=quantiles, duplicates="drop")


def make_length_bin_summary(frame: pd.DataFrame, metric: str, length_col: str, num_bins: int) -> pd.DataFrame:
    if length_col not in frame.columns:
        raise ValueError(f"Missing length column: {length_col}")
    binned = compute_length_bins(frame[length_col], num_bins)
    annotated = frame.copy()
    annotated["length_bin"] = binned

    grouped = (
        annotated.groupby(["run", "length_bin"], observed=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["bin_label"] = grouped["length_bin"].astype(str)
    return grouped


def summarize_offline_intervention(
    run: RunArtifacts,
    question_mean_repr: Optional[np.ndarray],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    num_samples = run.centroids.shape[0]
    if num_samples == 0:
        return pd.DataFrame()

    own_anchor = run.centroids
    shuffled_anchor = run.centroids[fixed_point_free_permutation(num_samples)]

    wrong_pool = np.flatnonzero(~run.correct_mask)
    wrong_anchor: Optional[np.ndarray]
    if len(wrong_pool) == 0:
        wrong_anchor = None
    else:
        chosen = rng.choice(wrong_pool, size=num_samples, replace=True)
        if len(wrong_pool) > 1:
            for sample_idx in range(num_samples):
                if chosen[sample_idx] == sample_idx:
                    alternatives = wrong_pool[wrong_pool != sample_idx]
                    chosen[sample_idx] = int(alternatives[0])
        wrong_anchor = run.centroids[chosen]

    rows = []
    for mode_name, anchor in (
        ("own", own_anchor),
        ("shuffled", shuffled_anchor),
        ("wrong", wrong_anchor),
    ):
        if anchor is None:
            continue
        token_anchor_dist = np.linalg.norm(run.latents - anchor[:, None, :], axis=-1).mean(axis=1)
        row = {
            "run": run.label,
            "anchor_mode": mode_name,
            "token_anchor_dist_mean": float(token_anchor_dist.mean()),
            "token_anchor_dist_std": float(token_anchor_dist.std(ddof=0)),
            "z1_anchor_cos_mean": float(cosine_similarity_rows(anchor, run.z1).mean()),
            "z1_anchor_cos_std": float(cosine_similarity_rows(anchor, run.z1).std(ddof=0)),
        }
        if question_mean_repr is not None:
            question_anchor_cos = cosine_similarity_rows(anchor, question_mean_repr)
            row["question_anchor_cos_mean"] = float(question_anchor_cos.mean())
            row["question_anchor_cos_std"] = float(question_anchor_cos.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_probing(
    sample_frame: pd.DataFrame,
    length_col: str,
    output_path: Path,
    num_length_bins: int,
) -> None:
    run_order = list(dict.fromkeys(sample_frame["run"].tolist()))
    colors = COLOR_LIST * ((len(run_order) + len(COLOR_LIST) - 1) // len(COLOR_LIST))
    has_question_probe = (
        "cos_mu_question_mean" in sample_frame.columns
        and sample_frame.groupby("run")["cos_mu_question_mean"].apply(lambda series: series.notna().all()).all()
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#FFFFFF")
    for axis in axes.flat:
        axis.set_facecolor("#FFFFFF")

    box_data = [sample_frame.loc[sample_frame["run"] == run_name, "cos_mu_z1"].values for run_name in run_order]
    box = axes[0, 0].boxplot(box_data, patch_artist=True, tick_labels=run_order)
    for patch, color in zip(box["boxes"], colors[: len(run_order)]):
        patch.set(facecolor=color, edgecolor=BAR_EDGE_COLOR, alpha=BAR_ALPHA)
    axes[0, 0].set_title("Anchor Similarity to First Latent Token")
    axes[0, 0].set_ylabel("cos(mu, z1)")

    if has_question_probe:
        question_box_data = [
            sample_frame.loc[sample_frame["run"] == run_name, "cos_mu_question_mean"].values
            for run_name in run_order
        ]
        q_box = axes[0, 1].boxplot(question_box_data, patch_artist=True, tick_labels=run_order)
        for patch, color in zip(q_box["boxes"], colors[: len(run_order)]):
            patch.set(facecolor=color, edgecolor=BAR_EDGE_COLOR, alpha=BAR_ALPHA)
        axes[0, 1].set_title("Anchor Similarity to Mean Question Representation")
        axes[0, 1].set_ylabel("cos(mu, mean-question)")
    else:
        radius_means = [
            sample_frame.loc[sample_frame["run"] == run_name, "radius_mean"].mean()
            for run_name in run_order
        ]
        radius_stds = [
            sample_frame.loc[sample_frame["run"] == run_name, "radius_mean"].std(ddof=0)
            for run_name in run_order
        ]
        axes[0, 1].bar(
            run_order,
            radius_means,
            yerr=radius_stds,
            color=colors[: len(run_order)],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.2,
            alpha=BAR_ALPHA,
            capsize=5,
        )
        axes[0, 1].set_title("Mean Radius Around Geometric Anchor")
        axes[0, 1].set_ylabel("Mean token-to-anchor distance")

    z1_bin_summary = make_length_bin_summary(sample_frame, "cos_mu_z1", length_col, num_length_bins)
    z1_bin_order = list(dict.fromkeys(z1_bin_summary["bin_label"].tolist()))
    for idx, run_name in enumerate(run_order):
        group = (
            z1_bin_summary[z1_bin_summary["run"] == run_name]
            .set_index("bin_label")
            .reindex(z1_bin_order)
            .reset_index()
        )
        x_positions = np.arange(len(z1_bin_order))
        axes[1, 0].plot(
            x_positions,
            group["mean"],
            marker="o",
            linewidth=2.0,
            label=run_name,
            color=colors[idx],
        )
        axes[1, 0].fill_between(
            x_positions,
            group["mean"] - group["std"].fillna(0.0),
            group["mean"] + group["std"].fillna(0.0),
            alpha=0.12,
            color=colors[idx],
        )
    axes[1, 0].set_title("Length Robustness of Anchor-to-z1 Similarity")
    axes[1, 0].set_xlabel(length_col.replace("_", " ").title())
    axes[1, 0].set_ylabel("Mean cos(mu, z1)")
    axes[1, 0].set_xticks(np.arange(len(z1_bin_order)))
    axes[1, 0].set_xticklabels(z1_bin_order, rotation=15)
    axes[1, 0].legend(frameon=False)

    summary_rows = []
    for run_name in run_order:
        group = sample_frame[sample_frame["run"] == run_name]
        summary_rows.append(
            {
                "run": run_name,
                "z1 std": float(group["cos_mu_z1"].std(ddof=0)),
                "|rho_len(z1)|": abs(safe_spearman(group[length_col], group["cos_mu_z1"])),
                "q std": float(group["cos_mu_question_mean"].std(ddof=0)) if has_question_probe else np.nan,
                "|rho_len(q)|": abs(safe_spearman(group[length_col], group["cos_mu_question_mean"]))
                if has_question_probe
                else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    x_positions = np.arange(len(run_order))
    width = 0.18 if has_question_probe else 0.32
    axes[1, 1].bar(
        x_positions - width / 2,
        summary_df["z1 std"],
        width=width,
        label="std[cos(mu, z1)]",
        color=colors[: len(run_order)],
        alpha=0.85,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=1.0,
    )
    axes[1, 1].bar(
        x_positions + width / 2,
        summary_df["|rho_len(z1)|"],
        width=width,
        label="|rho_len(cos(mu, z1))|",
        color="#444444",
        alpha=0.65,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=1.0,
    )
    if has_question_probe:
        axes[1, 1].bar(
            x_positions + 1.5 * width,
            summary_df["q std"],
            width=width,
            label="std[cos(mu, mean-question)]",
            color=LINE_COLOR,
            alpha=0.65,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.0,
        )
        axes[1, 1].bar(
            x_positions + 2.5 * width,
            summary_df["|rho_len(q)|"],
            width=width,
            label="|rho_len(cos(mu, mean-question))|",
            color="#8B5E3C",
            alpha=0.65,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.0,
        )
    axes[1, 1].set_xticks(x_positions)
    axes[1, 1].set_xticklabels(run_order)
    axes[1, 1].set_title("Stability and Length Robustness Summary")
    axes[1, 1].legend(frameon=False, fontsize=9)

    for axis in axes.flat:
        axis.grid(True, linestyle="--", alpha=GRID_ALPHA)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def plot_offline_intervention(intervention_df: pd.DataFrame, output_path: Path) -> None:
    if intervention_df.empty:
        return

    run_order = list(dict.fromkeys(intervention_df["run"].tolist()))
    mode_order = [mode for mode in ["own", "shuffled", "wrong"] if mode in intervention_df["anchor_mode"].unique()]
    colors = {
        "own": COLOR_LIST[0],
        "shuffled": COLOR_LIST[1],
        "wrong": COLOR_LIST[2],
    }

    has_question_probe = (
        "question_anchor_cos_mean" in intervention_df.columns
        and intervention_df["question_anchor_cos_mean"].notna().all()
    )
    num_cols = 3 if has_question_probe else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 5), facecolor="#FFFFFF")
    if num_cols == 1:
        axes = [axes]

    metrics = [
        ("token_anchor_dist_mean", "Token-to-anchor distance", "Lower is better"),
        ("z1_anchor_cos_mean", "Anchor to first latent cosine", "Higher is better"),
    ]
    if has_question_probe:
        metrics.append(("question_anchor_cos_mean", "Anchor to mean question cosine", "Higher is better"))

    x_positions = np.arange(len(run_order))
    width = 0.22

    for axis, (column_name, title, y_label) in zip(axes, metrics):
        for mode_idx, mode_name in enumerate(mode_order):
            subset = intervention_df[intervention_df["anchor_mode"] == mode_name]
            values = []
            for run_name in run_order:
                row = subset[subset["run"] == run_name]
                values.append(float(row.iloc[0][column_name]) if not row.empty else np.nan)
            axis.bar(
                x_positions + (mode_idx - (len(mode_order) - 1) / 2) * width,
                values,
                width=width,
                label=mode_name,
                color=colors.get(mode_name, COLOR_LIST[mode_idx]),
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=1.1,
            )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(run_order)
        axis.set_title(title)
        axis.set_ylabel(y_label)
        axis.grid(True, linestyle="--", alpha=GRID_ALPHA)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def plot_model_intervention(intervention_df: pd.DataFrame, output_path: Path) -> None:
    if intervention_df.empty:
        return

    run_order = list(dict.fromkeys(intervention_df["run"].tolist()))
    mode_order = list(dict.fromkeys(intervention_df["mode"].tolist()))
    x_positions = np.arange(len(run_order))
    width = 0.18
    colors = COLOR_LIST * ((len(mode_order) + len(COLOR_LIST) - 1) // len(COLOR_LIST))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#FFFFFF")

    for mode_idx, mode_name in enumerate(mode_order):
        subset = intervention_df[intervention_df["mode"] == mode_name]
        accuracy_values = []
        token_values = []
        for run_name in run_order:
            row = subset[subset["run"] == run_name]
            accuracy_values.append(float(row.iloc[0]["accuracy"]) if not row.empty else np.nan)
            token_values.append(float(row.iloc[0]["avg_output_tokens"]) if not row.empty else np.nan)
        offset = (mode_idx - (len(mode_order) - 1) / 2) * width
        axes[0].bar(
            x_positions + offset,
            accuracy_values,
            width=width,
            label=mode_name,
            color=colors[mode_idx],
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.1,
        )
        axes[1].bar(
            x_positions + offset,
            token_values,
            width=width,
            label=mode_name,
            color=colors[mode_idx],
            alpha=BAR_ALPHA,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.1,
        )

    axes[0].set_title("Inference-Time Centroid Injection Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Inference-Time Centroid Injection Output Length")
    axes[1].set_ylabel("Average output tokens")

    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels(run_order)
        axis.grid(True, linestyle="--", alpha=GRID_ALPHA)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def attach_model_backed_question_probing(
    runs: List[RunArtifacts],
    questions: Sequence[str],
    args: argparse.Namespace,
) -> None:
    explicit_model_paths = parse_labeled_paths(args.model_name_or_path, "model") if args.model_name_or_path else {}
    explicit_ckpt_dirs = parse_labeled_paths(args.ckpt_dir, "ckpt") if args.ckpt_dir else {}

    for run in runs:
        if run.label in explicit_model_paths:
            run.model_name_or_path = explicit_model_paths[run.label]
        elif len(explicit_model_paths) == 1 and run.model_name_or_path is None:
            run.model_name_or_path = next(iter(explicit_model_paths.values()))

        if run.label in explicit_ckpt_dirs:
            run.ckpt_dir = Path(explicit_ckpt_dirs[run.label])

        if run.model_name_or_path is None:
            continue
        if run.ckpt_dir is None:
            continue

        runtime = ModelProbeRuntime(
            label=run.label,
            model_name_or_path=run.model_name_or_path,
            ckpt_dir=run.ckpt_dir,
            args=args,
            num_iterations=run.latents.shape[1],
        )
        try:
            question_repr, question_lengths = runtime.compute_question_mean_representations(questions)
            run.question_mean_repr = question_repr
            run.question_token_lengths = question_lengths
            run.question_repr_cos = cosine_similarity_rows(run.centroids, question_repr)
        finally:
            runtime.close()


def run_model_interventions(
    runs: List[RunArtifacts],
    questions: Sequence[str],
    answers: Sequence[object],
    answer_type: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    if args.intervention_samples <= 0:
        subset_indices = np.arange(len(questions))
    else:
        subset_size = min(int(args.intervention_samples), len(questions))
        rng = np.random.default_rng(args.seed)
        subset_indices = np.sort(rng.choice(len(questions), size=subset_size, replace=False))

    subset_questions = [questions[idx] for idx in subset_indices]
    subset_answers = [answers[idx] for idx in subset_indices]

    for run in runs:
        if run.model_name_or_path is None or run.ckpt_dir is None:
            continue

        runtime = ModelProbeRuntime(
            label=run.label,
            model_name_or_path=run.model_name_or_path,
            ckpt_dir=run.ckpt_dir,
            args=args,
            num_iterations=run.latents.shape[1],
        )
        try:
            own_centroids = run.centroids[subset_indices]
            shuffled_centroids = own_centroids[fixed_point_free_permutation(len(subset_indices))]

            wrong_pool = np.flatnonzero(~run.correct_mask)
            if len(wrong_pool) > 0:
                rng = np.random.default_rng(args.seed)
                chosen_wrong = rng.choice(wrong_pool, size=len(subset_indices), replace=True)
                if len(wrong_pool) > 1:
                    for local_idx, global_idx in enumerate(subset_indices):
                        if chosen_wrong[local_idx] == global_idx:
                            alternatives = wrong_pool[wrong_pool != global_idx]
                            chosen_wrong[local_idx] = int(alternatives[0])
                wrong_centroids = run.centroids[chosen_wrong]
            else:
                wrong_centroids = None

            for mode_name, injected in (
                ("clean", None),
                ("self_centroid", own_centroids),
                ("shuffled_centroid", shuffled_centroids),
                ("wrong_centroid", wrong_centroids),
            ):
                if injected is None and mode_name != "clean":
                    continue
                summary = runtime.run_generation(
                    questions=subset_questions,
                    answers=subset_answers,
                    answer_type=answer_type,
                    injected_latents=injected,
                )
                rows.append(
                    {
                        "run": run.label,
                        "mode": mode_name,
                        "accuracy": summary["accuracy"],
                        "avg_output_tokens": summary["avg_output_tokens"],
                        "num_samples": summary["num_samples"],
                    }
                )
        finally:
            runtime.close()

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    clean_acc = result[result["mode"] == "clean"][["run", "accuracy"]].rename(columns={"accuracy": "clean_accuracy"})
    result = result.merge(clean_acc, on="run", how="left")
    result["delta_accuracy"] = result["accuracy"] - result["clean_accuracy"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze centroid-based trajectory anchors from saved GSM8K latent trajectories."
    )
    parser.add_argument("--dataset", default="gsm8k", help="Dataset name. Default: gsm8k")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run directory, optionally labeled as label=/abs/path/to/run_0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "centroid_reference",
        help="Output directory for CSV/plots/markdown tables.",
    )
    parser.add_argument("--num-length-bins", type=int, default=5, help="Number of quantile bins for length plots.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for intervention diagnostics.")
    parser.add_argument(
        "--length-metric",
        choices=["auto", "token", "word", "char"],
        default="auto",
        help="Question length metric used for robustness summaries.",
    )

    parser.add_argument(
        "--model-name-or-path",
        action="append",
        default=[],
        help="Optional model path for model-backed probing, optionally labeled as run_label=/path/to/model.",
    )
    parser.add_argument(
        "--ckpt-dir",
        action="append",
        default=[],
        help="Optional checkpoint dir for model-backed probing, optionally labeled as run_label=/path/to/checkpoint.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Batch size for question representation probing.")
    parser.add_argument("--intervention-batch-size", type=int, default=16, help="Batch size for centroid injection runs.")
    parser.add_argument("--model-max-length", type=int, default=28000)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--use-prj", type=str2bool, default=True)
    parser.add_argument("--prj-dim", type=int, default=2048)
    parser.add_argument("--prj-no-ln", type=str2bool, default=False)
    parser.add_argument("--prj-dropout", type=float, default=0.0)
    parser.add_argument("--remove-eos", type=str2bool, default=True)
    parser.add_argument("--greedy", type=str2bool, default=True)
    parser.add_argument(
        "--auto-model-probe",
        action="store_true",
        help="Auto-resolve local model/checkpoint paths from the run layout and enable model-backed probing.",
    )

    parser.add_argument(
        "--run-intervention",
        action="store_true",
        help="Run model-backed inference-time centroid injection diagnostics.",
    )
    parser.add_argument(
        "--intervention-samples",
        type=int,
        default=256,
        help="Number of examples for inference-time intervention. Use <=0 for all samples.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = canonicalize_dataset_name(args.dataset)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_paths = parse_labeled_paths(args.run, "run")
    runs = [load_run_artifacts(label, Path(path).resolve()) for label, path in run_paths.items()]

    dataset, dataset_config = load_dataset_by_name(args.dataset)
    questions, answers = prepare_questions_and_answers(dataset, dataset_config)
    if not runs:
        raise ValueError("No run directories were provided.")
    expected_samples = runs[0].latents.shape[0]
    if len(questions) != expected_samples:
        raise ValueError(
            f"Dataset size mismatch: dataset has {len(questions)} samples but run has {expected_samples}."
        )

    model_probe_enabled = bool(args.model_name_or_path or args.ckpt_dir or args.auto_model_probe or args.run_intervention)
    if model_probe_enabled:
        attach_model_backed_question_probing(runs, questions, args)

    question_word_lengths = np.asarray([len(question.split()) for question in questions], dtype=np.int32)
    question_char_lengths = np.asarray([len(question) for question in questions], dtype=np.int32)

    sample_frames = []
    for run in runs:
        frame = build_sample_frame(run, questions, question_word_lengths, question_char_lengths)
        sample_frames.append(frame)

    sample_frame = pd.concat(sample_frames, ignore_index=True)

    has_full_token_lengths = (
        "question_token_length" in sample_frame.columns
        and sample_frame["question_token_length"].notna().all()
    )

    if args.length_metric == "token" and not has_full_token_lengths:
        raise ValueError("Token length requested but token-based question probing was not available.")
    if args.length_metric == "auto":
        if has_full_token_lengths:
            length_col = "question_token_length"
        else:
            length_col = "question_word_length"
    elif args.length_metric == "token":
        length_col = "question_token_length"
    elif args.length_metric == "word":
        length_col = "question_word_length"
    else:
        length_col = "question_char_length"

    probe_summary = summarize_probe_metrics(sample_frame, length_col=length_col)
    probe_summary.to_csv(args.output_dir / "probe_summary.csv", index=False)
    sample_frame.to_csv(args.output_dir / "sample_level_metrics.csv", index=False)

    offline_intervention_frames = []
    for run in runs:
        offline_frame = summarize_offline_intervention(run, run.question_mean_repr, seed=args.seed)
        if not offline_frame.empty:
            offline_intervention_frames.append(offline_frame)
    offline_intervention_df = (
        pd.concat(offline_intervention_frames, ignore_index=True)
        if offline_intervention_frames
        else pd.DataFrame()
    )
    if not offline_intervention_df.empty:
        offline_intervention_df.to_csv(args.output_dir / "offline_intervention_summary.csv", index=False)

    intervention_df = pd.DataFrame()
    if args.run_intervention:
        intervention_df = run_model_interventions(
            runs=runs,
            questions=questions,
            answers=answers,
            answer_type=dataset_config["answer_type"],
            args=args,
        )
        if not intervention_df.empty:
            intervention_df.to_csv(args.output_dir / "model_intervention_summary.csv", index=False)

    plot_probing(
        sample_frame=sample_frame,
        length_col=length_col,
        output_path=args.output_dir / "centroid_probing_summary.png",
        num_length_bins=args.num_length_bins,
    )
    plot_offline_intervention(
        intervention_df=offline_intervention_df,
        output_path=args.output_dir / "centroid_offline_intervention.png",
    )
    plot_model_intervention(
        intervention_df=intervention_df,
        output_path=args.output_dir / "centroid_injection_intervention.png",
    )

    summary_lines = [
        "# Centroid Reference Analysis",
        "",
        "This report uses the wording `trajectory-level global reference / geometric anchor`.",
        "",
        "## Probing Summary",
        "",
        dataframe_to_markdown(probe_summary),
    ]
    if not offline_intervention_df.empty:
        summary_lines.extend(
            [
                "",
                "## Offline Intervention Summary",
                "",
                dataframe_to_markdown(offline_intervention_df),
            ]
        )
    if not intervention_df.empty:
        summary_lines.extend(
            [
                "",
                "## Inference-Time Centroid Injection Summary",
                "",
                dataframe_to_markdown(intervention_df),
            ]
        )

    (args.output_dir / "CENTROID_REFERENCE_SUMMARY.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print(f"[Done] Wrote analysis artifacts to: {args.output_dir}")
    print(f"[Done] Probe summary: {args.output_dir / 'probe_summary.csv'}")
    if not offline_intervention_df.empty:
        print(f"[Done] Offline intervention summary: {args.output_dir / 'offline_intervention_summary.csv'}")
    if not intervention_df.empty:
        print(f"[Done] Model intervention summary: {args.output_dir / 'model_intervention_summary.csv'}")


if __name__ == "__main__":
    main()
