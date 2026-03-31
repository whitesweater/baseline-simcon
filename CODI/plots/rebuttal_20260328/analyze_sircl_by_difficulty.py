"""
Analyze SIRCL success/failure stratified by GSM8K problem difficulty.

Difficulty proxy: number of reasoning steps (lines in CoT before ####).
Cross-references with per-sample trajectory geometry and correctness.

Usage:
    cd /data/yhao/baseline
    python CODI/plots/rebuttal_20260328/analyze_sircl_by_difficulty.py
"""

import re
import json
import csv
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

# ── Paths ──
BASE = Path("/data/yhao/baseline")
SAMPLE_METRICS = BASE / "CODI/plots/rebuttal_20260328/02_centroid_reference/results_t6/sample_level_metrics.csv"
LATENT_ROOT = BASE / "CODI/results/latent_sweep_gsm8k/latent_6/models"
OUT_DIR = BASE / "CODI/plots/rebuttal_20260328/06_difficulty_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load GSM8K test set and count reasoning steps ──
print("Loading GSM8K test set...")
try:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
except Exception:
    # Fallback: try from HF cache or mirror
    import subprocess
    ds = None

if ds is None:
    print("ERROR: Could not load GSM8K. Trying offline cache...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")

def count_reasoning_steps(answer_text: str) -> int:
    """Count reasoning steps = non-empty lines before ####."""
    parts = answer_text.split("####")
    if len(parts) < 2:
        return 0
    reasoning = parts[0]
    steps = [s.strip() for s in reasoning.split("\n") if s.strip()]
    return len(steps)

def count_calculations(answer_text: str) -> int:
    """Count <<...>> calculator annotations."""
    return len(re.findall(r'<<[^>]+>>', answer_text))

def extract_final_answer(answer_text: str) -> str:
    """Extract numeric answer after ####."""
    parts = answer_text.split("####")
    if len(parts) < 2:
        return ""
    return parts[1].strip().replace(",", "")

gsm8k_meta = []
for i, sample in enumerate(ds):
    answer = sample["answer"]
    gsm8k_meta.append({
        "idx": i,
        "question": sample["question"],
        "ground_truth": extract_final_answer(answer),
        "num_steps": count_reasoning_steps(answer),
        "num_calcs": count_calculations(answer),
        "question_words": len(sample["question"].split()),
    })

num_steps_arr = np.array([m["num_steps"] for m in gsm8k_meta])
print(f"GSM8K test: {len(gsm8k_meta)} samples")
print(f"  Steps: mean={num_steps_arr.mean():.2f}, std={num_steps_arr.std():.2f}, "
      f"min={num_steps_arr.min()}, max={num_steps_arr.max()}")
print(f"  Distribution: " + ", ".join(
    f"{k}步={v}" for k, v in sorted(
        zip(*np.unique(num_steps_arr, return_counts=True)),
    )
))

# ── 2. Load per-sample predictions from each model ──
print("\nLoading per-model predictions...")

models = ["simcon", "simcon_sircl", "codi", "codi_sircl"]
predictions = {}

for model in models:
    pred_path = LATENT_ROOT / model / "gsm8k" / "run_0" / "predictions.json"
    if pred_path.exists():
        with open(pred_path) as f:
            data = json.load(f)
        predictions[model] = data
        print(f"  {model}: {len(data['predictions'])} predictions")
    else:
        print(f"  {model}: predictions.json NOT FOUND at {pred_path}")

# ── 3. Load per-sample metrics (trajectory geometry) ──
print("\nLoading sample-level trajectory metrics...")
sample_metrics = defaultdict(dict)  # {(run, idx): {metric: value}}

with open(SAMPLE_METRICS) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["run"], int(row["sample_idx"]))
        sample_metrics[key] = {
            "radius_mean": float(row["radius_mean"]),
            "radius_std": float(row["radius_std"]),
            "radius_max": float(row["radius_max"]),
            "trajectory_path_length": float(row["trajectory_path_length"]),
            "cos_mu_z1": float(row["cos_mu_z1"]),
            "cos_mu_final": float(row["cos_mu_final"]),
            "correct": row["correct"] == "True",
        }

print(f"  Loaded {len(sample_metrics)} (run, sample) entries")

# ── 4. Build unified per-sample table ──
def is_correct(pred_data, idx):
    """Check if prediction matches ground truth."""
    if pred_data is None or idx >= len(pred_data["predictions"]):
        return None
    pred = str(pred_data["predictions"][idx]).strip()
    gt = str(pred_data["ground_truth"][idx]).strip()
    try:
        return abs(float(pred) - float(gt)) < 1e-3
    except (ValueError, TypeError):
        return pred == gt

# Difficulty bins
def difficulty_bin(num_steps):
    if num_steps <= 2:
        return "easy (1-2 steps)"
    elif num_steps <= 4:
        return "medium (3-4 steps)"
    elif num_steps <= 6:
        return "hard (5-6 steps)"
    else:
        return "very hard (7+ steps)"

unified = []
for i, meta in enumerate(gsm8k_meta):
    row = {
        "idx": i,
        "num_steps": meta["num_steps"],
        "num_calcs": meta["num_calcs"],
        "question_words": meta["question_words"],
        "difficulty": difficulty_bin(meta["num_steps"]),
    }

    for model in models:
        if model in predictions:
            row[f"{model}_correct"] = is_correct(predictions[model], i)

        key = (model, i)
        if key in sample_metrics:
            for metric in ["radius_mean", "trajectory_path_length", "cos_mu_z1"]:
                row[f"{model}_{metric}"] = sample_metrics[key].get(metric)

    unified.append(row)

# ── 5. Analyze SIRCL effect by difficulty ──
print("\n" + "="*80)
print("SIRCL EFFECT BY DIFFICULTY TIER")
print("="*80)

# For each backbone pair (base, +sircl)
pairs = [("simcon", "simcon_sircl"), ("codi", "codi_sircl")]

results_by_difficulty = {}

for base, sircl in pairs:
    print(f"\n{'─'*60}")
    print(f"  {base.upper()} → +SIRCL")
    print(f"{'─'*60}")

    if base not in predictions or sircl not in predictions:
        print(f"  SKIPPED (missing predictions)")
        continue

    bins = defaultdict(lambda: {
        "count": 0,
        "base_correct": 0, "sircl_correct": 0,
        "recovered": 0, "regressed": 0, "both_correct": 0, "both_wrong": 0,
        "base_radius": [], "sircl_radius": [],
        "base_path_len": [], "sircl_path_len": [],
        "recovered_radius": [], "regressed_radius": [],
        "recovered_steps": [], "regressed_steps": [],
    })

    for row in unified:
        bc = row.get(f"{base}_correct")
        sc = row.get(f"{sircl}_correct")
        if bc is None or sc is None:
            continue

        d = row["difficulty"]
        b = bins[d]
        b["count"] += 1
        b["base_correct"] += int(bc)
        b["sircl_correct"] += int(sc)

        if bc and sc:
            b["both_correct"] += 1
        elif not bc and sc:
            b["recovered"] += 1
            b["recovered_steps"].append(row["num_steps"])
        elif bc and not sc:
            b["regressed"] += 1
            b["regressed_steps"].append(row["num_steps"])
        else:
            b["both_wrong"] += 1

        # Trajectory geometry
        br = row.get(f"{base}_radius_mean")
        sr = row.get(f"{sircl}_radius_mean")
        if br is not None:
            b["base_radius"].append(br)
        if sr is not None:
            b["sircl_radius"].append(sr)

        bp = row.get(f"{base}_trajectory_path_length")
        sp = row.get(f"{sircl}_trajectory_path_length")
        if bp is not None:
            b["base_path_len"].append(bp)
        if sp is not None:
            b["sircl_path_len"].append(sp)

        if not bc and sc and br is not None:
            b["recovered_radius"].append(br)
        if bc and not sc and br is not None:
            b["regressed_radius"].append(br)

    print(f"\n  {'Difficulty':<22} {'N':>5} {'Base%':>7} {'SIRCL%':>7} {'Δ':>7} "
          f"{'Recov':>6} {'Regr':>6} {'NetFlip':>8} {'Base r̄':>8} {'SIRCL r̄':>8}")
    print(f"  {'─'*22} {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")

    for d in ["easy (1-2 steps)", "medium (3-4 steps)", "hard (5-6 steps)", "very hard (7+ steps)"]:
        b = bins[d]
        if b["count"] == 0:
            continue
        n = b["count"]
        base_acc = b["base_correct"] / n * 100
        sircl_acc = b["sircl_correct"] / n * 100
        delta = sircl_acc - base_acc
        br = np.mean(b["base_radius"]) if b["base_radius"] else float('nan')
        sr = np.mean(b["sircl_radius"]) if b["sircl_radius"] else float('nan')

        print(f"  {d:<22} {n:>5} {base_acc:>6.1f}% {sircl_acc:>6.1f}% {delta:>+6.1f}% "
              f"{b['recovered']:>6} {b['regressed']:>6} {b['recovered']-b['regressed']:>+8} "
              f"{br:>8.2f} {sr:>8.2f}")

    results_by_difficulty[base] = bins

# ── 6. Answer magnitude analysis ──
print("\n" + "="*80)
print("SIRCL EFFECT BY ANSWER MAGNITUDE")
print("="*80)

def answer_magnitude_bin(gt_str):
    try:
        val = abs(float(gt_str))
    except (ValueError, TypeError):
        return "non-numeric"
    if val <= 10:
        return "tiny (≤10)"
    elif val <= 100:
        return "small (11-100)"
    elif val <= 1000:
        return "medium (101-1K)"
    else:
        return "large (>1K)"

for base, sircl in pairs:
    if base not in predictions or sircl not in predictions:
        continue

    print(f"\n  {base.upper()} → +SIRCL")

    mag_bins = defaultdict(lambda: {"count": 0, "base_correct": 0, "sircl_correct": 0,
                                      "recovered": 0, "regressed": 0})

    for i, meta in enumerate(gsm8k_meta):
        bc = is_correct(predictions[base], i)
        sc = is_correct(predictions[sircl], i)
        if bc is None or sc is None:
            continue

        mag = answer_magnitude_bin(meta["ground_truth"])
        mb = mag_bins[mag]
        mb["count"] += 1
        mb["base_correct"] += int(bc)
        mb["sircl_correct"] += int(sc)
        if not bc and sc:
            mb["recovered"] += 1
        if bc and not sc:
            mb["regressed"] += 1

    print(f"  {'Magnitude':<22} {'N':>5} {'Base%':>7} {'SIRCL%':>7} {'Δ':>7} {'Recov':>6} {'Regr':>6}")
    print(f"  {'─'*22} {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*6}")

    for mag in ["tiny (≤10)", "small (11-100)", "medium (101-1K)", "large (>1K)", "non-numeric"]:
        mb = mag_bins[mag]
        if mb["count"] == 0:
            continue
        n = mb["count"]
        base_acc = mb["base_correct"] / n * 100
        sircl_acc = mb["sircl_correct"] / n * 100
        delta = sircl_acc - base_acc
        print(f"  {mag:<22} {n:>5} {base_acc:>6.1f}% {sircl_acc:>6.1f}% {delta:>+6.1f}% "
              f"{mb['recovered']:>6} {mb['regressed']:>6}")

# ── 7. Question length analysis ──
print("\n" + "="*80)
print("SIRCL EFFECT BY QUESTION LENGTH")
print("="*80)

def length_bin(wc):
    if wc <= 30:
        return "short (≤30w)"
    elif wc <= 50:
        return "medium (31-50w)"
    elif wc <= 70:
        return "long (51-70w)"
    else:
        return "very long (>70w)"

for base, sircl in pairs:
    if base not in predictions or sircl not in predictions:
        continue

    print(f"\n  {base.upper()} → +SIRCL")

    len_bins = defaultdict(lambda: {"count": 0, "base_correct": 0, "sircl_correct": 0,
                                      "recovered": 0, "regressed": 0})

    for i, meta in enumerate(gsm8k_meta):
        bc = is_correct(predictions[base], i)
        sc = is_correct(predictions[sircl], i)
        if bc is None or sc is None:
            continue

        lb = length_bin(meta["question_words"])
        b = len_bins[lb]
        b["count"] += 1
        b["base_correct"] += int(bc)
        b["sircl_correct"] += int(sc)
        if not bc and sc:
            b["recovered"] += 1
        if bc and not sc:
            b["regressed"] += 1

    print(f"  {'Length':<22} {'N':>5} {'Base%':>7} {'SIRCL%':>7} {'Δ':>7} {'Recov':>6} {'Regr':>6}")
    print(f"  {'─'*22} {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*6}")

    for lb in ["short (≤30w)", "medium (31-50w)", "long (51-70w)", "very long (>70w)"]:
        b = len_bins[lb]
        if b["count"] == 0:
            continue
        n = b["count"]
        base_acc = b["base_correct"] / n * 100
        sircl_acc = b["sircl_correct"] / n * 100
        delta = sircl_acc - base_acc
        print(f"  {lb:<22} {n:>5} {base_acc:>6.1f}% {sircl_acc:>6.1f}% {delta:>+6.1f}% "
              f"{b['recovered']:>6} {b['regressed']:>6}")

# ── 8. Cross-analysis: steps × trajectory geometry ──
print("\n" + "="*80)
print("TRAJECTORY GEOMETRY BY DIFFICULTY (SIMCON BASELINE)")
print("="*80)

if "simcon" in predictions:
    print(f"\n  {'Difficulty':<22} {'N':>5} {'Acc%':>6} {'r̄_mean':>8} {'r̄_std':>8} "
          f"{'PathLen':>8} {'cos(μ,z1)':>10}")
    print(f"  {'─'*22} {'─'*5} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

    for d in ["easy (1-2 steps)", "medium (3-4 steps)", "hard (5-6 steps)", "very hard (7+ steps)"]:
        items = [r for r in unified if r["difficulty"] == d and r.get("simcon_correct") is not None]
        if not items:
            continue
        n = len(items)
        acc = sum(1 for r in items if r["simcon_correct"]) / n * 100
        rm = np.mean([r["simcon_radius_mean"] for r in items if r.get("simcon_radius_mean") is not None])
        pl = np.mean([r["simcon_trajectory_path_length"] for r in items if r.get("simcon_trajectory_path_length") is not None])
        cm = np.mean([r["simcon_cos_mu_z1"] for r in items if r.get("simcon_cos_mu_z1") is not None])

        # Also get per-correct/wrong breakdown
        correct_items = [r for r in items if r["simcon_correct"] and r.get("simcon_radius_mean") is not None]
        wrong_items = [r for r in items if not r["simcon_correct"] and r.get("simcon_radius_mean") is not None]

        print(f"  {d:<22} {n:>5} {acc:>5.1f}% {rm:>8.2f} {'':>8} {pl:>8.1f} {cm:>10.4f}")

        if correct_items:
            rm_c = np.mean([r["simcon_radius_mean"] for r in correct_items])
            pl_c = np.mean([r["simcon_trajectory_path_length"] for r in correct_items])
            print(f"    → correct           {len(correct_items):>5}        {rm_c:>8.2f} {'':>8} {pl_c:>8.1f}")
        if wrong_items:
            rm_w = np.mean([r["simcon_radius_mean"] for r in wrong_items])
            pl_w = np.mean([r["simcon_trajectory_path_length"] for r in wrong_items])
            print(f"    → wrong             {len(wrong_items):>5}        {rm_w:>8.2f} {'':>8} {pl_w:>8.1f}")

# ── 9. Save CSV ──
out_csv = OUT_DIR / "sircl_by_difficulty.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "idx", "num_steps", "num_calcs", "question_words", "difficulty", "answer_magnitude",
        "simcon_correct", "simcon_sircl_correct", "simcon_transition",
        "codi_correct", "codi_sircl_correct", "codi_transition",
        "simcon_radius_mean", "simcon_sircl_radius_mean",
        "simcon_path_length", "simcon_sircl_path_length",
    ])
    for i, row in enumerate(unified):
        sc_base = row.get("simcon_correct")
        sc_sircl = row.get("simcon_sircl_correct")
        cd_base = row.get("codi_correct")
        cd_sircl = row.get("codi_sircl_correct")

        def transition(b, s):
            if b is None or s is None:
                return ""
            if b and s:
                return "both_correct"
            elif not b and s:
                return "recovered"
            elif b and not s:
                return "regressed"
            else:
                return "both_wrong"

        writer.writerow([
            i, row["num_steps"], row["num_calcs"], row["question_words"],
            row["difficulty"], answer_magnitude_bin(gsm8k_meta[i]["ground_truth"]),
            sc_base, sc_sircl, transition(sc_base, sc_sircl),
            cd_base, cd_sircl, transition(cd_base, cd_sircl),
            row.get("simcon_radius_mean", ""),
            row.get("simcon_sircl_radius_mean", ""),
            row.get("simcon_trajectory_path_length", ""),
            row.get("simcon_sircl_trajectory_path_length", ""),
        ])

print(f"\nSaved detailed CSV to {out_csv}")

# ── 10. Summary statistics ──
print("\n" + "="*80)
print("KEY FINDINGS SUMMARY")
print("="*80)

for base, sircl in pairs:
    if base not in predictions or sircl not in predictions:
        continue

    print(f"\n{base.upper()} → +SIRCL:")

    # Recovery rate by step count
    step_recovery = defaultdict(lambda: {"total_wrong": 0, "recovered": 0})
    step_regression = defaultdict(lambda: {"total_correct": 0, "regressed": 0})

    for i, meta in enumerate(gsm8k_meta):
        bc = is_correct(predictions[base], i)
        sc = is_correct(predictions[sircl], i)
        if bc is None or sc is None:
            continue

        steps = meta["num_steps"]
        if not bc:
            step_recovery[steps]["total_wrong"] += 1
            if sc:
                step_recovery[steps]["recovered"] += 1
        else:
            step_regression[steps]["total_correct"] += 1
            if not sc:
                step_regression[steps]["regressed"] += 1

    print(f"\n  Recovery rate by exact step count:")
    print(f"  {'Steps':>5} {'Wrong':>6} {'Recov':>6} {'Rate':>7}")
    for s in sorted(step_recovery.keys()):
        d = step_recovery[s]
        rate = d["recovered"] / d["total_wrong"] * 100 if d["total_wrong"] > 0 else 0
        print(f"  {s:>5} {d['total_wrong']:>6} {d['recovered']:>6} {rate:>6.1f}%")

    print(f"\n  Regression rate by exact step count:")
    print(f"  {'Steps':>5} {'Corr':>6} {'Regr':>6} {'Rate':>7}")
    for s in sorted(step_regression.keys()):
        d = step_regression[s]
        rate = d["regressed"] / d["total_correct"] * 100 if d["total_correct"] > 0 else 0
        print(f"  {s:>5} {d['total_correct']:>6} {d['regressed']:>6} {rate:>6.1f}%")

print("\nDone.")
