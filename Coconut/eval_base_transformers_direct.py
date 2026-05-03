#!/usr/bin/env python3
"""Direct base-model evaluation with plain transformers generation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASETS = {
    "gsm8k": "data/gsm8k_test.json",
    "gsm_hard": "data/gsm-hard_train.json",
    "multi_arith": "data/multi-arith_test.json",
    "svamp": "data/svamp_all.json",
    "asdiv": "data/asdiv_test.json",
    "math500": "data/math500_test.json",
    "aime": "data/aime_train.json",
}

DATASET_PRESETS = {
    "all": list(DATASETS),
    "basic": ["gsm8k", "gsm_hard", "svamp"],
    "hard": ["math500", "aime"],
}


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def expand_datasets(names: list[str]) -> list[str]:
    expanded: list[str] = []
    for name in names:
        expanded.extend(DATASET_PRESETS.get(name, [name]))
    unknown = [name for name in expanded if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
    deduped: list[str] = []
    seen: set[str] = set()
    for name in expanded:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        rows = rows[:limit]
    return rows


def normalize_answer(value: Any) -> str:
    text = str(value).replace(",", "").strip()
    if "####" in text:
        text = text.split("####")[-1].strip()
    return text


def extract_prediction(text: str) -> str:
    text = text.strip()
    if "###" in text:
        text = text.split("###")[-1].strip()
    elif "#" in text:
        text = text.split("#")[-1].strip()
    text = text.replace(",", "").strip()
    return text


def first_number(text: str) -> str | None:
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", text.replace(",", ""))
    if not match:
        return None
    value = match.group(0)
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number == int(number) else str(number)


def numeric_match(prediction: str, answer: str) -> bool:
    pred_num = first_number(prediction)
    answer_num = first_number(answer)
    return pred_num is not None and answer_num is not None and pred_num == answer_num


def max_new_tokens_for(dataset_name: str) -> int:
    return 256 if dataset_name in {"math500", "aime"} else 128


def model_context_limit(model: torch.nn.Module, tokenizer: Any) -> int | None:
    config = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        return tokenizer_limit
    return None


def make_prompt(question: Any) -> str:
    return f"{question}\n"


@torch.inference_mode()
def evaluate_dataset(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    dataset_name: str,
    dataset_path: Path,
    device: torch.device,
    rank: int,
    world_size: int,
    batch_size: int,
    limit: int | None,
    dtype: str,
) -> dict[str, Any]:
    rows = load_rows(dataset_path, limit)
    local_indices = list(range(rank, len(rows), world_size))
    context_limit = model_context_limit(model, tokenizer)
    max_new_tokens = max_new_tokens_for(dataset_name)
    max_input_length = None
    if context_limit is not None:
        max_input_length = max(1, context_limit - max_new_tokens)

    correct = 0
    numeric_correct = 0
    total = 0
    examples: list[dict[str, Any]] = []
    iterator = tqdm(
        range(0, len(local_indices), batch_size),
        desc=f"{dataset_name}",
        disable=rank != 0,
        dynamic_ncols=True,
    )

    for start in iterator:
        batch_indices = local_indices[start : start + batch_size]
        batch_rows = [rows[i] for i in batch_indices]
        prompts = [make_prompt(row["question"]) for row in batch_rows]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=max_input_length is not None,
            max_length=max_input_length,
            add_special_tokens=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(
            generated[:, input_width:],
            skip_special_tokens=True,
        )
        for row_index, row, text in zip(batch_indices, batch_rows, decoded):
            answer = normalize_answer(row["answer"])
            prediction = extract_prediction(text)
            is_correct = prediction == answer
            is_numeric_correct = numeric_match(prediction, answer)
            correct += int(is_correct)
            numeric_correct += int(is_numeric_correct)
            total += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "idx": row_index,
                        "question": str(row["question"]),
                        "answer": answer,
                        "prediction": prediction,
                        "raw_generation": text,
                        "exact": is_correct,
                        "numeric": is_numeric_correct,
                    }
                )
        if total:
            iterator.set_postfix(exact=f"{correct / total:.4f}", numeric=f"{numeric_correct / total:.4f}")

    stats = torch.tensor([correct, numeric_correct, total], device=device, dtype=torch.long)
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    gathered_examples: list[list[dict[str, Any]]] | None = None
    if world_size > 1:
        if rank == 0:
            gathered_examples = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.gather_object(examples, gathered_examples, dst=0)
    else:
        gathered_examples = [examples]

    all_examples: list[dict[str, Any]] = []
    if rank == 0 and gathered_examples is not None:
        for part in gathered_examples:
            all_examples.extend(part)
        all_examples = sorted(all_examples, key=lambda item: item["idx"])[:10]

    exact_correct, numeric_correct_all, total_all = [int(x) for x in stats.tolist()]
    return {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "correct": exact_correct,
        "numeric_correct": numeric_correct_all,
        "total": total_all,
        "accuracy": exact_correct / total_all if total_all else 0.0,
        "numeric_accuracy": numeric_correct_all / total_all if total_all else 0.0,
        "examples": all_examples if rank == 0 else [],
        "prompt": "question + '\\n'",
        "dtype": dtype,
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"direct_eval_{stamp}.json"
    csv_path = output_dir / f"direct_eval_{stamp}.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "accuracy",
                "correct",
                "total",
                "numeric_accuracy",
                "numeric_correct",
                "dataset_path",
            ],
        )
        writer.writeheader()
        for row in payload["results"]:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "accuracy": row["accuracy"],
                    "correct": row["correct"],
                    "total": row["total"],
                    "numeric_accuracy": row["numeric_accuracy"],
                    "numeric_correct": row["numeric_correct"],
                    "dataset_path": row["dataset_path"],
                }
            )
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def dtype_from_arg(value: str) -> torch.dtype | str | None:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    if value == "auto":
        return "auto"
    return None


def main() -> None:
    args = parse_args()
    rank, _local_rank, world_size, device = distributed_context()
    dataset_names = expand_datasets(args.datasets)
    data_root = Path(args.data_root)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_arg(args.dtype),
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    if rank == 0:
        print(f"model_name={args.model_name}")
        print(f"model_path={args.model_path}")
        print(f"datasets={dataset_names}")
        print(f"world_size={world_size} device={device}")

    results = []
    for name in dataset_names:
        result = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name=name,
            dataset_path=data_root / Path(DATASETS[name]).name,
            device=device,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            limit=args.limit,
            dtype=args.dtype,
        )
        if rank == 0:
            print(
                f"{name}: exact={result['accuracy']:.4f} "
                f"({result['correct']}/{result['total']}), "
                f"numeric={result['numeric_accuracy']:.4f} "
                f"({result['numeric_correct']}/{result['total']})"
            )
            results.append(result)

    if rank == 0:
        avg = sum(row["accuracy"] for row in results) / len(results)
        numeric_avg = sum(row["numeric_accuracy"] for row in results) / len(results)
        payload = {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "timestamp": datetime.now().isoformat(),
            "world_size": world_size,
            "batch_size": args.batch_size,
            "results": results,
            "average_accuracy": avg,
            "average_numeric_accuracy": numeric_avg,
        }
        json_path, csv_path = write_outputs(Path(args.output_dir), payload)
        print(f"average_exact={avg:.4f}")
        print(f"average_numeric={numeric_avg:.4f}")
        print(f"json={json_path}")
        print(f"csv={csv_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
