#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope import snapshot_download as ms_snapshot_download


MODEL_SPECS = {
    "llama1b": {
        "dest_name": "Llama-3.2-1B-Instruct",
        "local_env": "CODI_LLAMA1B_PATH",
        "modelscope_id": "LLM-Research/Llama-3.2-1B-Instruct",
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
    },
    "llama3b": {
        "dest_name": "Llama-3.2-3B-Instruct",
        "local_env": "CODI_LLAMA3B_PATH",
        "modelscope_id": "LLM-Research/Llama-3.2-3B-Instruct",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "llama8b": {
        "dest_name": "Meta-Llama-3.1-8B-Instruct",
        "local_env": "CODI_LLAMA8B_PATH",
        "modelscope_id": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "qwen3": {
        "dest_name": "Qwen3-4B",
        "local_env": "CODI_QWEN_PATH",
        "modelscope_id": "Qwen/Qwen3-4B",
        "hf_id": "Qwen/Qwen3-4B",
    },
}

DATASET_SPECS = {
    "gsm8k": {"hf_id": "zen-E/GSM8k-Aug", "split": "test"},
    "math500": {"hf_id": "HuggingFaceH4/MATH-500", "split": "test"},
    "aime": {"hf_id": "HuggingFaceH4/aime_2024", "split": "train"},
}


def model_is_ready(model_dir: Path) -> bool:
    required = model_dir / "config.json"
    if not required.exists():
        return False
    candidates = [
        model_dir / "model.safetensors",
        model_dir / "model.safetensors.index.json",
        model_dir / "pytorch_model.bin",
    ]
    return any(path.exists() for path in candidates)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_manifest(manifest_path: Path, payload: dict) -> None:
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def remove_existing_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def download_model(model_key: str, dest_root: Path, manifest_root: Path, backend: str) -> None:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model key: {model_key}")

    spec = MODEL_SPECS[model_key]
    model_dir = dest_root / spec["dest_name"]
    manifest_path = manifest_root / f"{spec['dest_name']}.manifest.json"

    if model_is_ready(model_dir):
        print(f"[skip] model already exists: {model_dir}")
        write_manifest(
            manifest_path,
            {"model_key": model_key, "backend": "existing", "path": str(model_dir)},
        )
        return

    local_env = spec.get("local_env")
    local_path_value = os.environ.get(local_env, "") if local_env else ""
    local_model_dir = Path(local_path_value).expanduser() if local_path_value else None
    if local_model_dir and model_is_ready(local_model_dir):
        if model_dir.exists() or model_dir.is_symlink():
            remove_existing_path(model_dir)
        print(f"[link] local-existing -> {local_model_dir} -> {model_dir}")
        os.symlink(local_model_dir, model_dir, target_is_directory=True)
        write_manifest(
            manifest_path,
            {
                "model_key": model_key,
                "backend": "local-existing",
                "path": str(model_dir),
                "source_path": str(local_model_dir),
                "source_env": local_env,
            },
        )
        return

    if local_model_dir:
        print(f"[warn] local model path is not ready, will continue with remote download: {local_model_dir}")

    if model_dir.exists() and not model_dir.is_dir():
        remove_existing_path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if backend == "modelscope":
        repo_id = spec["modelscope_id"]
        print(f"[download] modelscope -> {repo_id} -> {model_dir}")
        ms_snapshot_download(model_id=repo_id, local_dir=str(model_dir), local_files_only=False)
    elif backend in {"hf-mirror", "hf"}:
        repo_id = spec["hf_id"]
        endpoint = os.environ.get("HF_ENDPOINT") if backend == "hf-mirror" else None
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        print(f"[download] {backend} -> {repo_id} -> {model_dir}")
        hf_snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            endpoint=endpoint,
            token=token,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if not model_is_ready(model_dir):
        raise RuntimeError(f"Model download did not produce a complete directory: {model_dir}")

    write_manifest(
        manifest_path,
        {"model_key": model_key, "backend": backend, "path": str(model_dir)},
    )


def warm_dataset(dataset_key: str, manifest_root: Path | None = None) -> None:
    if dataset_key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    spec = DATASET_SPECS[dataset_key]
    repo_id = spec["hf_id"]
    split = spec.get("split")
    print(f"[dataset] warming cache for {dataset_key}: {repo_id} [{split}]")
    load_dataset(repo_id, split=split)
    if manifest_root is not None:
        write_manifest(
            manifest_root / f"{dataset_key}.manifest.json",
            {"dataset_key": dataset_key, "hf_id": repo_id, "split": split},
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare stage-specific models or warm dataset caches.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_parser = subparsers.add_parser("models")
    model_parser.add_argument("--backend", required=True, choices=["modelscope", "hf-mirror", "hf"])
    model_parser.add_argument("--dest-root", required=True)
    model_parser.add_argument("--manifest-root")
    model_parser.add_argument("--models", nargs="+", required=True)

    dataset_parser = subparsers.add_parser("datasets")
    dataset_parser.add_argument("--manifest-root")
    dataset_parser.add_argument("--datasets", nargs="+", required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "models":
        dest_root = Path(args.dest_root)
        dest_root.mkdir(parents=True, exist_ok=True)
        manifest_root = Path(args.manifest_root).expanduser() if args.manifest_root else dest_root
        manifest_root.mkdir(parents=True, exist_ok=True)
        for model_key in args.models:
            download_model(model_key, dest_root, manifest_root, args.backend)
    elif args.command == "datasets":
        manifest_root = Path(args.manifest_root).expanduser() if args.manifest_root else None
        if manifest_root is not None:
            manifest_root.mkdir(parents=True, exist_ok=True)
        for dataset_key in args.datasets:
            warm_dataset(dataset_key, manifest_root=manifest_root)


if __name__ == "__main__":
    main()
