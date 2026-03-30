#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

from datasets import load_dataset


CODI_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GSM8K_AUG_CACHE_DIR = Path("/data/yhao/hf_datasets_cache")

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


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
    "gsm8k": {"hf_id": "zen-E/GSM8k-Aug", "split": "all"},
    "math500": {"hf_id": "HuggingFaceH4/MATH-500", "split": "test"},
    "aime": {"hf_id": "HuggingFaceH4/aime_2024", "split": "train"},
    "svamp": {
        "local_path": "local_datasets/svamp/eval_42.json",
        "hf_id": "ChilleD/SVAMP",
        "split": "all",
    },
    "gsm-hard": {"hf_id": "juyoung-trl/gsm-hard", "split": "train"},
    "asdiv": {"hf_id": "EleutherAI/asdiv", "split": "validation"},
}

MODEL_DOWNLOAD_IGNORE_PATTERNS = ["original/*.pth"]


def get_ms_snapshot_download():
    try:
        from modelscope import snapshot_download as ms_snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Backend 'modelscope' requires the 'modelscope' package. "
            "Install it in the active environment or retry with hf-mirror/hf."
        ) from exc
    return ms_snapshot_download


def get_hf_snapshot_download():
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Backends 'hf-mirror' and 'hf' require the 'huggingface_hub' package."
        ) from exc
    return hf_snapshot_download


def model_is_ready(model_dir: Path) -> bool:
    required = model_dir / "config.json"
    if not required.exists():
        return False
    if (model_dir / "model.safetensors").exists():
        return True
    if (model_dir / "pytorch_model.bin").exists():
        return True

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return False

    try:
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    shard_names = sorted(set((index_payload.get("weight_map") or {}).values()))
    if not shard_names:
        return False
    return all((model_dir / shard_name).exists() for shard_name in shard_names)


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


def resolve_local_dataset_path(local_path: str) -> Path:
    dataset_path = Path(local_path).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = CODI_DIR / dataset_path
    return dataset_path


def resolve_dataset_cache_dir(dataset_key: str) -> Path | None:
    if dataset_key == "gsm8k":
        cache_dir = (
            os.environ.get("CODI_GSM8K_AUG_CACHE_DIR")
            or os.environ.get("HF_DATASETS_CACHE")
            or str(DEFAULT_GSM8K_AUG_CACHE_DIR)
        )
    else:
        cache_dir = os.environ.get("HF_DATASETS_CACHE")

    if not cache_dir:
        return None

    cache_path = Path(cache_dir).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def resolve_external_model_dir(dest_root: Path, local_model_dir: Path | None) -> Path | None:
    if local_model_dir is None:
        return None
    try:
        if local_model_dir.resolve(strict=False) == dest_root.resolve(strict=False):
            return None
    except OSError:
        if local_model_dir == dest_root:
            return None
    return local_model_dir


def ensure_stage_model_link(stage_model_dir: Path, ready_model_dir: Path) -> None:
    if stage_model_dir.exists() or stage_model_dir.is_symlink():
        if stage_model_dir.is_symlink():
            current_target = stage_model_dir.resolve(strict=False)
            if current_target == ready_model_dir.resolve(strict=False):
                return
        remove_existing_path(stage_model_dir)
    os.symlink(ready_model_dir, stage_model_dir, target_is_directory=True)


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
    external_model_dir = resolve_external_model_dir(model_dir, local_model_dir)
    if local_model_dir and model_is_ready(local_model_dir):
        print(f"[link] local-existing -> {local_model_dir} -> {model_dir}")
        ensure_stage_model_link(model_dir, local_model_dir)
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

    download_dir = external_model_dir or model_dir

    if download_dir.exists() and not download_dir.is_dir():
        remove_existing_path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    if backend == "modelscope":
        repo_id = spec["modelscope_id"]
        print(f"[download] modelscope -> {repo_id} -> {download_dir}")
        ms_snapshot_download = get_ms_snapshot_download()
        ms_snapshot_download(
            model_id=repo_id,
            local_dir=str(download_dir),
            local_files_only=False,
            ignore_file_pattern=MODEL_DOWNLOAD_IGNORE_PATTERNS,
            ignore_patterns=MODEL_DOWNLOAD_IGNORE_PATTERNS,
        )
    elif backend in {"hf-mirror", "hf"}:
        repo_id = spec["hf_id"]
        endpoint = os.environ.get("HF_ENDPOINT") if backend == "hf-mirror" else None
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        print(f"[download] {backend} -> {repo_id} -> {download_dir}")
        hf_snapshot_download = get_hf_snapshot_download()
        hf_snapshot_download(
            repo_id=repo_id,
            local_dir=str(download_dir),
            endpoint=endpoint,
            token=token,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=MODEL_DOWNLOAD_IGNORE_PATTERNS,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if not model_is_ready(download_dir):
        raise RuntimeError(f"Model download did not produce a complete directory: {download_dir}")

    if external_model_dir:
        print(f"[link] stage-model -> {download_dir} -> {model_dir}")
        ensure_stage_model_link(model_dir, download_dir)

    write_manifest(
        manifest_path,
        {
            "model_key": model_key,
            "backend": backend,
            "path": str(model_dir),
            "download_path": str(download_dir),
        },
    )


def warm_dataset(dataset_key: str, manifest_root: Path | None = None) -> None:
    if dataset_key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    spec = DATASET_SPECS[dataset_key]
    manifest_payload = {"dataset_key": dataset_key}

    local_path_value = spec.get("local_path")
    if local_path_value:
        local_path = resolve_local_dataset_path(local_path_value)
        manifest_payload["local_path"] = str(local_path)
        if local_path.exists():
            print(f"[dataset] using local dataset for {dataset_key}: {local_path}")
            if local_path.suffix.lower() == ".json":
                with local_path.open("r", encoding="utf-8") as handle:
                    json.load(handle)
            manifest_payload["source"] = "local"
            if manifest_root is not None:
                write_manifest(manifest_root / f"{dataset_key}.manifest.json", manifest_payload)
            return
        print(f"[dataset] local dataset missing for {dataset_key}, fallback to Hugging Face: {local_path}")

    repo_id = spec.get("hf_id")
    if not repo_id:
        raise RuntimeError(f"Dataset {dataset_key} has no available local path or Hugging Face source")

    split = spec.get("split")
    hf_config = spec.get("hf_config")
    cache_dir = resolve_dataset_cache_dir(dataset_key)
    load_kwargs = {}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)
    print(f"[dataset] warming cache for {dataset_key}: {repo_id} [{split}]")
    if cache_dir is not None:
        print(f"[dataset] cache dir for {dataset_key}: {cache_dir}")

    if split == "all":
        if hf_config:
            dataset = load_dataset(repo_id, hf_config, **load_kwargs)
        else:
            dataset = load_dataset(repo_id, **load_kwargs)
        available_splits = list(dataset.keys())
        for split_name in available_splits:
            len(dataset[split_name])
        manifest_payload.update(
            {
                "source": "huggingface",
                "hf_id": repo_id,
                "split": split,
                "available_splits": available_splits,
            }
        )
    else:
        if hf_config:
            dataset = load_dataset(repo_id, hf_config, split=split, **load_kwargs)
        else:
            dataset = load_dataset(repo_id, split=split, **load_kwargs)
        len(dataset)
        manifest_payload.update(
            {
                "source": "huggingface",
                "hf_id": repo_id,
                "split": split,
            }
        )

    if hf_config:
        manifest_payload["hf_config"] = hf_config
    if cache_dir is not None:
        manifest_payload["cache_dir"] = str(cache_dir)

    if manifest_root is not None:
        write_manifest(manifest_root / f"{dataset_key}.manifest.json", manifest_payload)


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
