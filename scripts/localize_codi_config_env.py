#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict


DEFAULT_HPC2_CACHE_ROOT = Path("/hpc2hdd/home/yhao481/jhupload/cache")
STAGE_TAG = "multimodel_gsm8k_math500_aime_v1"
RUN_TAG = "rebuttal_20260325"


def build_hpc2_values(repo_root: Path, cache_root: Path) -> Dict[str, str]:
    run_root = repo_root / "CODI_rebuttal_runs" / RUN_TAG
    stage_root = run_root / STAGE_TAG
    models_root = cache_root / "Models"
    return {
        "CODI_RUN_ROOT": str(run_root),
        "CODI_SAVE_DIR": "${CODI_RUN_ROOT}/outputs",
        "CODI_RESULT_DIR": "${CODI_RUN_ROOT}/results",
        "CODI_VENV_PATH": str(repo_root / ".venv" / "bin" / "activate"),
        "CODI_MULTIARITH_PATH": str(repo_root / "CODI" / "local_datasets" / "multiarith" / "train_42.json"),
        "CODI_SVAMP_PATH": str(repo_root / "CODI" / "local_datasets" / "svamp" / "train_42.json"),
        "CODI_COIN_FLIP_PATH": str(repo_root / "CODI" / "local_datasets" / "coin_flip" / "train_42.json"),
        "CODI_CACHE_DIR": str(stage_root / "cache"),
        "CODI_LLAMA1B_PATH": str(models_root / "Llama-3.2-1B-Instruct"),
        "CODI_LLAMA3B_PATH": str(models_root / "Llama-3.2-3B-Instruct"),
        "CODI_LLAMA8B_PATH": str(models_root / "Meta-Llama-3.1-8B-Instruct"),
        "CODI_QWEN_PATH": str(models_root / "Qwen3-4B"),
    }


def replace_exports(config_text: str, values: Dict[str, str]) -> str:
    updated = config_text
    for key, value in values.items():
        pattern = re.compile(rf"^export {re.escape(key)}=.*$", re.MULTILINE)
        replacement = f'export {key}="{value}"'
        if pattern.search(updated):
            updated = pattern.sub(replacement, updated, count=1)
        else:
            if not updated.endswith("\n"):
                updated += "\n"
            updated += replacement + "\n"
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Localize CODI/config.env for a target cluster copy.")
    parser.add_argument("--repo-root", required=True, help="Target repo root on the destination machine.")
    parser.add_argument("--cluster", choices=["hpc2"], default="hpc2")
    parser.add_argument("--cache-root", default=str(DEFAULT_HPC2_CACHE_ROOT))
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    config_path = repo_root / "CODI" / "config.env"

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    if args.cluster != "hpc2":
        raise ValueError(f"Unsupported cluster: {args.cluster}")

    values = build_hpc2_values(repo_root, cache_root)
    updated_text = replace_exports(config_path.read_text(encoding="utf-8"), values)
    config_path.write_text(updated_text, encoding="utf-8")

    print(f"[localize] updated {config_path}")
    for key in values:
        print(f"[localize] {key}={values[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
