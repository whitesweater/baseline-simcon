#!/usr/bin/env python3
"""Print AIStudio NPU pool choices for user selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_POOL = Path.home() / ".codex" / "tmp" / "hpc4_runtime" / "npu_pool.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List running NPU container choices from npu_pool.json.")
    parser.add_argument(
        "--pool-file",
        default=str(DEFAULT_POOL),
        help=f"Pool file path. Default: {DEFAULT_POOL}",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run id. If set, print only that entry as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool_file = Path(args.pool_file)
    data = json.loads(pool_file.read_text())
    items = data.get("items", [])

    if args.run_id:
        for item in items:
            if item.get("run_id") == args.run_id:
                print(json.dumps(item, indent=2, ensure_ascii=False))
                return 0
        raise SystemExit(f"run id not found: {args.run_id}")

    for index, item in enumerate(items, start=1):
        npu = item.get("npu") or {}
        image = item.get("image") or {}
        ssh = item.get("ssh") or {}
        print(
            f"[{index}] {item.get('name')} | "
            f"{npu.get('device_num')} NPU | "
            f"{npu.get('series') or npu.get('device_type')} | "
            f"{image.get('name')} | "
            f"run_id={item.get('run_id')} | "
            f"{ssh.get('command')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
