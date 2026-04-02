#!/usr/bin/env python3
"""Validate or refresh the HPC4 AIStudio token and emit reusable shell exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aistudio_auth import DEFAULT_BASE_URL, get_or_refresh_token, shell_exports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prefer the current HPC4 AIStudio token when it is still valid, "
            "otherwise log in directly through the HPC4 portal with "
            "hpc4user/hpc4password, and emit updated exports."
        )
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"HPC4 portal base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Print shell export commands for HPC4_AISTUDIO_TOKEN and related variables.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a human summary.",
    )
    parser.add_argument(
        "--env-file",
        help="Optional path to a shell snippet file that should be overwritten with export commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = get_or_refresh_token(args.base_url)
    exports = shell_exports(result)

    if args.env_file:
        env_path = Path(args.env_file)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(exports + "\n")

    if args.shell:
        print(exports)
    elif args.json:
        print(
            json.dumps(
                {
                    "base_url": result.base_url,
                    "source": result.source,
                    "end_org": result.end_org,
                    "has_ticket": bool(result.ticket),
                    "token_length": len(result.token),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            f"Auth source: {result.source}. "
            "Use --shell to emit export commands or --env-file to write a sourceable env file."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
