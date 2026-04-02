#!/usr/bin/env python3
"""Fetch HPC2 GPU container state through the validated remote VPN jump host."""

from __future__ import annotations

import argparse
import sys

from hpc2_remote_vpn import DEFAULT_REMOTE_HOST, DEFAULT_REMOTE_PROXY, run_remote_fetch


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run the HPC2 portal container query on the jump host through the validated SOCKS proxy."
    )
    parser.add_argument("--username", help="Portal username. Defaults to $HPC2_USER.")
    parser.add_argument(
        "--password-env",
        default="HPC2_PASS",
        help="Local environment variable that stores the HPC2 portal password.",
    )
    parser.add_argument(
        "--remote-host",
        default=DEFAULT_REMOTE_HOST,
        help=f"Jump host with VPN access. Default: {DEFAULT_REMOTE_HOST}",
    )
    parser.add_argument(
        "--remote-proxy",
        default=DEFAULT_REMOTE_PROXY,
        help=f"SOCKS proxy reachable on the jump host. Default: {DEFAULT_REMOTE_PROXY}",
    )
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    output = run_remote_fetch(
        username=args.username,
        local_password_env=args.password_env,
        remote_host=args.remote_host,
        remote_proxy=args.remote_proxy,
        passthrough=passthrough,
    )
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
