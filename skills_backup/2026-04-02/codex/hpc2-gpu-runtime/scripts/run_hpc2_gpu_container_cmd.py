#!/usr/bin/env python3
"""Connect to one HPC2 GPU development container through the validated remote VPN path."""

from __future__ import annotations

import argparse
import shlex
import subprocess

from hpc2_remote_vpn import (
    DEFAULT_REMOTE_HOST,
    DEFAULT_REMOTE_PROXY,
    choose_container,
    fetch_payload,
    resolve_container_connection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to one running HPC2 GPU development container and optionally run a command."
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--service-name", help="Container serviceName shown in the portal, for example a800_4_3.")
    selector.add_argument("--index", type=int, help="1-based index within the running-container list.")
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
    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help="Search every portal item instead of only running containers.",
    )
    parser.add_argument(
        "--cmd",
        help="Remote shell command to run. Omit this flag to open an interactive SSH session.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved connection target without opening SSH.",
    )
    return parser.parse_args()


def build_ssh_command(connection: dict[str, object], remote_host: str) -> list[str]:
    proxy_command = (
        f"ssh -o LogLevel=ERROR {shlex.quote(remote_host)} "
        f"nc -x 127.0.0.1:1080 -X 5 %h %p"
    )
    return [
        "sshpass",
        "-p",
        str(connection["password"]),
        "ssh",
        "-o",
        f"ProxyCommand={proxy_command}",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=20",
        "-p",
        str(connection["port"]),
        f"{connection['user']}@{connection['host']}",
    ]


def main() -> int:
    args = parse_args()
    payload = fetch_payload(
        username=args.username,
        local_password_env=args.password_env,
        remote_host=args.remote_host,
        remote_proxy=args.remote_proxy,
        include_sensitive=True,
        status="all",
    )
    item = choose_container(
        payload,
        service_name=args.service_name,
        index=args.index,
        running_only=not args.all_statuses,
    )
    connection = resolve_container_connection(item)

    summary = (
        f"service={connection['service_name']} "
        f"target={connection['user']}@{connection['host']}:{connection['port']}"
    )
    print(summary)
    if args.dry_run:
        return 0

    ssh_cmd = build_ssh_command(connection, args.remote_host)
    if args.cmd:
        ssh_cmd.extend(["bash", "-lc", args.cmd])
    raise SystemExit(subprocess.call(ssh_cmd))


if __name__ == "__main__":
    raise SystemExit(main())
