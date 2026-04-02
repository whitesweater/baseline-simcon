#!/usr/bin/env python3
"""Print canonical HPC login host metadata."""

from __future__ import annotations

import argparse
import json


HOSTS = {
    1: {
        "alias": "hpc1",
        "hostname": "hpc1login.hpc.hkust-gz.edu.cn",
        "user": "yhao481",
        "workdir": "/hpc2hdd/home/yhao481/jhupload/proj",
    },
    2: {
        "alias": "hpc2",
        "hostname": "hpc2login.hpc.hkust-gz.edu.cn",
        "user": "yhao481",
        "workdir": "/hpc2hdd/home/yhao481/jhupload/proj",
    },
    3: {
        "alias": "hpc3",
        "hostname": "hpc3login.hpc.hkust-gz.edu.cn",
        "user": "yhao481",
        "workdir": "/data/user/yhao481/proj",
    },
    4: {
        "alias": "hpc4",
        "hostname": "hpc4login.hpc.hkust-gz.edu.cn",
        "user": "user224",
        "workdir": "/data/user/user224/proj",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render canonical HPC1-4 login host metadata.")
    parser.add_argument(
        "--cluster",
        type=int,
        nargs="*",
        choices=[1, 2, 3, 4],
        help="Optional cluster numbers. Defaults to 1 2 3 4.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of plain text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clusters = args.cluster or [1, 2, 3, 4]
    hosts = [{"cluster": cluster, **HOSTS[cluster]} for cluster in clusters]
    if args.json:
        print(json.dumps(hosts, indent=2))
    else:
        for host in hosts:
            print(
                f"HPC{host['cluster']}: {host['alias']} -> {host['user']}@{host['hostname']} "
                f"-> {host['workdir']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
