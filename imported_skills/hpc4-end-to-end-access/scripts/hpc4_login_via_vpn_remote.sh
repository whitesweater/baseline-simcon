#!/usr/bin/env bash
set -euo pipefail

remote_host="${HPC4_VPN_HOST:-ubuntu@43.134.118.168}"
remote_skill_root="${HPC4_REMOTE_SKILL_ROOT:-/home/ubuntu/.github/skills}"
remote_script="${remote_skill_root}/hpc4-end-to-end-access/scripts/hpc4_login_via_vpn.sh"

remote_cmd="$(printf '%q' "$remote_script")"
for arg in "$@"; do
  remote_cmd+=" $(printf '%q' "$arg")"
done

exec ssh -tt -o BatchMode=yes -o ConnectTimeout=10 "$remote_host" \
  "bash -lc $(printf '%q' "$remote_cmd")"
