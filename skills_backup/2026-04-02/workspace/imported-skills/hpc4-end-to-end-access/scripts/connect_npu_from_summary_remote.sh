#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo 'usage: connect_npu_from_summary_remote.sh <index> [summary_json]' >&2
  exit 2
fi

index="$1"
local_summary="${2:-./npu_pool_summary.json}"
remote_host="${HPC4_VPN_HOST:-ubuntu@43.134.118.168}"
remote_skill_root="${HPC4_REMOTE_SKILL_ROOT:-/home/ubuntu/.github/skills}"
remote_script="${remote_skill_root}/hpc4-end-to-end-access/scripts/connect_npu_from_summary.sh"
remote_summary="${HPC4_REMOTE_SUMMARY_PATH:-/tmp/npu_pool_summary_connect_$$.json}"

[[ -f "$local_summary" ]] || {
  echo "local summary file not found: $local_summary" >&2
  exit 1
}

scp -q "$local_summary" "${remote_host}:${remote_summary}"

remote_cmd="$(printf '%q' "$remote_script") $(printf '%q' "$index") $(printf '%q' "$remote_summary")"
exec ssh -tt -o BatchMode=yes -o ConnectTimeout=10 "$remote_host" \
  "bash -lc $(printf '%q' "$remote_cmd")"
