#!/usr/bin/env bash
set -euo pipefail

project_id="${1:-670}"
local_output="${2:-./npu_pool_summary.json}"
remote_host="${HPC4_VPN_HOST:-ubuntu@43.134.118.168}"
remote_skill_root="${HPC4_REMOTE_SKILL_ROOT:-/home/ubuntu/.github/skills}"
remote_script="${remote_skill_root}/hpc4-end-to-end-access/scripts/refresh_npu_pool_via_hpc4.sh"
remote_output="${HPC4_REMOTE_SUMMARY_PATH:-/tmp/npu_pool_summary_$$.json}"

mkdir -p "$(dirname "$local_output")"

remote_cmd="$(printf '%q' "$remote_script") $(printf '%q' "$project_id") $(printf '%q' "$remote_output")"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote_host" \
  "bash -lc $(printf '%q' "$remote_cmd")"

scp -q "${remote_host}:${remote_output}" "$local_output"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote_host" \
  "rm -f $(printf '%q' "$remote_output")" >/dev/null

echo "Copied NPU pool summary to $local_output"
