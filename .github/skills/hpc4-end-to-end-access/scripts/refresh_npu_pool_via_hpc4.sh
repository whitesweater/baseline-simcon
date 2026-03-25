#!/usr/bin/env bash
set -euo pipefail

project_id="${1:-670}"
output_file="${2:-$HOME/npu_pool_summary.json}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

. "$HOME/.profile"

if [[ -z "${hpc4user:-}" || -z "${hpc4password:-}" ]]; then
  echo 'hpc4user or hpc4password is not set in ~/.profile' >&2
  exit 1
fi

sshpass -p "$hpc4password" ssh -T \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=10 \
  -o ProxyCommand='nc -x 127.0.0.1:1080 -X 5 %h %p' \
  "$hpc4user"@hpc4login.hpc.hkust-gz.edu.cn \
  python3 - "$hpc4user" "$hpc4password" "$project_id" \
  < "$script_dir/list_remote_npu_pool.py" > "$output_file"

echo "Wrote NPU pool summary to $output_file"
