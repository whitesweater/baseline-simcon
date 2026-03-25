#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo 'usage: connect_npu_from_summary.sh <index> [summary_json]' >&2
  exit 2
fi

index="$1"
summary_file="${2:-$HOME/npu_pool_summary.json}"

if [[ ! -f "$summary_file" ]]; then
  echo "summary file not found: $summary_file" >&2
  exit 1
fi

readarray -t fields < <(python3 - "$summary_file" "$index" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
index = int(sys.argv[2]) - 1
data = json.loads(summary_path.read_text())
items = data.get('running_npu_entries') or []
if index < 0 or index >= len(items):
    raise SystemExit('index out of range')
entry = items[index]
ssh = entry.get('ssh') or ''
password = entry.get('ssh_password') or ''
parts = ssh.split()
if len(parts) < 4 or parts[0] != 'ssh' or parts[1] != '-p':
    raise SystemExit('invalid ssh command in summary')
port = parts[2]
user, host = parts[3].split('@', 1)
print(user)
print(host)
print(port)
print(password)
print(entry.get('name') or '')
PY
)

user="${fields[0]}"
host="${fields[1]}"
port="${fields[2]}"
password="${fields[3]}"
name="${fields[4]}"

echo "Connecting to NPU container ${name}"
exec sshpass -p "$password" ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -p "$port" \
  "$user"@"$host"
