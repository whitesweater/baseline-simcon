#!/usr/bin/env bash
set -euo pipefail

. "$HOME/.profile"

if [[ -z "${hpc4user:-}" || -z "${hpc4password:-}" ]]; then
  echo 'hpc4user or hpc4password is not set in ~/.profile' >&2
  exit 1
fi

exec sshpass -p "$hpc4password" ssh \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=10 \
  -o ProxyCommand='nc -x 127.0.0.1:1080 -X 5 %h %p' \
  "$hpc4user"@hpc4login.hpc.hkust-gz.edu.cn "$@"
