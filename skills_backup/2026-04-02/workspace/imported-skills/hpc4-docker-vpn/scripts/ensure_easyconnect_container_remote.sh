#!/usr/bin/env bash
set -euo pipefail

remote_host="${HPC4_VPN_HOST:-ubuntu@43.134.118.168}"
remote_skill_root="${HPC4_REMOTE_SKILL_ROOT:-/home/ubuntu/.github/skills}"
remote_script="${remote_skill_root}/hpc4-docker-vpn/scripts/ensure_easyconnect_container.sh"

remote_env=()
for key in \
  EASYCONNECT_URL \
  EASYCONNECT_USER \
  EASYCONNECT_PASSWORD \
  EASYCONNECT_CONTAINER_NAME \
  EASYCONNECT_IMAGE \
  EC_VER \
  EASYCONNECT_EXIT_ON_DISCONNECT \
  EASYCONNECT_RESTART_POLICY; do
  if [[ -n "${!key:-}" ]]; then
    remote_env+=("${key}=$(printf '%q' "${!key}")")
  fi
done

remote_cmd=""
if ((${#remote_env[@]})); then
  remote_cmd+="env ${remote_env[*]} "
fi
remote_cmd+="$(printf '%q' "$remote_script")"
for arg in "$@"; do
  remote_cmd+=" $(printf '%q' "$arg")"
done

exec ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote_host" \
  "bash -lc $(printf '%q' "$remote_cmd")"
