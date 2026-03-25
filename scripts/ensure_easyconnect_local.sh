#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

export EASYCONNECT_DOCKER_HOST="${EASYCONNECT_DOCKER_HOST:-unix:///tmp/docker.sock}"
export EASYCONNECT_DOCKER_USE_SUDO="${EASYCONNECT_DOCKER_USE_SUDO:-0}"

"${script_dir}/start_local_dockerd_for_easyconnect.sh"
"${script_dir}/check_local_easyconnect_runtime.sh"
exec "${repo_root}/imported_skills/hpc4-docker-vpn/scripts/ensure_easyconnect_container.sh" "$@"
