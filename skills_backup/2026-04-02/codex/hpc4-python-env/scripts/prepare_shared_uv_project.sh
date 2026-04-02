#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /data/user/user224/proj/<project>" >&2
  exit 1
fi

PROJECT_ROOT="$1"
WORKSPACE_ROOT="/data/user/user224/proj"

case "${PROJECT_ROOT}" in
  "${WORKSPACE_ROOT}"/*) ;;
  *)
    echo "error: project must live under ${WORKSPACE_ROOT}" >&2
    exit 1
    ;;
esac

export HPC4_PROJ_ROOT="${WORKSPACE_ROOT}"
export UV_CACHE_DIR="${HPC4_PROJ_ROOT}/.uv-cache"
export UV_LINK_MODE=copy

mkdir -p "${UV_CACHE_DIR}"
cd "${PROJECT_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY no_proxy NO_PROXY

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "cwd=$(pwd)"
