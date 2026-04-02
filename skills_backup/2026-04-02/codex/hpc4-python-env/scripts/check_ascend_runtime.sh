#!/usr/bin/env bash

set -euo pipefail

echo "== python =="
command -v python3 || true
python3 --version || true

echo "== project root guard =="
pwd || true
realpath . || true
printf '%s\n' "/data/user/user224/proj"

echo "== uv =="
command -v uv || true
uv --version || true

echo "== npu-smi =="
command -v npu-smi || true
npu-smi info || true

echo "== ascend dirs =="
find /usr/local/Ascend -maxdepth 4 -type d 2>/dev/null | sort || true
find /opt/conda/Ascend -maxdepth 4 -type d 2>/dev/null | sort || true

echo "== set_env candidates =="
find /usr/local/Ascend /opt/conda/Ascend -name set_env.sh 2>/dev/null | sort || true

echo "== libhccl =="
find / -name libhccl.so 2>/dev/null | sed -n '1,20p' || true

echo "== env =="
env | grep -E 'ASCEND|HCCL|LD_LIBRARY_PATH|PYTHONPATH' || true
