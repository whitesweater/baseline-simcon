#!/usr/bin/env bash
set -euo pipefail

echo "=== BASIC ==="
hostname || true
whoami || true
pwd || true

echo
echo "=== NPU SMI ==="
npu-smi info 2>/dev/null || true

echo
echo "=== NPU MAP ==="
npu-smi info -m 2>/dev/null || true

echo
echo "=== NPU TOPO ==="
npu-smi info -t topo 2>/dev/null || true

echo
echo "=== ENV ==="
env | grep -E '^(ASCEND|RANK|HCCL|MASTER|WORLD|LOCAL|PYTHONPATH|LD_LIBRARY_PATH)=' | sort || true

echo
echo "=== HCCL FILE ==="
ls -l /etc/hccl_conf.json 2>/dev/null || true
sed -n '1,120p' /etc/hccl_conf.json 2>/dev/null || true

echo
echo "=== PROCESSES ==="
ps -eo pid,ppid,user,etime,cmd | grep -E 'python|torchrun' | grep -v grep || true
