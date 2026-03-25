#!/usr/bin/env bash
set -euo pipefail

container_name="${EASYCONNECT_CONTAINER_NAME:-easyconnect}"
target_host="${1:-hpc4login.hpc.hkust-gz.edu.cn}"

echo '[1/3] EasyConnect container'
sudo docker ps --filter "name=${container_name}" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

echo
echo '[2/3] SOCKS5 proxy test'
curl -sS --max-time 10 --proxy socks5h://127.0.0.1:1080 https://remote.hkust-gz.edu.cn/ >/dev/null
echo 'SOCKS5 OK: remote.hkust-gz.edu.cn reachable via 127.0.0.1:1080'

echo
echo '[3/3] HPC login SSH port test'
nc -vz -x 127.0.0.1:1080 -X 5 "${target_host}" 22
echo "SSH TCP OK: ${target_host}:22 reachable via SOCKS5"
