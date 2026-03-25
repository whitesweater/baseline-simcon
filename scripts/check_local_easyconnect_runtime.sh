#!/usr/bin/env bash
set -euo pipefail

docker_host="${EASYCONNECT_DOCKER_HOST:-unix:///tmp/docker.sock}"
status=0

check_cmd() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    echo "[ok] command available: ${name}"
  else
    echo "[missing] command unavailable: ${name}"
    status=1
  fi
}

echo '== commands =='
check_cmd docker
check_cmd dockerd
check_cmd curl
check_cmd nc
check_cmd sshpass

echo
echo '== docker daemon =='
if docker -H "${docker_host}" info >/dev/null 2>&1; then
  echo "[ok] docker daemon reachable at ${docker_host}"
else
  echo "[warn] docker daemon not reachable at ${docker_host}"
  status=1
fi

echo
echo '== nested container capability =='
if unshare -m true >/dev/null 2>&1; then
  echo '[ok] mount namespace creation is allowed'
else
  echo '[blocked] unshare -m failed; nested containers cannot start in this environment'
  status=1
fi

echo
echo '== tun device =='
if [[ -e /dev/net/tun ]]; then
  ls -l /dev/net/tun
else
  echo '[blocked] /dev/net/tun is missing'
  status=1
fi

echo
echo '== summary =='
if ((status == 0)); then
  echo 'Local EasyConnect Docker runtime looks ready.'
else
  echo 'Local EasyConnect Docker runtime is not ready yet.'
  echo 'This environment needs nested-container support plus /dev/net/tun exposure.'
fi

exit "${status}"
