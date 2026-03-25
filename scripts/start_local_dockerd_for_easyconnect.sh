#!/usr/bin/env bash
set -euo pipefail

docker_host="${EASYCONNECT_DOCKER_HOST:-unix:///tmp/docker.sock}"
docker_data_root="${EASYCONNECT_DOCKER_DATA_ROOT:-/tmp/docker-data}"
docker_exec_root="${EASYCONNECT_DOCKER_EXEC_ROOT:-/tmp/docker-exec}"
docker_pidfile="${EASYCONNECT_DOCKER_PIDFILE:-/tmp/dockerd.pid}"
docker_log="${EASYCONNECT_DOCKER_LOG:-/tmp/dockerd-easyconnect.log}"
docker_timeout="${EASYCONNECT_DOCKER_START_TIMEOUT:-20}"

if [[ "${docker_host}" != unix://* ]]; then
  echo "Only unix:// Docker hosts are supported by this helper: ${docker_host}" >&2
  exit 2
fi

sock_path="${docker_host#unix://}"

if ! command -v docker >/dev/null 2>&1; then
  echo 'docker client is not installed' >&2
  exit 1
fi

if ! command -v dockerd >/dev/null 2>&1; then
  echo 'dockerd is not installed' >&2
  exit 1
fi

if docker -H "${docker_host}" info >/dev/null 2>&1; then
  echo "Docker daemon already reachable at ${docker_host}"
  exit 0
fi

mkdir -p "$(dirname "${sock_path}")" "${docker_data_root}" "${docker_exec_root}" "$(dirname "${docker_log}")"
rm -f "${sock_path}" "${docker_pidfile}"

nohup dockerd \
  --host="${docker_host}" \
  --data-root="${docker_data_root}" \
  --exec-root="${docker_exec_root}" \
  --pidfile="${docker_pidfile}" \
  --storage-driver=vfs \
  --iptables=false \
  --bridge=none \
  --ip-forward=false \
  --ip-masq=false \
  >"${docker_log}" 2>&1 &

daemon_is_alive() {
  local pid=""
  pid="$(cat "${docker_pidfile}" 2>/dev/null || true)"
  [[ -n "${pid}" && -d "/proc/${pid}" ]]
}

for ((i = 0; i < docker_timeout; ++i)); do
  if daemon_is_alive && docker -H "${docker_host}" info >/dev/null 2>&1; then
    sleep 1
    if daemon_is_alive && docker -H "${docker_host}" info >/dev/null 2>&1; then
      echo "Docker daemon is ready at ${docker_host}"
      echo "Log file: ${docker_log}"
      exit 0
    fi
  fi

  if [[ -f "${docker_pidfile}" ]] && ! daemon_is_alive; then
    echo "Docker daemon exited during startup" >&2
    echo "Last log lines:" >&2
    tail -n 80 "${docker_log}" >&2 || true
    exit 1
  fi

  if [[ ! -S "${sock_path}" && ! -f "${docker_pidfile}" ]]; then
    sleep 1
    continue
  fi

  sleep 1
done

echo "Docker daemon did not become ready within ${docker_timeout}s" >&2
echo "Last log lines:" >&2
tail -n 80 "${docker_log}" >&2 || true
exit 1
