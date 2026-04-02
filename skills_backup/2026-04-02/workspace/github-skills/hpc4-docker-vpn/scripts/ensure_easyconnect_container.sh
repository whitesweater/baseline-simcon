#!/usr/bin/env bash
set -euo pipefail

container_name="${EASYCONNECT_CONTAINER_NAME:-easyconnect}"
image="${EASYCONNECT_IMAGE:-hagb/docker-easyconnect:cli}"
ec_ver="${EC_VER:-7.6.7}"
exit_on_disconnect="${EASYCONNECT_EXIT_ON_DISCONNECT:-1}"
restart_policy="${EASYCONNECT_RESTART_POLICY:-no}"

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

container_has_expected_exit_mode() {
  if ! is_truthy "${exit_on_disconnect}"; then
    return 0
  fi

  sudo docker inspect "${container_name}" \
    --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -Fxq 'EXIT=1'
}

if sudo docker ps --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  if ! container_has_expected_exit_mode; then
    echo "Container ${container_name} is running, but it does not have EXIT=1. Recreate it with recreate_easyconnect_single_session.sh." >&2
    exit 2
  fi

  echo "Container ${container_name} is already running"
  exit 0
fi

if sudo docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  if ! container_has_expected_exit_mode; then
    echo "Container ${container_name} exists, but it does not have EXIT=1. Recreate it with recreate_easyconnect_single_session.sh." >&2
    exit 2
  fi

  echo "Starting existing container ${container_name}"
  sudo docker start "${container_name}" >/dev/null
  exit 0
fi

if [[ -z "${EASYCONNECT_URL:-}" || -z "${EASYCONNECT_USER:-}" || -z "${EASYCONNECT_PASSWORD:-}" ]]; then
  echo 'EASYCONNECT_URL, EASYCONNECT_USER, and EASYCONNECT_PASSWORD are required to create a new container' >&2
  exit 1
fi

cli_opts="-d ${EASYCONNECT_URL} -u ${EASYCONNECT_USER} -p ${EASYCONNECT_PASSWORD}"

docker_run_args=(
  -d
  --name "${container_name}"
  --restart "${restart_policy}"
  --device /dev/net/tun
  --cap-add NET_ADMIN
  --privileged
  -e EC_VER="${ec_ver}"
  -e CLI_OPTS="${cli_opts}"
  -p 127.0.0.1:1080:1080
  -p 127.0.0.1:8888:8888
  -v "${HOME}/.easyconnect:/root/"
)

if is_truthy "${exit_on_disconnect}"; then
  docker_run_args+=( -e EXIT=1 )
fi

sudo docker run "${docker_run_args[@]}" "${image}" >/dev/null

echo "Created container ${container_name} from ${image}"
