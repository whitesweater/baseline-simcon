#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${script_dir}/docker_cli.sh"

container_name="${EASYCONNECT_CONTAINER_NAME:-easyconnect}"
restart_policy="${EASYCONNECT_RESTART_POLICY:-no}"
exit_on_disconnect="${EASYCONNECT_EXIT_ON_DISCONNECT:-1}"

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

if ! docker_cli ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  echo "Container ${container_name} does not exist" >&2
  exit 1
fi

image="$(docker_cli inspect "${container_name}" --format '{{.Config.Image}}')"
env_lines="$(docker_cli inspect "${container_name}" --format '{{range .Config.Env}}{{println .}}{{end}}')"
ec_ver="$(printf '%s\n' "${env_lines}" | sed -n 's/^EC_VER=//p' | head -n1)"
cli_opts="$(printf '%s\n' "${env_lines}" | sed -n 's/^CLI_OPTS=//p' | head -n1)"
root_mount_source="$(docker_cli inspect "${container_name}" --format '{{range .Mounts}}{{if eq .Destination "/root"}}{{println .Source}}{{end}}{{end}}' | head -n1)"
logs_mount_source="$(docker_cli inspect "${container_name}" --format '{{range .Mounts}}{{if eq .Destination "/usr/share/sangfor/EasyConnect/resources/logs"}}{{println .Source}}{{end}}{{end}}' | head -n1)"

if [[ -z "${ec_ver}" || -z "${cli_opts}" ]]; then
  echo "Container ${container_name} is missing EC_VER or CLI_OPTS; aborting recreate" >&2
  exit 1
fi

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
)

if is_truthy "${exit_on_disconnect}"; then
  docker_run_args+=( -e EXIT=1 )
fi

if [[ -n "${root_mount_source}" ]]; then
  docker_run_args+=( -v "${root_mount_source}:/root/" )
elif [[ -d "${HOME}/.easyconnect" ]]; then
  docker_run_args+=( -v "${HOME}/.easyconnect:/root/" )
fi

if [[ -n "${logs_mount_source}" ]]; then
  docker_run_args+=( -v "${logs_mount_source}:/usr/share/sangfor/EasyConnect/resources/logs" )
fi

if docker_cli ps --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  docker_cli stop "${container_name}" >/dev/null
fi

docker_cli rm "${container_name}" >/dev/null
docker_cli run "${docker_run_args[@]}" "${image}" >/dev/null

echo "Recreated ${container_name} with restart=${restart_policy}"
if is_truthy "${exit_on_disconnect}"; then
  echo "EXIT=1 is enabled; the container will stop after the VPN session ends instead of auto re-login"
fi
