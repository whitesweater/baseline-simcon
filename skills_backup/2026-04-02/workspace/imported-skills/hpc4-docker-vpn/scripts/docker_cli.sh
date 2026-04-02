#!/usr/bin/env bash

docker_cli() {
  local docker_bin="${EASYCONNECT_DOCKER_BIN:-docker}"
  local use_sudo="${EASYCONNECT_DOCKER_USE_SUDO:-auto}"
  local -a cmd=()

  case "${use_sudo}" in
    auto|"")
      if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
        cmd+=(sudo)
      fi
      ;;
    1|true|TRUE|yes|YES|on|ON)
      cmd+=(sudo)
      ;;
    0|false|FALSE|no|NO|off|OFF)
      ;;
    *)
      echo "Invalid EASYCONNECT_DOCKER_USE_SUDO=${use_sudo}" >&2
      return 2
      ;;
  esac

  cmd+=("${docker_bin}")
  if [[ -n "${EASYCONNECT_DOCKER_HOST:-}" ]]; then
    cmd+=(-H "${EASYCONNECT_DOCKER_HOST}")
  fi

  "${cmd[@]}" "$@"
}
