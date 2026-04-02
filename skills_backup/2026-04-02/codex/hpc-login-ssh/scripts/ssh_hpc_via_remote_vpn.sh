#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_CONFIG="${SCRIPT_DIR%/scripts}/references/ssh_config_hpc_via_remote_vpn"

exec ssh -F "$SSH_CONFIG" "$@"
