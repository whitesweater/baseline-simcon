#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CODI_QWEN3_METHOD_FAMILY="codi"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/train_qwen3_common.sh" "$@"
