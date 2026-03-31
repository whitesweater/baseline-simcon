#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CODI_QWEN3_METHOD_FAMILY="codi"
export CODI_QWEN3_MODEL_KEY="qwen3_1p7b"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/train_qwen3_common.sh" "$@"
