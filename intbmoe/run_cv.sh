#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACKAGE_NAME="$(basename "${SCRIPT_DIR}")"
cd "${REPOSITORY_ROOT}"

DATA_PATH="${DATA_PATH:-/var/tmp/hf_datasets_cache/}"
SEED="${SEED:-0}"

export MOE_TYPE="block_moe"
export BLOCK_MOE_SFN_INTERMEDIATE_SIZE="${BLOCK_MOE_SFN_INTERMEDIATE_SIZE:-768}"
export BLOCK_MOE_NUM_BASIS_EXPERTS="${BLOCK_MOE_NUM_BASIS_EXPERTS:-8}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-cv_${RUN_TIMESTAMP}}"

NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPOSITORY_ROOT}/output/${EXP_NAME}}"
mkdir -p "${OUTPUT_DIR}"

torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  -m "${PACKAGE_NAME}.cv.train" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  "$@" >"${OUTPUT_DIR}/train.log" 2>&1
