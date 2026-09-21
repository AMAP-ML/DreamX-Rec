#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACKAGE_NAME="$(basename "${SCRIPT_DIR}")"
cd "${REPOSITORY_ROOT}"

DATA_PATH="${DATA_PATH:-${REPOSITORY_ROOT}/data_process/output/processed_features.csv}"
SEED="${SEED:-0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-recommendation_${RUN_TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPOSITORY_ROOT}/output/${EXP_NAME}}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')}"
mkdir -p "${OUTPUT_DIR}"

echo "recommendation log: ${OUTPUT_DIR}/train.log"
exec torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  -m "${PACKAGE_NAME}.recommendation.train" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  "$@" >"${OUTPUT_DIR}/train.log" 2>&1
