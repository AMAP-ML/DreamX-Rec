#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACKAGE_NAME="$(basename "${SCRIPT_DIR}")"
cd "${REPOSITORY_ROOT}"

export MODEL_SIZE="${MODEL_SIZE:-150m}"
export MOE_TYPE="block_moe"
export BLOCK_MOE_NUM_BASIS_EXPERTS="${BLOCK_MOE_NUM_BASIS_EXPERTS:-8}"
SEED="${SEED:-0}"

[[ "${MODEL_SIZE}" == "150m" ]] || { echo "This example supports only MODEL_SIZE=150m" >&2; exit 1; }
[[ "${BLOCK_MOE_NUM_BASIS_EXPERTS}" == "8" ]] || { echo "This model requires 8 basis experts" >&2; exit 1; }

DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/nlp/data}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-nlp_${RUN_TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPOSITORY_ROOT}/output/${EXP_NAME}}"
NPROC="${NPROC:-8}"
PER_DEVICE_BS="${PER_DEVICE_BS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
MASTER_PORT="${MASTER_PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')}"
mkdir -p "${OUTPUT_DIR}"

torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  -m "${PACKAGE_NAME}.nlp.train" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --per-device-batch-size "${PER_DEVICE_BS}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --learning-rate "${LEARNING_RATE}" \
  --seed "${SEED}" \
  "$@" >"${OUTPUT_DIR}/train.log" 2>&1
