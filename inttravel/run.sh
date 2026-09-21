#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
NPROC="${NPROC:-1}"

if [ "$NPROC" -eq 1 ]; then
  exec "$PYTHON" -m inttravel.train "$@"
fi

BACKEND="torch"
EXPECT_BACKEND=0
for ARG in "$@"; do
  if [ "$EXPECT_BACKEND" -eq 1 ]; then
    BACKEND="$ARG"
    EXPECT_BACKEND=0
  elif [ "$ARG" = "--embedding-backend" ]; then
    EXPECT_BACKEND=1
  elif [[ "$ARG" == --embedding-backend=* ]]; then
    BACKEND="${ARG#*=}"
  fi
done
if [ "$BACKEND" != "recis" ]; then
  printf '%s\n' "NPROC>1 requires --embedding-backend recis" >&2
  exit 2
fi

exec "$PYTHON" -m torch.distributed.run \
  --nproc-per-node="$NPROC" \
  -m inttravel.train "$@"
