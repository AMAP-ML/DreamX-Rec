#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  "$PYTHON" -m pip install -q -r inthq_demo/requirements.txt
fi

"$PYTHON" -m inthq_demo.train "$@"
