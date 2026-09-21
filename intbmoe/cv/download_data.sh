#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-${1:-/var/tmp/hf_datasets_cache}}"
mkdir -p "${DATA_PATH}"

python3 - "${DATA_PATH}" <<'PY'
import os
import sys
import time

cache_dir = os.path.abspath(sys.argv[1])
os.environ["HF_DATASETS_CACHE"] = cache_dir

import datasets

print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] downloading imagenet-1k')
print(f"cache_dir: {cache_dir}")
print(f"datasets version: {datasets.__version__}")

dataset = datasets.load_dataset("imagenet-1k", cache_dir=cache_dir)

print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] download/load complete')
print(f"splits: {list(dataset.keys())}")
for split_name, split_dataset in dataset.items():
    print(f"  {split_name}: {len(split_dataset)} records")

sample = dataset["train"][0]
print(f"first train sample keys: {list(sample.keys())}")
print(f'first train label: {sample["label"]}')
PY
