#!/usr/bin/env bash
# Download MiniPile to a specified data directory.
# Usage: bash download_data.sh [data directory]
# Example: bash download_data.sh ./data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-${SCRIPT_DIR}/data}"
mkdir -p "${DATA_DIR}"

echo "Downloading MiniPile to ${DATA_DIR}/raw ..."

python3 -c "
import os
os.environ.setdefault('HF_ENDPOINT', os.environ.get('HF_ENDPOINT', ''))
from datasets import load_dataset

out_dir = '${DATA_DIR}/raw'
os.makedirs(out_dir, exist_ok=True)

print('Downloading MiniPile...')
ds = load_dataset('JeanKaddour/minipile')
print(f'splits: {list(ds.keys())}')
print(f'train: {len(ds[\"train\"])} records')
print(f'validation: {len(ds[\"validation\"])} records')
print(f'test: {len(ds[\"test\"])} records')

# Inspect the optional domain-label field.
sample = ds['train'][0]
print(f'fields: {list(sample.keys())}')
if 'source' in sample:
    print(f'domain-label field: source, example: {sample[\"source\"]}')
elif 'meta' in sample:
    print(f'domain-label field: meta, example: {sample[\"meta\"]}')
else:
    print('No domain-label field found')

# Save the dataset to disk.
ds.save_to_disk(out_dir)
print(f'Saved to {out_dir}')
"

echo "Download complete: ${DATA_DIR}/raw/"
