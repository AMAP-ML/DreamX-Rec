#!/usr/bin/env bash
# Train a 32K BPE tokenizer on MiniPile.
# Usage: bash train_tokenizer.sh [data directory]
# Prerequisite: run download_data.sh first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-${SCRIPT_DIR}/data}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"

echo "Training BPE tokenizer (vocab_size=${VOCAB_SIZE})..."
echo "Input: ${DATA_DIR}/raw"
echo "Output: ${DATA_DIR}/tokenizer/"

python3 -c "
import os
from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

data_dir = '${DATA_DIR}'
vocab_size = ${VOCAB_SIZE}

# Load the downloaded dataset.
print('Loading data...')
ds = load_from_disk(os.path.join(data_dir, 'raw'))
print(f'train: {len(ds[\"train\"])} records')

# Train the BPE tokenizer.
print(f'Training BPE tokenizer (vocab_size={vocab_size})...')
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=['<unk>', '<s>', '</s>'],
    show_progress=True,
)

def text_iterator(batch_size=1000):
    for i in range(0, len(ds['train']), batch_size):
        batch = ds['train'][i:i+batch_size]
        yield batch['text']

tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=len(ds['train']))

# Save the tokenizer.
tok_path = os.path.join(data_dir, 'tokenizer')
os.makedirs(tok_path, exist_ok=True)

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token='<unk>',
    bos_token='<s>',
    eos_token='</s>',
)
hf_tokenizer.save_pretrained(tok_path)
print(f'Saved tokenizer to {tok_path} (vocab_size={hf_tokenizer.vocab_size})')

# Count tokens in the training split.
print('Counting training tokens...')
total_tokens = 0
for item in ds['train']:
    total_tokens += len(hf_tokenizer.encode(item['text']))
print(f'MiniPile train tokens: {total_tokens:,}')
"

echo "Tokenizer training complete: ${DATA_DIR}/tokenizer/"
