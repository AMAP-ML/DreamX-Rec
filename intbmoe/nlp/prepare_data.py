# -*- coding: utf-8 -*-
"""Prepare MiniPile for language-model training.

This script:
1. Downloads JeanKaddour/minipile, or reuses an existing data/raw dataset.
2. Detects an optional domain-label field.
3. Trains a 32K BPE tokenizer, or reuses an existing tokenizer.
4. Tokenizes and stream-packs documents to the requested context length.
5. Saves Hugging Face datasets that train.py can load from disk.

Usage:
    python3 prepare_data.py
    HF_ENDPOINT=https://hf-mirror.com python3 prepare_data.py
"""

import argparse
import json
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "data")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                        help="data root; relative paths are resolved from this script")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--ctx", type=int, default=1024,
                        help="target sequence length for packing")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="tokenization batch size")
    parser.add_argument("--retrain_tokenizer", action="store_true",
                        help="retrain the tokenizer even if one already exists")
    parser.add_argument("--retokenize", action="store_true",
                        help="rerun packing even if tokenized data already exists")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir = os.path.abspath(os.path.join(PROJECT_ROOT, args.out_dir))
    return args


def download_dataset(out_dir):
    """Download MiniPile and inspect its fields, or load an existing copy."""
    from datasets import load_dataset, load_from_disk, DatasetDict
    raw_path = os.path.join(out_dir, "raw")
    if os.path.isdir(raw_path):
        print(f"Found existing data at {raw_path}; skipping download.")
        if os.path.isfile(os.path.join(raw_path, "dataset_dict.json")):
            ds = load_from_disk(raw_path)
        else:
            ds = DatasetDict()
            for split_name in sorted(os.listdir(raw_path)):
                split_path = os.path.join(raw_path, split_name)
                if os.path.isdir(split_path):
                    ds[split_name] = load_from_disk(split_path)
        if "train" not in ds:
            raise ValueError(f"No train split found under {raw_path}; check the dataset")
    else:
        print("Downloading MiniPile...")
        ds = load_dataset("JeanKaddour/minipile")
        ds.save_to_disk(raw_path)
        print(f"Saved raw data to {raw_path}")
    print(f"splits: {list(ds.keys())}")
    for s in ds:
        print(f"  {s}: {len(ds[s])} records")

    # Detect an optional domain-label field.
    sample = ds['train'][0]
    print(f"\nFields: {list(sample.keys())}")
    domain_field = None
    if 'source' in sample:
        domain_field = 'source'
    elif 'pile_set_name' in sample:
        domain_field = 'pile_set_name'
    elif 'meta' in sample and isinstance(sample['meta'], dict) and 'pile_set_name' in sample['meta']:
        domain_field = 'meta.pile_set_name'

    if domain_field:
        print(f"✓ Found domain-label field: '{domain_field}'")
        for i in range(min(5, len(ds['train']))):
            item = ds['train'][i]
            if domain_field == 'meta.pile_set_name':
                val = item['meta']['pile_set_name']
            else:
                val = item[domain_field]
            print(f"  [{i}] {val}")
    else:
        print("✗ No domain-label field found; per-domain analysis is unavailable")

    return ds, domain_field


def train_tokenizer(ds, vocab_size, out_dir, retrain=False):
    """Train a 32K BPE tokenizer on MiniPile, or reuse an existing one."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tok_path = os.path.join(out_dir, "tokenizer")
    if not retrain and os.path.isfile(os.path.join(tok_path, "tokenizer.json")):
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        print(f"\nReusing tokenizer: {tok_path}  (vocab_size={tokenizer.vocab_size})")
        return tokenizer

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    print(f"\nTraining BPE tokenizer (vocab_size={vocab_size})...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>"],
        show_progress=True,
    )

    # Stream text batches instead of loading the full corpus into memory.
    def text_iterator(batch_size=1000):
        for i in range(0, len(ds['train']), batch_size):
            yield ds['train'][i:i + batch_size]['text']

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=len(ds['train']))

    os.makedirs(tok_path, exist_ok=True)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
    )
    hf_tokenizer.save_pretrained(tok_path)
    print(f"Saved tokenizer to {tok_path}  (vocab_size={hf_tokenizer.vocab_size})")
    return hf_tokenizer


def tokenize_and_pack(ds, tokenizer, ctx, out_dir, split, batch_size=4096):
    """Tokenize and pack one split in a single streaming pass.

    Encoded batches are appended to a temporary binary token stream and then
    wrapped as an Arrow ListArray through a zero-copy memory map. Documents are
    separated by eos_id. A final segment shorter than ctx is discarded, so no
    padding tokens or intermediate cache copies are introduced.
    """
    import pyarrow as pa
    from datasets import Dataset

    print(f"\nProcessing {split} split...")
    eos_id = tokenizer.eos_token_id
    split_ds = ds[split]
    n_docs = len(split_ds)

    bin_path = os.path.join(out_dir, f".{split}_tokens.bin")
    total_tokens = 0
    total_bytes = 0
    with open(bin_path, "wb") as f:
        for i in range(0, n_docs, batch_size):
            batch_texts = split_ds[i:i + batch_size]['text']
            total_bytes += sum(len(t.encode('utf-8')) for t in batch_texts)
            encoded = tokenizer(batch_texts, add_special_tokens=False)['input_ids']
            flat = np.fromiter(
                (tid for ids in encoded for tid in (ids + [eos_id])), dtype=np.int32
            )
            f.write(flat.tobytes())
            total_tokens += flat.size
            if (i // batch_size) % 50 == 0:
                print(f"  Tokenized {i:,}/{n_docs:,} documents, {total_tokens:,} tokens")

    n_seqs = total_tokens // ctx
    dropped = total_tokens - n_seqs * ctx
    print(f"  Documents: {n_docs:,}, tokens including EOS: {total_tokens:,}, bytes: {total_bytes:,}")
    print(f"  Packed sequences: {n_seqs:,}, sequence length: {ctx}")
    print(f"  Token utilization: {n_seqs * ctx / max(total_tokens, 1) * 100:.2f}% "
          f"(discarded {dropped} trailing tokens)")
    if n_seqs == 0:
        raise ValueError(f"{split} does not contain enough tokens for one ctx={ctx} sequence")
    if n_seqs * ctx >= 2 ** 31:
        raise ValueError(f"{split} exceeds the int32 offset limit and must be sharded")

    # Wrap the memory map as Arrow without copying the token data.
    stream = np.memmap(bin_path, dtype=np.int32, mode="r", shape=(total_tokens,))
    values = pa.Array.from_buffers(
        pa.int32(), n_seqs * ctx, [None, pa.py_buffer(stream[:n_seqs * ctx])]
    )
    offsets = pa.array(np.arange(0, n_seqs * ctx + 1, ctx, dtype=np.int32))
    table = pa.table({"input_ids": pa.ListArray.from_arrays(offsets, values)})

    dataset = Dataset(table)
    save_path = os.path.join(out_dir, "tokenized", split)
    dataset.save_to_disk(save_path)
    print(f"  Saved to {save_path}")

    del dataset, table, values, stream
    os.remove(bin_path)
    return total_tokens, total_bytes


def split_is_packed(out_dir, split, ctx):
    """Return True when a packed split exists with the requested length."""
    from datasets import load_from_disk

    save_path = os.path.join(out_dir, "tokenized", split)
    if not os.path.isfile(os.path.join(save_path, "dataset_info.json")):
        return False
    packed = load_from_disk(save_path)
    got = len(packed[0]["input_ids"])
    if got != ctx:
        print(f"  Existing {split} length is {got}, not ctx={ctx}; regenerating.")
        return False
    print(f"  Found packed {split}: {packed.num_rows:,} x {ctx}; skipping.")
    return True


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Data root: {args.out_dir}")

    # 1. Download the dataset and inspect its optional domain label.
    ds, domain_field = download_dataset(args.out_dir)

    # 2. Train or reuse the tokenizer.
    tokenizer = train_tokenizer(ds, args.vocab_size, args.out_dir, args.retrain_tokenizer)

    # 3. Tokenize and pack, retaining prior statistics for reusable splits.
    meta_path = os.path.join(args.out_dir, "meta.json")
    prev_meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            prev_meta = json.load(f)

    meta = {"vocab_size": tokenizer.vocab_size, "ctx": args.ctx, "domain_field": domain_field}
    for split in ['train', 'test', 'validation']:
        if split not in ds:
            continue
        if not args.retokenize and split_is_packed(args.out_dir, split, args.ctx):
            for key in (f"{split}_tokens", f"{split}_bytes"):
                if key in prev_meta:
                    meta[key] = prev_meta[key]
            continue
        total_tokens, total_bytes = tokenize_and_pack(
            ds, tokenizer, args.ctx, args.out_dir, split, args.batch_size
        )
        meta[f"{split}_tokens"] = total_tokens
        meta[f"{split}_bytes"] = total_bytes

    # 4. Save metadata.
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata to {meta_path}")
    print(json.dumps(meta, indent=2))
    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
