"""Packed MiniPile datasets and the no-padding language-model collator."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_from_disk


def load_packed_minipile(data_path: str | Path):
    root = Path(data_path) / "tokenized"
    train_dataset = load_from_disk(str(root / "train"))
    test_dataset = load_from_disk(str(root / "test"))
    return train_dataset, test_dataset


def packed_language_model_collator(features: list[dict]) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([feature["input_ids"] for feature in features], dtype=torch.long)
    return {"input_ids": input_ids, "labels": input_ids.clone()}
