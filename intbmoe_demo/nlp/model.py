"""LLaMA-style language model with BlockMoE in every decoder layer."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from ..core import BlockMoE
from .config import NLPConfig


class NLPBlockMoE(nn.Module):
    """Adapter matching the HuggingFace LLaMA MLP contract."""

    def __init__(self, config: NLPConfig):
        super().__init__()
        self.block_moe = BlockMoE(config.hidden_size, config.block_moe)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # MiniPile is continuously packed into fixed-length sequences, so every
        # position is a real token. Keep routing under the surrounding bf16
        # autocast, matching the training precision used by this task.
        sequence_mask = hidden_states.new_ones(hidden_states.shape[:2])
        return self.block_moe(hidden_states, sequence_mask)


def build_nlp_model(
    config: NLPConfig | None = None,
    *,
    attention_implementation: str | None = None,
) -> LlamaForCausalLM:
    """Build the 18-layer backbone and replace every FFN with BlockMoE."""

    cfg = config or NLPConfig()
    hf_config = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        initializer_range=cfg.initializer_range,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        attention_bias=False,
        attention_dropout=cfg.attention_dropout,
        mlp_bias=False,
        tie_word_embeddings=cfg.tie_word_embeddings,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=None,
        attn_implementation=attention_implementation or cfg.attention_implementation,
        torch_dtype=torch.bfloat16,
    )
    model = LlamaForCausalLM(hf_config)
    for layer in model.model.layers:
        layer.mlp = NLPBlockMoE(cfg)
    model.moe_layer_flags = tuple(True for _ in model.model.layers)
    return model
