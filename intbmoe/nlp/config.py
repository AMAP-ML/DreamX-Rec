"""Configuration for the LLaMA-style MiniPile language model."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core import BlockMoEConfig


def _nlp_block_config() -> BlockMoEConfig:
    return BlockMoEConfig(
        num_blocks=8,
        block_embedding_dim=32,
        router_hidden_dim=64,
        top_k=2,
        num_basis_experts=8,
        layer_dims=(2844, 768),
        use_bias=True,
        use_layer_norm=True,
        basis_weight_network="two_layer",
        basis_weight_hidden_dim=16,
        conditioning_dim=0,
        composition_alpha_init=1.0,
        composition_alpha_trainable=True,
        composition_gate_norm=True,
        composition_gate_norm_type="rmsnorm",
        use_shared_expert=True,
        shared_expert_type="swiglu",
        shared_expert_hidden_dim=2048,
        num_shared_experts=1,
    )


@dataclass(frozen=True)
class NLPConfig:
    # Eighteen-layer LLaMA-style backbone used by this example.
    vocab_size: int = 32_000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 18
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    max_position_embeddings: int = 1024
    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 10_000.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_implementation: str = "flash_attention_2"

    # Every decoder FFN is replaced. Packed samples contain no padding.
    block_moe: BlockMoEConfig = field(default_factory=_nlp_block_config)

    # Training defaults for MODEL_SIZE=150m.
    epochs: float = 1.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 4.0e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1.0e-8
    warmup_ratio: float = 0.1
    minimum_learning_rate_ratio: float = 0.1
    maximum_gradient_norm: float = 1.0
    logging_steps: int = 10
    random_seed: int = 0
    dataloader_workers: int = 0
    experiment_name: str = "block_moe"
