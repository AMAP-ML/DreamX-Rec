"""Configuration for the IntTravel recommendation example."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ..core import BlockMoEConfig


def _recommendation_block_config() -> BlockMoEConfig:
    """Build the recommendation-only BlockMoE configuration."""

    return BlockMoEConfig(
        num_blocks=8,
        block_embedding_dim=32,
        router_hidden_dim=64,
        top_k=2,
        num_basis_experts=32,
        layer_dims=(384, 96),
        use_bias=True,
        use_layer_norm=True,
        basis_weight_network="two_layer",
        basis_weight_hidden_dim=16,
        conditioning_dim=0,
        composition_alpha_init=0.0,
        composition_alpha_trainable=True,
        composition_gate_norm=True,
        composition_gate_norm_type="rmsnorm",
        use_shared_expert=True,
        shared_expert_type="swiglu",
        shared_expert_hidden_dim=256,
        num_shared_experts=1,
    )


@dataclass(frozen=True)
class RecommendationConfig:
    # Input layout: newest session first, [F, I, S], followed by six U tokens.
    max_sessions: int = 40
    tokens_per_session: int = 3
    num_profile_tokens: int = 6
    num_negative_samples: int = 64

    model_dim: int = 96
    num_encoder_layers: int = 8
    num_attention_heads: int = 1
    transformer_dropout: float = 0.0
    position_encoding: str = "relative"
    # Four-way multi-hash embeddings: 4 * 24 = 96.
    hash_primes: Tuple[int, ...] = (5008057, 5008121, 5008259, 5008433)
    hash_sub_dim: int = 24
    sparse_embedding_init_std: float = 0.001
    sparse_embedding: bool = False

    sequence_type_vocab_size: int = 4
    detail_type_vocab_size: int = 8
    weather_vocab_size: int = 21
    administrative_region_vocab_size: int = 3274
    category_vocab_size: int = 21
    profile_vocab_size: int = 47
    travel_mode_vocab_size: int = 6
    weather_embedding_dim: int = 48

    # Training defaults for the bundled IntTravel example.
    batch_size: int = 64
    epochs: int = 5
    dense_learning_rate: float = 1.0e-4
    sparse_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    random_seed: int = 0

    exclude_newest_session_from_loss: bool = True
    encoder_block: BlockMoEConfig = field(default_factory=_recommendation_block_config)
    head_block: BlockMoEConfig = field(default_factory=_recommendation_block_config)

    @property
    def max_action_tokens(self) -> int:
        return self.max_sessions * self.tokens_per_session

    @property
    def max_sequence_length(self) -> int:
        return self.max_action_tokens + self.num_profile_tokens


# Input records are adapted to these contiguous token-type ids.
SCENARIO_TYPE = 0
INTENTION_TYPE = 1
FEEDBACK_TYPE = 2
PROFILE_TYPE = 3
