"""Configuration for the Tiny ViT + BlockMoE ImageNet experiment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Tuple

from ..core import BlockMoEConfig


def _cv_block_config() -> BlockMoEConfig:
    return BlockMoEConfig(
        num_blocks=8,
        block_embedding_dim=32,
        router_hidden_dim=64,
        top_k=2,
        num_basis_experts=16,
        layer_dims=(768, 192),
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
        shared_expert_hidden_dim=512,
        num_shared_experts=1,
    )


@dataclass(frozen=True)
class CVConfig:
    # Eight-layer DeiT-Tiny-style backbone used by this example.
    image_size: int = 224
    patch_size: int = 16
    input_channels: int = 3
    num_classes: int = 1000
    model_dim: int = 192
    depth: int = 8
    num_attention_heads: int = 3
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1.0e-6
    dropout: float = 0.0
    drop_path: float = 0.1

    # Replace the FFN in alternating layers. CLS keeps a dense FFN; only the
    # 196 patch tokens are routed through BlockMoE.
    moe_layer_indices: Tuple[int, ...] = (0, 2, 4, 6)
    cls_token_mode: str = "dense"
    fp32_router: bool = True
    block_moe: BlockMoEConfig = field(default_factory=_cv_block_config)

    # Training defaults.
    epochs: int = 200
    warmup_epochs: int = 5
    batch_size_per_device: int = 256
    base_learning_rate: float = 5.0e-4
    warmup_learning_rate: float = 1.0e-6
    minimum_learning_rate: float = 1.0e-5
    weight_decay: float = 0.05
    gradient_clip_norm: float = 1.0
    num_workers: int = 10
    random_seed: int = 0

    color_jitter: float = 0.4
    auto_augment: str = "rand-m9-mstd0.5-inc1"
    random_erasing_probability: float = 0.25
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_probability: float = 1.0
    mixup_switch_probability: float = 0.5
    label_smoothing: float = 0.1

    experiment_name: str = "block_moe"


def cv_config_from_env() -> CVConfig:
    """Apply the environment overrides supported by the CV launcher."""

    moe_type = os.environ.get("MOE_TYPE", "block_moe")
    if moe_type != "block_moe":
        raise ValueError("the minimal CV package only supports MOE_TYPE=block_moe")

    cfg = CVConfig(
        experiment_name=os.environ.get("EXP_NAME", "block_moe"),
    )
    intermediate = int(os.environ.get("BLOCK_MOE_SFN_INTERMEDIATE_SIZE", "768"))
    num_basis = int(os.environ.get("BLOCK_MOE_NUM_BASIS_EXPERTS", "8"))
    block = replace(
        cfg.block_moe,
        num_basis_experts=num_basis,
        layer_dims=(intermediate, cfg.model_dim),
    )
    return replace(cfg, block_moe=block)
