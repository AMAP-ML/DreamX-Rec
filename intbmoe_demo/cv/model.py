"""Eight-layer Tiny ViT with BlockMoE FFNs."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from ..core import BlockMoE
from .config import CVConfig


def _autocast_off(device_type: str):
    if hasattr(torch, "autocast") and device_type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


class Float32Router(nn.Module):
    """Run the router MLP in fp32, then restore the input dtype."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with _autocast_off(inputs.device.type):
            output = self.inner(inputs.float())
        return output.to(inputs.dtype)


class VisionBlockMoE(nn.Module):
    """Adapter matching the ``timm`` ViT MLP contract."""

    def __init__(self, config: CVConfig, num_prefix_tokens: int):
        super().__init__()
        if config.cls_token_mode != "dense":
            raise ValueError("the CV model requires cls_token_mode='dense'")
        self.num_prefix_tokens = num_prefix_tokens
        self.block_moe = BlockMoE(config.model_dim, config.block_moe)
        if config.fp32_router:
            self.block_moe.router = Float32Router(self.block_moe.router)

        dense_hidden = int(config.model_dim * config.mlp_ratio)
        self.cls_ffn = nn.Sequential(
            nn.Linear(config.model_dim, dense_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dense_hidden, config.model_dim),
        )
        for module in self.cls_ffn:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        prefix = inputs[:, : self.num_prefix_tokens]
        patches = inputs[:, self.num_prefix_tokens :]
        patch_mask = patches.new_ones(patches.shape[:2])
        patch_output = self.block_moe(patches, patch_mask)
        return torch.cat([self.cls_ffn(prefix), patch_output], dim=1)


class IntBMoEVisionTransformer(VisionTransformer):
    """Vision Transformer with alternating IntBMoE FFNs."""

    def __init__(self, config: CVConfig | None = None):
        self.config = config or CVConfig()
        cfg = self.config
        super().__init__(
            img_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.input_channels,
            num_classes=cfg.num_classes,
            embed_dim=cfg.model_dim,
            depth=cfg.depth,
            num_heads=cfg.num_attention_heads,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            drop_rate=cfg.dropout,
            drop_path_rate=cfg.drop_path,
            norm_layer=partial(nn.LayerNorm, eps=cfg.layer_norm_eps),
        )
        if self.num_prefix_tokens != 1:
            raise ValueError("the Tiny ViT model must have exactly one CLS token")
        self.moe_layer_flags = tuple(
            layer_index in cfg.moe_layer_indices for layer_index in range(cfg.depth)
        )
        for layer_index in cfg.moe_layer_indices:
            self.blocks[layer_index].mlp = VisionBlockMoE(cfg, self.num_prefix_tokens)

    def no_weight_decay(self):
        skipped = set(super().no_weight_decay())
        for name, _ in self.named_parameters():
            if "router" in name.split("."):
                skipped.add(name)
        return skipped
