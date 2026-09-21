"""The domain-independent IntBMoE block-routing module.

The recommendation, CV, and NLP models are expected to own their model
configuration.  This module receives every architectural choice explicitly and
does not import a task-specific/global configuration file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BlockMoEConfig:
    """Configuration of one BlockMoE module.

    Defaults match the recommendation example's BlockMoE settings.
    ``layer_dims`` is overridden by the calling model when a different
    projection shape is required.
    """

    num_blocks: int = 8
    block_embedding_dim: int = 32
    router_hidden_dim: int = 64
    top_k: int = 2

    num_basis_experts: int = 32
    layer_dims: Tuple[int, ...] = (384, 96)
    use_bias: bool = True
    use_layer_norm: bool = True

    basis_weight_network: str = "two_layer"
    basis_weight_hidden_dim: int = 16

    conditioning_dim: int = 0

    composition_alpha_init: float = 0.0
    composition_alpha_trainable: bool = True
    composition_gate_norm: bool = True
    composition_gate_norm_type: str = "rmsnorm"

    use_shared_expert: bool = True
    shared_expert_type: str = "swiglu"
    shared_expert_hidden_dim: int = 384
    num_shared_experts: int = 1

    def with_layer_dims(self, *dims: int) -> "BlockMoEConfig":
        return replace(self, layer_dims=tuple(dims))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(inputs.square(), dim=-1, keepdim=True) + self.eps)
        return inputs / rms * self.scale


class ExpertNetwork(nn.Module):
    """Basis-expert parameters shared by all routed blocks.

    Weights use the ``[expert, input, output]`` layout and the same uniform
    initialization as ``nn.Linear``. Biases are initialized to zero.
    """

    def __init__(self, num_experts: int, input_dim: int, output_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.empty(num_experts, input_dim, output_dim))
        self.bias = nn.Parameter(torch.empty(num_experts, output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.input_dim)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)

class SwiGLUExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs))


def _build_shared_expert(kind: str, input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(input_dim, output_dim)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    if kind == "swiglu":
        return SwiGLUExpert(input_dim, hidden_dim, output_dim)
    raise ValueError(f"unsupported shared_expert_type: {kind}")


class BlockMoE(nn.Module):
    """Per-block top-k routing over parameter-composed blocks.

    Each block owns an embedding, not a separate expert network.  Its effective
    weights are composed from shared basis matrices. Routed tokens are gathered
    and evaluated one block at a time.
    """

    def __init__(self, input_dim: int, config: BlockMoEConfig):
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.output_dim = config.layer_dims[-1]
        self.top_k = min(config.top_k, config.num_blocks)

        if self.output_dim != input_dim:
            raise ValueError("BlockMoE requires layer_dims[-1] == input_dim")
        if config.basis_weight_network not in {"linear", "two_layer"}:
            raise ValueError("basis_weight_network must be 'linear' or 'two_layer'")

        self.router = nn.Sequential(
            nn.Linear(input_dim, config.router_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.router_hidden_dim, config.num_blocks),
        )

        self.block_embeddings = nn.Embedding(config.num_blocks, config.block_embedding_dim)

        basis_layers = []
        layer_input_dim = input_dim
        for layer_output_dim in config.layer_dims:
            basis_layers.append(ExpertNetwork(config.num_basis_experts, layer_input_dim, layer_output_dim))
            layer_input_dim = layer_output_dim
        self.basis_layers = nn.ModuleList(basis_layers)

        composition_input_dim = config.block_embedding_dim + config.conditioning_dim
        if config.basis_weight_network == "linear":
            self.basis_weight_projection = nn.Linear(composition_input_dim, config.num_basis_experts)
            composition_hidden_dim = composition_input_dim
        else:
            self.basis_weight_projection = nn.Sequential(
                nn.Linear(composition_input_dim, config.basis_weight_hidden_dim),
                nn.LayerNorm(config.basis_weight_hidden_dim),
                nn.ReLU(),
            )
            composition_hidden_dim = config.basis_weight_hidden_dim

        if config.basis_weight_network == "linear":
            self.value_basis_projection = self.basis_weight_projection
        else:
            self.value_basis_projection = nn.Linear(composition_hidden_dim, config.num_basis_experts)
        self.gate_basis_projection = nn.Linear(composition_hidden_dim, config.num_basis_experts)
        alpha = torch.tensor([config.composition_alpha_init], dtype=torch.float32)
        if config.composition_alpha_trainable:
            self.composition_alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("composition_alpha", alpha)

        filter_input_dim = input_dim + config.block_embedding_dim + config.conditioning_dim
        self.feature_filter = nn.Linear(filter_input_dim, input_dim)

        if config.use_layer_norm:
            self.layer_norms = nn.ModuleList(nn.LayerNorm(dim) for dim in config.layer_dims)

        if config.composition_gate_norm:
            norm_cls = RMSNorm if config.composition_gate_norm_type == "rmsnorm" else nn.LayerNorm
            self.composition_gate_norms = nn.ModuleList(norm_cls(dim) for dim in config.layer_dims)

        if config.use_shared_expert:
            self.shared_experts = nn.ModuleList(
                _build_shared_expert(
                    config.shared_expert_type,
                    input_dim,
                    config.shared_expert_hidden_dim,
                    self.output_dim,
                )
                for _ in range(config.num_shared_experts)
            )

    def _normalize_basis_weights(self, weights: torch.Tensor) -> torch.Tensor:
        return weights / math.sqrt(weights.size(-1))

    def _routing(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        route_indices = router_logits.topk(self.top_k, dim=-1).indices
        route_probabilities = F.softmax(router_logits, dim=-1).gather(2, route_indices)
        return route_indices, route_probabilities

    def forward(
        self,
        inputs: torch.Tensor,
        sequence_mask: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.config
        if conditioning is not None and cfg.conditioning_dim == 0:
            raise ValueError("conditioning was passed but conditioning_dim is zero")
        if conditioning is None and cfg.conditioning_dim != 0:
            raise ValueError("conditioning is required by this BlockMoE configuration")

        batch_size, sequence_length, input_dim = inputs.shape
        mask_2d = sequence_mask.squeeze(-1) if sequence_mask.dim() == 3 else sequence_mask
        router_logits = self.router(inputs)
        route_indices, route_probabilities = self._routing(router_logits)

        flat_inputs = inputs.reshape(batch_size * sequence_length, input_dim)
        flat_mask = mask_2d.reshape(-1)
        flat_route_indices = route_indices.reshape(-1, self.top_k)
        flat_route_indices = flat_route_indices.masked_fill(
            flat_mask.unsqueeze(-1) == 0, -1
        )
        flat_route_probabilities = route_probabilities.reshape(-1, self.top_k)
        flat_output = inputs.new_zeros(
            batch_size * sequence_length, self.output_dim
        )

        block_embedding_table = self.block_embeddings.weight
        if conditioning is not None:
            condition_table = conditioning.unsqueeze(0).expand(cfg.num_blocks, -1)
            composition_input = torch.cat([block_embedding_table, condition_table], dim=-1)
        else:
            composition_input = block_embedding_table

        composition_hidden = self.basis_weight_projection(composition_input)
        if cfg.basis_weight_network == "linear":
            value_weights = self._normalize_basis_weights(composition_hidden)
            gate_weights = self._normalize_basis_weights(
                self.gate_basis_projection(composition_input)
            )
        else:
            value_weights = self._normalize_basis_weights(
                self.value_basis_projection(composition_hidden)
            )
            gate_weights = self._normalize_basis_weights(
                self.gate_basis_projection(composition_hidden)
            )

        layer_parameters = []
        for basis in self.basis_layers:
            value_matrix = torch.einsum("ke,edh->kdh", value_weights, basis.weight)
            value_bias = torch.einsum("ke,ed->kd", value_weights, basis.bias)
            gate_matrix = torch.einsum("ke,edh->kdh", gate_weights, basis.weight)
            gate_bias = torch.einsum("ke,ed->kd", gate_weights, basis.bias)
            layer_parameters.append((value_matrix, value_bias, gate_matrix, gate_bias))

        for block_index in range(cfg.num_blocks):
            route_matches = flat_route_indices == block_index
            token_indices = route_matches.any(dim=-1).nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue

            block_matches = route_matches[token_indices]
            block_probabilities = (
                flat_route_probabilities[token_indices]
                * block_matches.to(flat_route_probabilities.dtype)
            ).sum(dim=-1)
            current = flat_inputs[token_indices]

            block_embedding = block_embedding_table[block_index]
            filter_parts = [
                current,
                block_embedding.unsqueeze(0).expand(current.size(0), -1),
            ]
            if conditioning is not None:
                filter_parts.append(
                    conditioning.unsqueeze(0).expand(current.size(0), -1)
                )
            filter_gate = torch.sigmoid(
                self.feature_filter(torch.cat(filter_parts, dim=-1))
            )
            current = current * filter_gate

            for layer_index, (
                value_matrix,
                value_bias,
                gate_matrix,
                gate_bias,
            ) in enumerate(layer_parameters):
                is_last = layer_index == len(layer_parameters) - 1
                value_output = torch.matmul(
                    current, value_matrix[block_index]
                )
                if cfg.use_bias:
                    value_output = value_output + value_bias[block_index]

                gate_output = torch.matmul(
                    current, gate_matrix[block_index]
                )
                if cfg.use_bias:
                    gate_output = gate_output + gate_bias[block_index]
                if cfg.composition_gate_norm:
                    gate_output = self.composition_gate_norms[layer_index](
                        gate_output
                    )
                current = value_output * (
                    1.0 + self.composition_alpha * F.silu(gate_output)
                )

                if not is_last and cfg.use_layer_norm:
                    current = self.layer_norms[layer_index](current)

            flat_output[token_indices] += (
                current * block_probabilities.unsqueeze(-1)
            )

        output = flat_output.reshape(
            batch_size, sequence_length, self.output_dim
        ).to(inputs.dtype)

        if cfg.use_shared_expert:
            shared_output = sum(expert(inputs) for expert in self.shared_experts)
            output = output + shared_output * mask_2d.unsqueeze(-1)
        return output
