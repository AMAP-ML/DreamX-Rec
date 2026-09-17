"""IntBMoE model for single-task POI recommendation."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import BlockMoE
from .config import (
    FEEDBACK_TYPE,
    INTENTION_TYPE,
    PROFILE_TYPE,
    SCENARIO_TYPE,
    RecommendationConfig,
)


TensorDict = Dict[str, torch.Tensor]


class DenseIDEmbedding(nn.Module):
    """Trainable ID table with an explicit zero-padding row."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        nn.init.normal_(self.weight)
        self.register_buffer("padding", torch.zeros(1, embedding_dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        table = torch.cat([self.weight, self.padding], dim=0)
        adjusted = torch.remainder(ids + table.size(0), table.size(0))
        adjusted = adjusted.clamp(0, table.size(0) - 1)
        return table[adjusted]


def recis_available() -> bool:
    """Return whether the optional RecIS embedding backend is importable."""

    try:
        import recis  # noqa: F401
    except Exception:
        return False
    return True


class TorchMultiHashEmbedding(nn.Module):
    """Sample-fitted pure-PyTorch multi-hash embedding tables.

    Only hash keys present in the public sample are materialized.  This changes
    storage, not lookup semantics, for fitted keys.  Unknown keys map to zero;
    callers should fit the vocabulary over train and evaluation data first.
    """

    def __init__(
        self,
        primes: Sequence[int],
        sub_dim: int,
        raw_ids: Iterable[int],
        init_std: float,
        sparse: bool,
    ):
        super().__init__()
        raw = torch.as_tensor(sorted({int(value) for value in raw_ids if int(value) >= 0}), dtype=torch.long)
        self.primes = tuple(int(prime) for prime in primes)
        self.tables = nn.ModuleList()
        for table_index, prime in enumerate(self.primes):
            keys = torch.unique(torch.remainder(raw, prime), sorted=True) if raw.numel() else raw.clone()
            self.register_buffer(f"keys_{table_index}", keys, persistent=True)
            table = nn.Embedding(keys.numel() + 1, sub_dim, padding_idx=0, sparse=sparse)
            nn.init.trunc_normal_(table.weight, std=init_std)
            with torch.no_grad():
                table.weight[0].zero_()
            self.tables.append(table)

    def _lookup_one(self, ids: torch.Tensor, prime: int, table_index: int) -> torch.Tensor:
        keys = getattr(self, f"keys_{table_index}")
        if keys.numel() == 0:
            return torch.zeros(*ids.shape, self.tables[table_index].embedding_dim, device=ids.device)
        hashed = torch.remainder(ids.clamp(min=0), prime)
        positions = torch.bucketize(hashed, keys)
        safe_positions = positions.clamp(max=keys.numel() - 1)
        found = (positions < keys.numel()) & (keys[safe_positions] == hashed) & (ids >= 0)
        rows = torch.where(found, safe_positions + 1, torch.zeros_like(safe_positions))
        return self.tables[table_index](rows)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self._lookup_one(ids, prime, index) for index, prime in enumerate(self.primes)],
            dim=-1,
        )


class RecisMultiHashEmbedding(nn.Module):
    """Optional RecIS-backed multi-hash embedding tables."""

    def __init__(
        self,
        primes: Sequence[int],
        sub_dim: int,
        init_std: float,
        name_prefix: str,
        block_size: int = 10240,
    ):
        super().__init__()
        from recis.nn import DynamicEmbedding, EmbeddingOption
        from recis.nn.initializers import TruncNormalInitializer

        if not torch.cuda.is_available():
            raise RuntimeError("the RecIS embedding backend requires a CUDA device")

        self.primes = tuple(primes)
        self.tables = nn.ModuleList(
            DynamicEmbedding(
                EmbeddingOption(
                    embedding_dim=sub_dim,
                    block_size=block_size,
                    dtype=torch.float32,
                    device=torch.device("cuda"),
                    trainable=True,
                    shared_name=f"{name_prefix}_{index + 1}",
                    coalesced=True,
                    initializer=TruncNormalInitializer(std=init_std),
                    combiner="sum",
                )
            )
            for index in range(len(self.primes))
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        original_shape = ids.shape
        flat = ids.reshape(-1, 1).long()
        if flat.device.type != "cuda":
            flat = flat.cuda()
        safe = flat.clamp(min=0)
        parts = [table(torch.remainder(safe, prime)) for prime, table in zip(self.primes, self.tables)]
        output = torch.cat(parts, dim=-1).reshape(*original_shape, -1)
        return output * (ids >= 0).unsqueeze(-1).to(output.device, output.dtype)


def _init_linear(layer: nn.Linear) -> nn.Linear:
    nn.init.normal_(layer.weight, mean=0.0, std=2.0 / layer.in_features**0.5)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class MaskedProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = _init_linear(nn.Linear(input_dim, output_dim))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.linear(inputs))) * mask


class POIEncoder(nn.Module):
    def __init__(
        self,
        config: RecommendationConfig,
        poi_embedding: nn.Module,
        geographic_embedding: nn.Module,
        category_embedding: DenseIDEmbedding,
        region_embedding: DenseIDEmbedding,
    ):
        super().__init__()
        self.poi_embedding = poi_embedding
        self.geographic_embedding = geographic_embedding
        self.category_embedding = category_embedding
        self.region_embedding = region_embedding
        self.score_projection = MaskedProjection(1, config.model_dim)

    def forward(
        self,
        poi_id: torch.Tensor,
        geographic_id: torch.Tensor,
        score: torch.Tensor,
        category_id: torch.Tensor,
        region_id: torch.Tensor,
    ) -> torch.Tensor:
        mask = (poi_id >= 0).unsqueeze(-1).to(score.dtype)
        score_embedding = self.score_projection(score.unsqueeze(-1), mask)
        output = self.poi_embedding(poi_id) + self.geographic_embedding(geographic_id)
        output = output + score_embedding
        output = output + self.category_embedding(category_id) + self.region_embedding(region_id)
        return output * mask


class TransformerBlock(nn.Module):
    def __init__(self, config: RecommendationConfig):
        super().__init__()
        hidden = config.model_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden // self.num_heads
        if self.head_dim * self.num_heads != hidden:
            raise ValueError("model_dim must be divisible by num_attention_heads")

        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)
        self.attention_dropout = nn.Dropout(config.transformer_dropout)
        self.output_dropout = nn.Dropout(config.transformer_dropout)
        self.ffn_dropout = nn.Dropout(config.transformer_dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = BlockMoE(
            hidden,
            config.encoder_block.with_layer_dims(
                *config.encoder_block.layer_dims[:-1], hidden
            ),
        )

        max_length = config.max_sequence_length
        causal_mask = torch.tril(torch.ones(max_length, max_length, dtype=torch.bool), diagonal=-1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.relative_position_bias = nn.Parameter(
            torch.zeros(config.num_attention_heads, 2 * max_length - 1)
        )
        nn.init.normal_(self.relative_position_bias, mean=0.0, std=0.02)
        row = torch.arange(max_length).view(-1, 1)
        column = torch.arange(max_length).view(1, -1)
        self.register_buffer(
            "relative_position_indices",
            max_length - 1 + column - row,
            persistent=False,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        residual = inputs
        normalized = self.norm1(inputs)
        query = self.query(normalized).view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        key = self.key(normalized).view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        value = self.value(normalized).view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)

        causal = self.causal_mask[:sequence_length, :sequence_length]
        padding_keys = (sequence_mask == 0).view(batch_size, 1, 1, sequence_length)
        masked = causal.view(1, 1, sequence_length, sequence_length) | padding_keys
        attention_bias = masked.to(query.dtype) * -1e9
        relative_indices = self.relative_position_indices[:sequence_length, :sequence_length]
        relative_bias = self.relative_position_bias[:, relative_indices.reshape(-1)]
        relative_bias = relative_bias.reshape(self.num_heads, sequence_length, sequence_length)
        attention_bias = attention_bias + relative_bias.unsqueeze(0).to(query.dtype)

        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        hidden = residual + self.output_dropout(self.output(attention))
        hidden = hidden + self.ffn_dropout(self.ffn(self.norm2(hidden), sequence_mask))
        return hidden * sequence_mask.unsqueeze(-1)


class TransformerEncoder(nn.Module):
    def __init__(self, config: RecommendationConfig):
        super().__init__()
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_encoder_layers))
        self.output_norms = nn.ModuleList(nn.LayerNorm(config.model_dim) for _ in self.layers)

    def forward(
        self,
        inputs: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        hidden = inputs * sequence_mask.unsqueeze(-1)
        outputs = []
        for layer, output_norm in zip(self.layers, self.output_norms):
            hidden = layer(hidden, sequence_mask)
            outputs.append(output_norm(hidden))
        return outputs


class IntBMoERecommendation(nn.Module):
    """Single-task POI recommendation model built with IntBMoE blocks."""

    def __init__(
        self,
        config: Optional[RecommendationConfig] = None,
        *,
        poi_ids: Iterable[int] = (),
        geographic_ids: Iterable[int] = (),
        embedding_backend: str = "torch",
    ):
        super().__init__()
        self.config = config or RecommendationConfig()
        cfg = self.config

        if embedding_backend == "torch":
            self.poi_id_embedding = TorchMultiHashEmbedding(
                cfg.hash_primes,
                cfg.hash_sub_dim,
                poi_ids,
                cfg.sparse_embedding_init_std,
                cfg.sparse_embedding,
            )
            self.geographic_embedding = TorchMultiHashEmbedding(
                cfg.hash_primes,
                cfg.hash_sub_dim,
                geographic_ids,
                cfg.sparse_embedding_init_std,
                cfg.sparse_embedding,
            )
        elif embedding_backend == "recis":
            self.poi_id_embedding = RecisMultiHashEmbedding(
                cfg.hash_primes,
                cfg.hash_sub_dim,
                cfg.sparse_embedding_init_std,
                "global_poi_emb_table",
            )
            self.geographic_embedding = RecisMultiHashEmbedding(
                cfg.hash_primes,
                cfg.hash_sub_dim,
                cfg.sparse_embedding_init_std,
                "global_tile_emb_table",
            )
        else:
            raise ValueError("embedding_backend must be 'torch' or 'recis'")

        self.embedding_backend = embedding_backend

        self.sequence_type_embedding = DenseIDEmbedding(cfg.sequence_type_vocab_size, cfg.model_dim)
        self.detail_type_embedding = DenseIDEmbedding(cfg.detail_type_vocab_size, cfg.model_dim)
        self.weather_embedding = DenseIDEmbedding(cfg.weather_vocab_size, cfg.weather_embedding_dim)
        self.weather_projection = MaskedProjection(cfg.weather_embedding_dim, cfg.model_dim)
        self.region_embedding = DenseIDEmbedding(cfg.administrative_region_vocab_size, cfg.model_dim)
        self.category_embedding = DenseIDEmbedding(cfg.category_vocab_size, cfg.model_dim)
        self.profile_embedding = DenseIDEmbedding(cfg.profile_vocab_size, cfg.model_dim)
        self.travel_mode_embedding = DenseIDEmbedding(cfg.travel_mode_vocab_size, cfg.model_dim)

        self.poi_encoder = POIEncoder(
            cfg,
            self.poi_id_embedding,
            self.geographic_embedding,
            self.category_embedding,
            self.region_embedding,
        )
        self.encoder = TransformerEncoder(cfg)
        self.recommendation_head = BlockMoE(
            cfg.model_dim,
            cfg.head_block.with_layer_dims(
                *cfg.head_block.layer_dims[:-1], cfg.model_dim
            ),
        )

    def split_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Return sparse torch-embedding parameters and all dense parameters."""

        sparse_ids = {
            id(module.weight)
            for module in self.modules()
            if isinstance(module, nn.Embedding) and module.sparse
        }
        sparse, dense = [], []
        for parameter in self.parameters():
            (sparse if id(parameter) in sparse_ids else dense).append(parameter)
        return sparse, dense

    def embed_tokens(self, tokens: Mapping[str, torch.Tensor]) -> torch.Tensor:
        type_ids = tokens["type_id"]
        valid = tokens["valid"].unsqueeze(-1).to(torch.float32)
        scenario_mask = (type_ids == SCENARIO_TYPE).unsqueeze(-1).to(torch.float32)
        intention_mask = (type_ids == INTENTION_TYPE).unsqueeze(-1).to(torch.float32)
        feedback_mask = (type_ids == FEEDBACK_TYPE).unsqueeze(-1).to(torch.float32)
        profile_mask = (type_ids == PROFILE_TYPE).unsqueeze(-1).to(torch.float32)

        weather = self.weather_projection(
            self.weather_embedding(tokens["weather_id"]), scenario_mask
        )
        scenario = (
            weather
            + self.geographic_embedding(tokens["scenario_geographic_id"])
            + self.region_embedding(tokens["scenario_region_id"])
        ) * scenario_mask
        intention = self.poi_encoder(
            tokens["poi_id"],
            tokens["poi_geographic_id"],
            tokens["poi_score"],
            tokens["poi_category_id"],
            tokens["poi_region_id"],
        ) * intention_mask
        feedback = self.travel_mode_embedding(tokens["travel_mode_id"]) * feedback_mask
        profile = self.profile_embedding(tokens["profile_id"]) * profile_mask
        common = self.sequence_type_embedding(type_ids) + self.detail_type_embedding(tokens["detail_id"])
        return (scenario + intention + feedback + profile + common) * valid

    def forward(self, tokens: Mapping[str, torch.Tensor]) -> torch.Tensor:
        sequence_mask = tokens["valid"].to(torch.float32)
        embedded = self.embed_tokens(tokens)
        layer_outputs = self.encoder(embedded, sequence_mask)
        return self.recommendation_head(layer_outputs[-1], sequence_mask)

    @staticmethod
    def _gather_target_states(sequence_output: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        indices = target_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))
        return torch.gather(sequence_output, 1, indices)

    def candidate_logits(
        self,
        sequence_output: torch.Tensor,
        target_positions: torch.Tensor,
        labels: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        target_output = self._gather_target_states(sequence_output, target_positions)
        positive = self.poi_encoder(
            labels["positive_poi_id"],
            labels["positive_geographic_id"],
            labels["positive_score"],
            labels["positive_category_id"],
            labels["positive_region_id"],
        ).unsqueeze(2)
        negative = self.poi_encoder(
            labels["negative_poi_id"],
            labels["negative_geographic_id"],
            labels["negative_score"],
            labels["negative_category_id"],
            labels["negative_region_id"],
        )
        candidates = torch.cat([positive, negative], dim=2)
        candidate_ids = torch.cat(
            [labels["positive_poi_id"].unsqueeze(-1), labels["negative_poi_id"]], dim=-1
        )
        logits = torch.matmul(candidates, target_output.unsqueeze(-1)).squeeze(-1)
        return logits.masked_fill(candidate_ids < 0, -1e20)

    def loss(self, batch: Mapping[str, object]) -> tuple[torch.Tensor, TensorDict]:
        tokens = batch["tokens"]
        labels = batch["labels"]
        target_positions = batch["target_positions"]
        target_valid = batch["target_valid"].bool()
        if not isinstance(tokens, Mapping) or not isinstance(labels, Mapping):
            raise TypeError("batch tokens and labels must be mappings")
        sequence_output = self(tokens)
        logits = self.candidate_logits(sequence_output, target_positions, labels)
        train_valid = target_valid & (labels["positive_poi_id"] >= 0)
        if self.config.exclude_newest_session_from_loss:
            train_valid = train_valid.clone()
            train_valid[:, 0] = False
        per_query = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            torch.zeros(logits.numel() // logits.size(-1), dtype=torch.long, device=logits.device),
            reduction="none",
        ).view_as(train_valid)
        loss = (per_query * train_valid).sum() / train_valid.float().sum().clamp(min=1)
        return loss, {"logits": logits, "target_valid": target_valid, "train_valid": train_valid}
