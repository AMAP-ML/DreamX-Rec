import torch
import torch.nn as nn

from src.models.components.network_blocks.aggregation_strategy import (
    AggregationStrategy,
)


class EmbeddingAggregator(nn.Module):
    """Embedding aggregator function. this determins how user embeddings are aggregated to form the final user embedding.

    Parameters
    ----------
    aggregation_type: str
        aggregation function type
    """

    def __init__(
        self,
        aggregation_strategy: AggregationStrategy,
    ):
        super(EmbeddingAggregator, self).__init__()
        self.aggregation_strategy = aggregation_strategy

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # embeddings: (batch_size, sequence_length, embedding_dim)
        # attention_mask: (batch_size, sequence_length)

        # we -1 here because the token index starts from 0
        last_item_index = attention_mask.sum(dim=1) - 1

        # The following 3 steps are equivalent to
        # row_ids = torch.arange(embeddings.size(0))
        # but in a way that is traceable with Fx.

        # 1. Create a dummy tensor with the same batch shape as attention_mask
        dummy_tensor_for_batch_shape = attention_mask[:, 0]  # Shape (batch_size,)

        # 2. Use torch.ones_like to create a tensor of ones with that shape.
        # Note that torch.ones is not traceable in Fx, so we use torch.ones_like.
        ones_tensor = torch.ones_like(dummy_tensor_for_batch_shape, dtype=torch.long)

        # 3. Use cumsum to get the 0 to batch_size - 1 sequence
        row_ids = torch.cumsum(ones_tensor, dim=0) - 1

        return self.aggregation_strategy.aggregate(embeddings, row_ids, last_item_index)
