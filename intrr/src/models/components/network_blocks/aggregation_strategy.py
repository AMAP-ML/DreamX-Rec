from abc import ABC, abstractmethod
from typing import Optional

import torch

from src.utils.masking_utils import create_last_k_mask


class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        pass


class MeanAggregation(AggregationStrategy):
    """
    Aggregates the embeddings by computing their mean. If last_k is specified, only the last K embeddings are considered.
    """

    def __init__(self, last_k: Optional[int] = None):
        """
        Initializes the MeanAggregation class with the specified number of last embeddings to consider.

        Args:
            last_k Optional[int] = None
                The number of last K embeddings to consider for aggregation. If None, all embeddings are considered.
        """
        self.last_k = last_k

    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregates the last K embeddings for each row by computing their mean.

        Args:
            embeddings (torch.Tensor): Shape (batch_size, sequence_length, embedding_dim).
                The tensor containing embeddings for each row.
            row_ids (torch.Tensor): Shape (return_size,).
                The tensor containing row ids for which the aggregated embedding has to be returned.
            last_item_index (torch.Tensor): Shape (return_size,).
                The tensor containing the indices of the last items in emdeddings for each row in row_ids.

        Returns:
            torch.Tensor: The aggregated embeddings of shape (return_size, embedding_dim).
        """
        # Select the embeddings for the specified row ids
        embeddings = embeddings[
            row_ids
        ]  # Shape (return_size, sequence_length, embedding_dim)
        # Create a mask to select the last K items of sequences
        mask = create_last_k_mask(embeddings.size(1), last_item_index, self.last_k)
        mask = mask.to(dtype=embeddings.dtype, device=embeddings.device)

        # Apply the mask to the embeddings
        masked_embeddings = embeddings * mask.unsqueeze(
            2
        )  # Shape (return_size, sequence_length, embedding_dim)

        # Sum the masked embeddings and divide by the count of non-zero elements in the mask
        sum_embeddings = torch.sum(
            masked_embeddings, dim=1
        )  # Shape (return_size, embedding_dim)
        count = (
            torch.sum(mask, dim=1).clamp(min=1).unsqueeze(1)
        )  # Shape (return_size, embedding_dim)
        return sum_embeddings / count  # Shape (return_size, embedding_dim)


class LastAggregation(AggregationStrategy):
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        return embeddings[row_ids, last_item_index]


class FirstAggregation(AggregationStrategy):
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        # Return the first item in each sequence, assuming sequences are right-padded
        # TODO(liam): allow all aggregation strategies to handle left padding
        return embeddings[row_ids, 0]
