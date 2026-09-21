from typing import Optional

import torch

def create_last_k_mask(
    sequence_length: int, last_item_index: torch.Tensor, last_k: Optional[int] = None
) -> torch.tensor:
    """
    Creates a mask to select the last K items of sequences.
    If a sequence has less than K items, all items are considered for the row.
    If last_k is None, all items are considered for all rows.

    Args:
        sequence_length (int): The length of the sequences.
        last_item_index (torch.Tensor) of shape (batch_size,).
            The tensor containing the indices of the last items in the each row
        last_k (Optional[int]): The number of last K items to consider.
            If None, all items are considered.
    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, sequence_length) with
            True for the last K items in each row and False for the rest.
    """

    if last_k is None:
        start_index = torch.zeros_like(last_item_index)
    else:
        if last_k < 1:
            raise ValueError("last_k must be None or greater than or equal to 1")
        start_index = torch.clamp(
            last_item_index - last_k + 1, min=0
        )  # Shape (batch_size,)

    indices = (
        torch.arange(sequence_length, device=last_item_index.device)
        .unsqueeze(0)
        .expand(last_item_index.size(0), -1)
    )  # shape (batch_size, sequence_length)

    mask = (indices >= start_index.unsqueeze(1)) & (
        indices <= last_item_index.unsqueeze(1)
    )  # Shape (batch_size, sequence_length)
    return mask
