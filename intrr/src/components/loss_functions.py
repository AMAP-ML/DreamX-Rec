import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FullBatchCrossEntropyLoss(nn.Module):
    """
    Contrastive loss with negative samples being all candidates in the embedding table.
    """

    def __init__(
        self,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Initialize the FullBatchContrastiveLoss.

        Parameters
        ----------
        contrastive_tau: float
            Temperature parameter for the contrastive loss.
        normalize: bool
            Whether to normalize the embeddings before computing the logits via dot product.
        """
        super().__init__()
        self.normalize = normalize
        self.cross_entroy_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        label_locations: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the contrastive loss with negative samples from the full vocabulary.

        Parameters
        ----------
        query_embeddings: torch.Tensor (batch_size x sequence length x embedding_dim)
            The embeddings of the query items.
        key_embeddings: torch.Tensor (total number of items x embedding_dim)
            The embeddings of all items, i.e the full embedding table.
        label_locations: torch.Tensor (number of labels x 2)
            The locations of the labels in the input sequences.
        labels: torch.Tensor (number of labels)
            The labels for the input sequences.

        Returns
        -------
        torch.Tensor
            The contrastive loss.
        """
        # get representation of masked tokens
        # label_locations[:, 0] refers to the index of sequences
        # label_locations[:, 1] refers to the index of tokens in the sequences
        query_embeddings = query_embeddings[
            label_locations[:, 0], label_locations[:, 1]
        ]

        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, dim=-1)
            key_embeddings = F.normalize(key_embeddings, dim=-1)

        logits = torch.mm(query_embeddings, key_embeddings.t())

        loss = self.cross_entroy_loss(logits, labels.long())

        return loss
    
class WeightedSquaredError(torch.nn.Module):
    def __init__(self):
        """Initialize the WeightedSquaredError loss function."""
        super().__init__()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the weighted squared error loss.

        Args:
            x: Predicted values of shape (n_points, n_features)
            y: Target values of shape (n_points, n_features)
            weights: Weights for each point of shape (n_points,)

        Returns:
            A tensor containing the weighted squared error loss of shape (1,)
        """
        error = x - y
        squared_error = torch.sum(error**2, dim=-1)
        # If weights are not provided, use uniform weights
        # This is equivalent to the standard squared error loss
        if weights is None:
            return torch.sum(squared_error)
        return torch.sum(weights * squared_error)
    
class BetaQuantizationLoss(torch.nn.Module):
    def __init__(self, beta: float = 0.25, reduction: str = "sum"):
        """Initialize the Beta Quantization Loss.

        Parameters
        ----------
        beta: float
            Weighting factor for the reconstruction loss.
        reduction: str
            Reduction method to apply to the loss. Options are 'none', 'mean', and 'sum'.
        """
        super().__init__()
        self.beta = beta
        self.criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        """
        Compute the beta quantization loss.
        Args:
            x: Original tensor of shape (batch_size, n_features)
            x: Quantized tensor of shape (batch_size, n_features)
        Returns:
            A tensor containing the beta quantization loss of shape (1,)
        """
        x_no_grad = x.detach()
        xq_no_grad = xq.detach()
        loss = self.criterion(x_no_grad, xq) + self.beta * self.criterion(x, xq_no_grad)
        return loss
