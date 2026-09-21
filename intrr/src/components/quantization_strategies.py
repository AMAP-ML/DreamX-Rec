from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F

from src.components.distance_functions import DistanceFunction
from src.utils.utils import gumbel_softmax_sample


class QuantizationStrategy(ABC):
    """Base class for quantization strategies used in vector quantization."""

    def __init__(
        self,
        distance_function: DistanceFunction,
        compute_reconstruction_loss_embeddings: bool = False,
    ):
        """
        Initialize the quantization strategy.

        Args:
            distance_function: The distance function to compute distances between embeddings.
            compute_reconstruction_loss_embeddings: Whether to compute a version of the
                quantized embeddings in a way that enables gradient backpropagation through the
                input embeddings. If True, these embeddings will be computed and
                returned by quantize(). If False, the quantize will return None for
                reconstruction_loss_embeddings. Please see the parameter description
                of reconstruction_loss_embeddings in quantize() for more details.
        """
        self.distance_function = distance_function
        self.compute_reconstruction_loss_embeddings = (
            compute_reconstruction_loss_embeddings
        )

    def get_nearest_neighbors(
        self,
        codebook: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the nearest neighbors of the batch in the codebook.
        This is used for the STE and rotation trick quantization strategies.
        """
        dists = self.distance_function.compute(batch, codebook)
        ids = torch.argmin(dists, dim=-1)
        return ids, codebook[ids]

    @abstractmethod
    def quantize(
        self,
        codebook: torch.Tensor,
        batch: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize the input batch using the specified quantization strategy.

        This method should return the indices of the nearest neighbors in the codebook,
        the quantized embeddings, and the quantized embeddings computed in a way that
        enables gradient backpropagation through the input embeddings, which is useful
        for training RQ-VAE-style models.

        Args:
            codebook: The codebook tensor of shape (num_embeddings, embedding_dim).
            batch: The input batch tensor of shape (batch_size, embedding_dim).

        Returns:
            ids: Indices of the nearest neighbors in the codebook.
            embeddings: The quantized embeddings.
            reconstruction_loss_embeddings (Optional): The quantized embeddings computed
                in a way that enables gradient backpropagation through the input
                embeddings. This is useful for training RQ-VAE-style models where the
                raw embeddings are encoded in a lower-dimensional space, then quantized,
                and then the quantized embeddings are decoded to reconstruct the
                original raw input embeddings. To train the encoder, we compute the
                reconstruction loss on the decoded reconstruction_loss_embeddings, so
                that we can backpropagate through the quantization steps to obtain the
                gradients for the encoder.
        """
        pass


class GumbelSoftmaxQuantization(QuantizationStrategy):
    def __init__(
        self,
        temperature: float = 0.7,
        **kwargs,
    ):
        """
        Initialize the Gumbel Softmax quantization strategy.

        This strategy uses the Gumbel Softmax distribution to sample from the
        categorical distribution defined by the distances to the codebook embeddings.

        Args:
            temperature: The temperature parameter for the Gumbel Softmax distribution.
        """
        super().__init__(**kwargs)
        self.temperature = temperature

    def quantize(
        self,
        codebook: torch.Tensor,
        batch: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = self.distance_function.compute(batch, codebook)
        weights = gumbel_softmax_sample(-dists, temperature=self.temperature)
        embeddings = weights @ codebook
        reconstruction_loss_embeddings = embeddings
        ids = torch.argmax(weights, dim=-1)
        return ids, embeddings, reconstruction_loss_embeddings


class STEQuantization(QuantizationStrategy):
    def quantize(
        self,
        codebook: torch.Tensor,
        batch: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes the input batch using the Straight-Through Estimator (STE).

        Args:
            codebook: The codebook tensor of shape (num_embeddings, embedding_dim).
            batch: The input batch tensor of shape (batch_size, embedding_dim).

        Returns:
            ids: Indices of the nearest neighbors in the codebook.
            embeddings: The embeddings corresponding to the nearest neighbors.
            reconstruction_loss_embeddings: .
        """
        ids, embeddings = self.get_nearest_neighbors(codebook, batch)
        reconstruction_loss_embeddings = batch + (embeddings - batch).detach()
        return ids, embeddings, reconstruction_loss_embeddings


class RotationTrickQuantization(QuantizationStrategy):
    def rotate_and_scale_batch(
        self,
        batch: torch.Tensor,
        quantized_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotate and scale the batch embeddings to make them equal quantized_embeddings.

        Args:
            batch: The input batch tensor of shape (batch_size, embedding_dim).
            quantized_embeddings: The quantized embeddings tensor of shape
                (num_embeddings, embedding_dim).
        Returns:
            The transformed batch embeddings of shape (batch_size, embedding_dim).
        """
        # Detach tensors that will be used in the transformation
        quantized_embeddings = quantized_embeddings.detach()
        detached_batch = batch.detach()

        quantized_norms = torch.linalg.vector_norm(
            quantized_embeddings, dim=-1
        ).unsqueeze(
            1
        )  # batch_size x 1
        batch_norms = torch.linalg.vector_norm(detached_batch, dim=-1).unsqueeze(
            1
        )  # batch_size x 1
        lambda_ = quantized_norms / batch_norms  # batch_size x 1

        normalized_batch = detached_batch / batch_norms  # batch_size x embedding_dim
        normalized_embeddings = (
            quantized_embeddings / quantized_norms
        )  # batch_size x embedding_dim

        normalized_sum = F.normalize(
            normalized_batch + normalized_embeddings, p=2, dim=1
        )
        batch = batch.unsqueeze(1)  # -> batch_size x 1 x embedding_dim

        # the following implements equation 4.2 in https://arxiv.org/abs/2410.06424
        sum_projection = (
            batch @ normalized_sum.unsqueeze(2) @ normalized_sum.unsqueeze(1)
        )  # batch_size x 1 x embedding_dim
        rescaled_embeddings = (
            batch @ normalized_batch.unsqueeze(2) @ normalized_embeddings.unsqueeze(1)
        )  # batch_size x 1 x embedding_dim
        return (
            lambda_ * (batch - 2 * sum_projection + 2 * rescaled_embeddings).squeeze()
        )

    def quantize(
        self,
        codebook: torch.Tensor,
        batch: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes the input batch using the rotation trick: https://arxiv.org/abs/2410.06424

        For each input in the batch, it finds the nearest neighbor in the codebook and
        multiplies the input by a transformation matrix that maps the input to the
        nearest neighbor. It returns this mapped vector as the reconstruction loss
        embedding, which allows gradients to flow back to the encoder.

        Args:
            codebook: The codebook tensor of shape (num_embeddings, embedding_dim).
            batch: The input batch tensor of shape (batch_size, embedding_dim).

        Returns:
            ids: Indices of the nearest neighbors in the codebook.
            embeddings: The embeddings corresponding to the nearest neighbors.
            reconstruction_loss_embeddings: The transformed embeddings using the efficient rotation trick.
        """
        ids, embeddings = self.get_nearest_neighbors(codebook, batch)
        reconstruction_loss_embeddings = self.rotate_and_scale_batch(
            batch, embeddings
        )  # batch_size x embedding_dim
        return ids, embeddings, reconstruction_loss_embeddings
