import functools
from typing import Tuple

import torch

from src.components.distance_functions import DistanceFunction
from src.components.clustering_initializers import ClusteringInitializer
from src.components.loss_functions import WeightedSquaredError
from src.components.quantization_strategies import QuantizationStrategy
from src.models.modules.clustering.base_clustering_module import BaseClusteringModule


class VectorQuantization(BaseClusteringModule):
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        distance_function: DistanceFunction,
        initializer: ClusteringInitializer,
        quantization_strategy: QuantizationStrategy,
        loss_function: torch.nn.Module = WeightedSquaredError(),
        optimizer: torch.optim.Optimizer = functools.partial(
            torch.optim.SGD,
            lr=0.5,
        ),
        init_buffer_size: int = 1000,
    ):
        """
        Initialize the VectorQuantization module.

        Args:
            n_clusters: Number of clusters.
            n_features: Number of features in the input data.
            distance_function: Distance function to use for computing distances between points.
            loss_function: Loss function to use for training.
            optimizer: Optimizer to use for training.
            init_method: Initialization method ("random" or "k-means++").
            init_buffer_size: Number of points to buffer for initialization.
        """

        super().__init__(
            n_clusters=n_clusters,
            n_features=n_features,
            distance_function=distance_function,
            loss_function=loss_function,
            optimizer=optimizer,
            initializer=initializer,
            init_buffer_size=init_buffer_size,
        )

        self.quantization_strategy = quantization_strategy

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass of the K-Means model on the input batch.

        This function computes the cluster assignments for each input point, the number
        of points in each cluster, and the sum of points in each cluster.

        Args:
            batch: Data points of shape (batch_size, n_features)

        Returns:
            assignments: Cluster assignments of shape (batch_size,)
            embeddings: Embeddings of shape (batch_size, n_features).
                These embeddings will be used for computing the quantization loss.
            reconstruction_loss_embeddings: Embeddings of shape (batch_size, n_features)
                computed in a way that enables gradient backpropagation through the input
                embeddings. If the quantization strategy does not support this, this will
                be None.
        """
        codebook = self.get_centroids()
        (
            ids,
            embeddings,
            reconstruction_loss_embeddings,
        ) = self.quantization_strategy.quantize(
            codebook=codebook,
            batch=batch,
        )
        return ids, embeddings, reconstruction_loss_embeddings

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Perform a forward pass of the K-Means model on the batch and compute the loss.

        This function may be called by another LightningModule, such as a residual
        K-means module, that is using this MiniBatchKMeans module as a submodule.

        Calling this function along will not update the centroids, and will not
        increment self.global_step. If a parent module is using this module as a
        submodule, the parent will be responsible for updating those parameters.
        Otherwise, these will be updated by Lightning after it calls
        training_step.

        Args:
            batch: Data points of shape (batch_size, n_features)

        Returns:
            assignments: Cluster assignments of shape (batch_size,)
            global_loss_embeddings: Embeddings of shape (batch_size, n_features)
            loss: Loss value. Tensor of shape (1,)
        """
        if batch.device != self.device:
            batch = batch.to(self.device)

        # Initialize centroids using the chosen method
        # Buffer initial batches for better initialization
        if self.is_initial_step:
            self.is_initial_step = False
            self.is_initialized = True
        if not self.is_initialized:
            return self.initialization_step(batch)

        assignments, embeddings, reconstruction_loss_embeddings = self.forward(batch)
        loss = self.loss_function(batch, embeddings)  # quantization loss
        return (
            assignments,
            reconstruction_loss_embeddings
            if reconstruction_loss_embeddings is not None
            else embeddings,
            loss,
        )
