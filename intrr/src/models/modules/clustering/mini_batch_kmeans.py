import functools
from typing import Tuple

import torch
import torch.nn as nn

from src.components.distance_functions import DistanceFunction
from src.components.clustering_initializers import (
    ClusteringInitializer,
    KMeansPlusPlusInitInitializer,
)
from src.components.loss_functions import WeightedSquaredError
from src.models.modules.clustering.base_clustering_module import BaseClusteringModule


class MiniBatchKMeans(BaseClusteringModule):
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        distance_function: DistanceFunction,
        initializer: ClusteringInitializer = KMeansPlusPlusInitInitializer,
        loss_function: torch.nn.Module = WeightedSquaredError(),
        optimizer: torch.optim.Optimizer = functools.partial(
            torch.optim.SGD,
            lr=0.5,
        ),
        init_buffer_size: int = 1000,
        update_manually: bool = False,
    ):
        """
        Initialize an implementation of the mini-batch k-Means algorithm (Sculley 2010).

        Paper reference: https://dl.acm.org/doi/abs/10.1145/1772690.1772862

        Args:
            n_clusters: Number of clusters.
            n_features: Number of features in the input data.
            distance_function: Distance function to use for computing distances between points.
            loss_function: Loss function to use for training.
            optimizer: Optimizer to use for training.
            initializer: Initialization method.
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
            update_manually=update_manually,
        )
        self.cluster_counts = torch.zeros(self.n_clusters)

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
            batch_cluster_counts: Number of points in each cluster of shape (n_clusters,)
            batch_cluster_sums: Sum of points in each cluster of shape (n_clusters, n_features)
        """
        # Compute cluster assignments
        # Note that assignments is automatically detached from the computation graph
        # because it results from argmin
        assignments = self.predict_step(batch, return_embeddings=False)
        assignments_one_hot = (
            nn.functional.one_hot(assignments, self.n_clusters)
        ).detach()
        # Count points in each cluster
        batch_cluster_counts = torch.sum(assignments_one_hot, dim=0)
        self.cluster_counts += batch_cluster_counts
        # Accumulate points for each cluster
        batch_cluster_sums = torch.mm(assignments_one_hot.float().t(), batch)

        return assignments, batch_cluster_counts, batch_cluster_sums

    def model_step(
        self, batch: torch.Tensor
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
            embeddings: Embeddings of shape (batch_size, n_features)
            loss: Loss value. Tensor of shape (1,), or None if update_manually is True.
        """
        batch = batch.to(self.device)

        # Initialize centroids using the chosen method
        # Buffer initial batches for better initialization
        if self.is_initial_step:
            self.is_initial_step = False
            self.is_initialized = True
        if not self.is_initialized:
            return self.initialization_step(batch)

        assignments, batch_cluster_counts, batch_cluster_sums = self.forward(batch)

        centroids = self.get_centroids()
        # Use a mask to avoid division by zero
        mask = batch_cluster_counts != 0
        mask_target = batch_cluster_sums[mask] / batch_cluster_counts[mask].unsqueeze(1)
        centroid_weights = batch_cluster_counts[mask] / self.cluster_counts[mask]

        if self.update_manually:
            self.centroids[mask] = self.centroids[mask].data - (
                (centroids[mask].data - mask_target) * centroid_weights.unsqueeze(1)
            )
            return assignments, centroids[assignments], None
        else:
            # The MiniBatchKMeans algorithm update above is equivalent to an SGD step
            # with learning rate 0.5 on the loss function below
            loss = self.loss_function(centroids[mask], mask_target, centroid_weights)
            return assignments, centroids[assignments], loss

    def on_train_start(self) -> None:
        """Lightning callback to reset the model state at the start of training."""
        self.cluster_counts = torch.zeros(self.n_clusters, device=self.device)
        super().on_train_start()
