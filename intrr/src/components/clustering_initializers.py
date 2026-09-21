from abc import abstractmethod

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from src.components.distance_functions import DistanceFunction


class ClusteringInitializer(nn.Module):
    """
    Base class for clustering initializers.
    This class provides an interface for initializing centroids for clustering algorithms.
    """

    def __init__(self, n_clusters: int, initialize_on_cpu: bool = False):
        """
        Initializes the ClusteringInitializer class.

        Args:
            n_clusters: Number of clusters to form.
            initialize_on_cpu: Whether to move the tensors to the CPU for computing the
                initialization. This is useful for large initialization buffer sizes for
                which GPU memory might be a constraint. Otherwise, it is recommended to
                keep the tensors on the GPU for faster computation.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.initialize_on_cpu = initialize_on_cpu

    @abstractmethod
    def forward(self, buffer: torch.Tensor) -> torch.Tensor:
        """
        Initialize centroids for clustering algorithms.
        Args:
            buffer: Data points of shape (batch_size, n_features)
        Returns:
            Initialized centroids of shape (n_clusters, n_features)
        """
        pass


class RandomInitializer(ClusteringInitializer):
    """
    Random initialization for clustering algorithms.
    This class provides a method to initialize centroids randomly from the data points.
    """

    def __init__(self, n_clusters: int, initialize_on_cpu: bool = True):
        """
        Initializes the RandomInitialize class with the specified number of clusters.
        Args:
            n_clusters: Number of clusters to form.
        """
        super().__init__(n_clusters=n_clusters, initialize_on_cpu=initialize_on_cpu)

    def forward(self, buffer: torch.Tensor) -> torch.Tensor:
        """
        Initialize centroids randomly from the data points.
        Args:
            buffer: Data points of shape (batch_size, n_features)
        Returns:
            Initialized centroids of shape (n_clusters, n_features)
        """
        if self.initialize_on_cpu:
            old_device = buffer.device
            # Move the buffer to CPU for initialization
            buffer = buffer.to("cpu")

        n_samples = buffer.shape[0]

        # Choose centroids randomly from the data points
        indices = torch.randperm(n_samples, device=buffer.device)[: self.n_clusters]
        centroids = buffer[indices].clone().data
        if self.initialize_on_cpu:
            # Move the centroids back to the original device
            centroids = centroids.to(old_device)

        return centroids


class KMeansPlusPlusInitInitializer(ClusteringInitializer):
    """
    KMeans++ initialization for clustering algorithms.
    This class provides a method to initialize centroids using the k-means++
    algorithm, which is a smarter way to choose the initial centroids for k-means
    clustering. It helps in faster convergence and better clustering results.
    """

    def __init__(
        self,
        n_clusters: int,
        distance_function: DistanceFunction,
        initialize_on_cpu: bool = True,
    ):
        """
        Initializes the KMeansPlusPlusInitInitializer class with the specified parameters.
        Args:
            n_clusters: Number of clusters to form.
            distance_function: Function to compute distances between points.
            initialize_on_cpu: Whether to move the tensors to the CPU for computing the
                initialization.
        """
        super().__init__(n_clusters=n_clusters, initialize_on_cpu=initialize_on_cpu)
        self.distance_function = distance_function

    def forward(self, buffer: torch.Tensor) -> torch.Tensor:
        """
        Initialize centroids using k-means++ algorithm for better convergence.
        The k-means++ algorithm iteratively samples the next initial centroid by
        sampling points with probability proportional to their distance to the nearest
        existing centroid.
        Args:
            buffer: Data points of shape (batch_size, n_features)
        Returns:
            Initialized centroids of shape (n_clusters, n_features)
        """
        if self.initialize_on_cpu:
            old_device = buffer.device
            # Move the buffer to CPU for initialization
            buffer = buffer.to("cpu")

        n_samples = buffer.shape[0]
        n_features = buffer.shape[1]
        centroids = torch.zeros(
            (self.n_clusters, n_features), dtype=buffer.dtype, device=buffer.device
        )

        # Choose first centroid randomly
        first_centroid_idx = torch.randint(0, n_samples, (1,), device=buffer.device)
        centroids[0] = buffer[first_centroid_idx]

        # Choose remaining centroids
        # We cannot vectorize this part because it is inherently sequential
        # However, we only need to execute this loop once at initialization
        for i in range(1, self.n_clusters):
            # Compute distances to the nearest existing centroid
            min_distances = torch.min(
                self.distance_function.compute(buffer, centroids[:i]), dim=1
            )[0]
            if min_distances.sum() == 0:
                # All points are already centroids, so we simply assign the remaining
                # centroids randomly
                centroids[i:] = buffer[
                    torch.randint(
                        0, n_samples, (self.n_clusters - i,), device=buffer.device
                    )
                ]
                break

            # Choose the next centroid with probability proportional to distance
            next_centroid_idx = torch.multinomial(min_distances, num_samples=1)

            # Assign the next centroid
            # We do not need to remove the point from buffer because the probability
            # will be zero for all future iterations
            centroids[i] = buffer[next_centroid_idx]

        if self.initialize_on_cpu:
            # Move the centroids back to the original device
            centroids = centroids.to(old_device)

        return centroids


class ClusteringModuleInitializer(ClusteringInitializer):
    """
    Module to initialize clustering algorithm with the result of another
    clustering algorithm.
    """

    def __init__(
        self,
        n_clusters: int,
        clustering_module: LightningModule,
        initialize_on_cpu: bool = False,
        max_iter: int = 100,
        atol: float = 1e-8,
    ):
        """
        Initializes the KMeansInitializer class with the specified parameters.

        Args:
            n_clusters: Number of clusters to form.
            clustering_module: An instance of a `BaseClusteringModule` to be used to
                initialize the centroids.
            initialize_on_cpu: Whether to move the tensors to the CPU for computing the
                initialization. This is useful for large initialization buffer sizes for
                which GPU memory might be a constraint. Otherwise, it is recommended to
                keep the tensors on the GPU for faster computation.
            max_iter: Maximum number of iterations to run the clustering module for.
            atol: Absolute tolerance for convergence. If all elements of the centroids
                do not change more than this value on consecutive iterations, the
                initialization is considered converged.
        """
        super().__init__(n_clusters=n_clusters, initialize_on_cpu=initialize_on_cpu)

        from src.models.modules.clustering.base_clustering_module import (  # we import here to avoid circular imports
            BaseClusteringModule,
        )

        assert isinstance(
            clustering_module, BaseClusteringModule
        ), "clustering_module must be an instance of BaseClusteringModule"

        self.clustering_module = clustering_module
        self.max_iter = max_iter
        self.atol = atol

    def forward(self, buffer: torch.Tensor) -> torch.Tensor:
        """
        Initialize centroids using self.clustering_module.

        Args:
            buffer: Data points of shape (batch_size, n_features)

        Returns:
            Initialized centroids of shape (n_clusters, n_features)
        """
        self.clustering_module.on_train_start()
        cur_centroids = self.clustering_module.get_centroids()
        for step in range(self.max_iter):
            # Perform a training step
            self.clustering_module.model_step(buffer)
            new_centroids = self.clustering_module.get_centroids()

            # Check for convergence
            if step > 0 and torch.allclose(
                cur_centroids, new_centroids, atol=self.atol
            ):
                print(f"Initialization converged after {step} iterations")
                break
            cur_centroids = new_centroids.clone()

        return cur_centroids.detach()
