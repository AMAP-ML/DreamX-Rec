from typing import Any, Dict, Optional, Tuple

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from torch import nn
from torchmetrics import MeanMetric

from src.components.distance_functions import DistanceFunction
from src.components.clustering_initializers import ClusteringInitializer
from src.components.loss_functions import WeightedSquaredError


class BaseClusteringModule(LightningModule):
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        distance_function: DistanceFunction,
        initializer: ClusteringInitializer,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        init_buffer_size: int = 1000,
        update_manually: bool = False,
    ):
        """
        Initialize the base clustering module.

        Args:
            n_clusters: Number of clusters.
            n_features: Number of features in the input data.
            distance_function: Distance function to use for computing distances between points.
            initializer: Initialization method.
            loss_function: Loss function to use for training.
            optimizer: Optimizer to use for training.
            scheduler: Learning rate scheduler to use for training.
            init_buffer_size: Number of points to buffer for initialization.
            update_manually: Whether to manually update the centroids without gradients.
        """
        super(BaseClusteringModule, self).__init__()

        self.n_clusters = n_clusters
        self.n_features = n_features
        self.distance_function = distance_function
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.initializer = initializer
        self.init_buffer_size = init_buffer_size

        self.centroids = torch.nn.Parameter(
            torch.zeros(self.n_clusters, self.n_features), requires_grad=True
        )
        self.update_manually = update_manually
        if self.update_manually:
            self.centroids.requires_grad = False

        self.init_loss_function = WeightedSquaredError()
        self.init_buffer = torch.tensor([])
        self.is_initialized = False
        self.is_initial_step = False
        self.train_loss = MeanMetric()

    def _buffer_points(self, batch: torch.Tensor) -> None:
        """
        Buffer points for initialization.

        Args:
            batch: Data points of shape (batch_size, n_features)
        """
        batch = batch.detach()
        n_to_add = min(
            self.init_buffer_size - self.init_buffer.shape[0], batch.shape[0]
        )
        self.init_buffer = torch.cat([self.init_buffer, batch[:n_to_add]], dim=0)

    @rank_zero_only
    def compute_initial_centroids(self, buffer: torch.Tensor) -> None:
        """
        Initialize the centroids by setting `self.init_centroids`.

        Args:
            buffer: Data points of shape (batch_size, n_features)

        Raises:
            ValueError: If the buffer size is less than the number of clusters.
        """
        if buffer.shape[0] < self.n_clusters:
            raise ValueError(
                f"Buffer size {buffer.shape[0]} is less than the number of clusters"
                f" {self.n_clusters}."
            )

        self.init_centroids = self.initializer(buffer)

    def initialization_step(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform a model step that occurs before centroids are initialized.

        This step is used to buffer points for initialization and, if the buffer is
        large enough (containing at least self.init_buffer_size points), to compute the
        initial centroids using the specified initialization method.

        This method does not update the centroids or perform any optimization step.
        Instead, it computes the loss that will result in the centroids being updated
        to the initial centroids in the next step.

        Args:
            batch: Data points of shape (batch_size, n_features)
        """
        self._buffer_points(batch)
        if self.init_buffer.shape[0] < self.init_buffer_size:
            centroid_zero_embeddings = torch.zeros_like(
                self.centroids.data, dtype=batch.dtype, device=self.device
            )
            loss = self.init_loss_function(self.centroids, centroid_zero_embeddings)
            # The centroids at the start of this step are 0, so we return dummy embeddings
            # and assignments
            batch_zero_embeddings = torch.zeros_like(
                batch, dtype=batch.dtype, device=self.device
            )
            batch_zero_assignments = torch.zeros(
                batch.shape[0], dtype=torch.long, device=self.device
            )
            return batch_zero_assignments, batch_zero_embeddings, loss
        else:
            self.init_centroids = torch.zeros_like(
                self.centroids.data, dtype=batch.dtype, device=self.device
            )
            # This function is rank zero only, so only the buffer from the first device
            # is used to initialize the centroids
            self.compute_initial_centroids(self.init_buffer)
            self.is_initial_step = True
            self.init_buffer = torch.tensor([], device=self.device)

            if self.update_manually:
                # If we are updating manually, we set the centroids to the initial
                # centroids without gradients
                self.centroids[:] = self.init_centroids.data
                distances = self.distance_function.compute(batch, self.centroids.data)
                assignments = torch.argmin(distances, dim=1).to(self.device)
                return assignments, self.centroids[assignments], None

            loss = self.init_loss_function(self.centroids, self.init_centroids)
            distances = self.distance_function.compute(batch, self.init_centroids)
            assignments = torch.argmin(distances, dim=1).to(self.device)
            return assignments, self.init_centroids[assignments], loss

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Update clusters based on the points in `batch`.

        Args:
            batch: Data points of shape (batch_size, n_features)

        Returns:
            Tuple of a tensor of cluster assignments of shape (batch_size,), a tensor of
            embeddings of shape (batch_size, n_features), and a Boolean indicating
            whether the algorithm has completed, meaning it has either converged or
            reached the maximum number of iterations.
        """
        raise NotImplementedError(
            "Inherit from this class and implement the forward method."
        )

    def model_step(
        self, batch: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Update clusters based on the points in `batch`.

        Args:
            batch: Data points of shape (batch_size, n_features)

        Returns:
            Tuple of a tensor of cluster assignments of shape (batch_size,), a tensor of
            embeddings of shape (batch_size, n_features), and a Boolean indicating
            whether the algorithm has completed, meaning it has either converged or
            reached the maximum number of iterations.
        """
        raise NotImplementedError(
            "Inherit from this class and implement the forward method."
        )

    def training_step(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Update clusters based on the points in `batch`.

        Args:
            batch: Data points of shape (batch_size, n_features)

        Returns:
            Tuple of a tensor of cluster assignments of shape (batch_size,), a tensor of
            embeddings of shape (batch_size, n_features), and a Boolean indicating
            whether the algorithm has completed, meaning it has either converged or
            reached the maximum number of iterations.
        """
        _, _, loss = self.model_step(batch)

        self.train_loss(loss)
        train_dict_to_log = {
            "train/loss": self.train_loss,
        }
        self.log_dict(
            train_dict_to_log,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def predict_step(
        self, batch: torch.Tensor, return_embeddings: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict cluster assignments for input points.

        Args:
            batch: Data points of shape (batch_size, n_features)
            return_embeddings: Whether to return the embeddings of the assigned centroids

        Returns:
            Tuple of cluster assignments of shape (batch_size,) and embeddings of shape
            (batch_size, n_features) if `return_embeddings` is True, else only the
            cluster assignments.
        """
        batch = batch.to(self.device)
        with torch.no_grad():
            centroids = self.get_centroids().data
            distances = self.distance_function.compute(batch, centroids)
            assignments = torch.argmin(distances, dim=1)
            if not return_embeddings:
                return assignments
            return assignments, centroids[assignments]

    def get_centroids(self) -> nn.Parameter:
        """
        Get the current centroids.

        Returns:
            Centroids of shape (n_clusters, n_features)
        """
        return self.centroids

    def get_residuals(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Get residuals of points from the nearest centroids.

        Args:
            batch: Data points of shape (batch_size, n_features)

        Returns:
            Residuals of shape (batch_size, n_features)
        """
        _, centroids = self.predict_step(batch)
        return batch - centroids

    def on_train_start(self) -> None:
        """Lightning callback to reset the model state at the start of training."""
        self.train_loss.reset()
        self.init_buffer = torch.tensor([], device=self.device)
        self.centroids = self.centroids.to(self.device)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(params=(self.centroids,))
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
