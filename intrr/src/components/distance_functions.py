from abc import ABC, abstractmethod
import torch

# Compatibility exports: canonical loss implementations live in loss_functions.
from src.components.loss_functions import (
    BetaQuantizationLoss as BetaQuantizationLoss,
    WeightedSquaredError as WeightedSquaredError,
)


__all__ = [
    "DistanceFunction",
    "SquaredEuclideanDistance",
    "WeightedSquaredError",
    "BetaQuantizationLoss",
]


class DistanceFunction(ABC):
    @abstractmethod
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between the rows of x and the rows of y.

        Args:
            x: Data points of shape (n1, d)
            y: Centroids of shape (n2, d)

        Returns:
            Distances of shape (n1, n2)
        """
        pass


class SquaredEuclideanDistance(DistanceFunction):
    def compute(
        self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 256
    ) -> torch.Tensor:
        """
        Compute squared Euclidean distances between the rows of x and the rows of y,
        with optional batching along the x-axis to manage memory.

        Args:
            x: Data points of shape (n1, d)
            y: Centroids of shape (n2, d)
            batch_size: Optional. The number of rows from x to process at a time.
                        If None, no batching is performed (original behavior).

        Returns:
            Squared distances of shape (n1, n2)

        Raises:
            AssertionError: If the input tensors do not have the expected shapes
        """
        assert x.dim() == 2, f"Data must be 2D, got {x.dim()} dimensions"
        assert y.dim() == 2, f"Data must be 2D, got {y.dim()} dimensions"
        assert x.size(1) == y.size(1), f"Data must have the same number of columns"

        n1, d = x.shape
        n2, _ = y.shape

        if batch_size is None or batch_size >= n1:
            # No batching needed or batch_size is larger than n1, compute directly
            x_expanded = x.unsqueeze(1)  # Shape (n1, 1, d)
            y_expanded = y.unsqueeze(0)  # Shape (1, n2, d)
            sq_diffs = (x_expanded - y_expanded).pow(2)  # Shape (n1, n2, d)
            sq_distances = torch.sum(sq_diffs, dim=2)  # Shape (n1, n2)
            return sq_distances
        else:
            # Perform batching
            all_sq_distances = []
            num_batches = (n1 + batch_size - 1) // batch_size  # Ceiling division

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n1)
                x_batch = x[start_idx:end_idx]  # Shape (current_batch_size, d)

                # Expand and compute for the current batch
                x_batch_expanded = x_batch.unsqueeze(
                    1
                )  # Shape (current_batch_size, 1, d)
                y_expanded = y.unsqueeze(0)  # Shape (1, n2, d) - y remains the same

                sq_diffs_batch = (x_batch_expanded - y_expanded).pow(
                    2
                )  # Shape (current_batch_size, n2, d)
                sq_distances_batch = torch.sum(
                    sq_diffs_batch, dim=2
                )  # Shape (current_batch_size, n2)

                all_sq_distances.append(sq_distances_batch)

            # Concatenate the results from all batches
            return torch.cat(all_sq_distances, dim=0)
