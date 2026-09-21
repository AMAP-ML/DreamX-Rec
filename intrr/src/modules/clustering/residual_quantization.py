import copy
import logging
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torch.distributions import Categorical
from torchmetrics import MeanMetric

from src.data.loading.components.interfaces import ItemData
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.modules.clustering.base_clustering_module import BaseClusteringModule


class ResidualQuantization(LightningModule):
    def __init__(
        self,
        n_layers: Optional[int] = None,
        normalization_layer: nn.Module = nn.Identity(),
        encoder: nn.Module = nn.Identity(),
        decoder: nn.Module = nn.Identity(),
        quantization_layer: Optional[BaseClusteringModule] = None,
        quantization_layer_list: Optional[nn.ModuleList] = None,
        init_buffer_size: int = 1000,
        training_loop_function: callable = None,
        quantization_loss_weight: float = 1.0,
        reconstruction_loss_function: Optional[nn.Module] = None,
        reconstruction_loss_weight: float = 0.0,
        normalize_residuals: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        train_layer_wise: bool = False,
        track_residuals: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the Residual Quantization module.

        Args:
            n_layers: The number of quantization layers.
            quantization_layer: The quantization layer to use.
            quantization_layer_list: The list of quantization layers to use.
            init_buffer_size: The size of the buffer for initializing the centroids.
            training_loop_function: The custom training loop function to use.
            normalize_residuals: Whether to normalize the residuals before quantization.
            optimizer: The optimizer to use.
            scheduler: The learning rate scheduler to use.
            quantization_loss_weight: The weight of the quantization loss.
            reconstruction_loss_function: The loss function to use for reconstruction.
            reconstruction_loss_weight: The weight of the reconstruction loss.
            normalize_inputs: Whether to normalize the input embeddings.
            batch_norm_inputs: Whether to apply batch normalization to the input embeddings.
            train_layer_wise: Whether to train the layers one at a time. If true, each layer
                will be trained for the same, plus or minus one, number of steps.
            track_residuals: Whether to track residuals at each layer.
            verbose: Whether to log progress during training.
        """
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "optimizer",
                "scheduler",
                "training_loop_function",
                "normalization_layer",
                "encoder",
                "decoder",
                "quantization_layer_list",
                "quantization_layer",
            ],
        )

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.verbose = verbose
        self.log_if_true("Verbose mode enabled", self.verbose)
        # We always track residuals if verbose mode is enabled
        self.track_residuals = track_residuals or self.verbose

        self.normalization_layer = normalization_layer
        self.encoder = encoder
        self.decoder = decoder
        self.quantization_layer_list = self._instantiate_quantization_layer_list(
            quantization_layer,
            quantization_layer_list,
            n_layers,
        )
        self.n_layers = len(self.quantization_layer_list)

        self.training_loop_function = training_loop_function
        if self.training_loop_function is not None:
            self.log_if_true("Using custom training loop function", self.verbose)
            self.automatic_optimization = False
        self.train_layer_wise = train_layer_wise
        self.normalize_residuals = normalize_residuals

        self.quantization_loss_weight = quantization_loss_weight
        self.reconstruction_loss_function = reconstruction_loss_function
        self.reconstruction_loss_weight = reconstruction_loss_weight

        self.train_loss = MeanMetric()
        self.train_quantization_loss = MeanMetric()
        self.train_reconstruction_loss = MeanMetric()
        if self.verbose:
            # Note that if normalize_residuals is True, the residuals norm metrics below are uninformative
            self.train_first_residuals_norm_ratio = MeanMetric()
            self.train_last_residuals_norm_ratio = MeanMetric()
            self.first_centroids_norm = MeanMetric()
            self.last_centroids_norm = MeanMetric()
            self.train_frac_unique_ids = MeanMetric()
            self.train_mse = MeanMetric()
            for layer_idx in range(self.n_layers):
                # We use MeanMetric to track the fraction of unique ids and the
                # entropy of the cluster ids for each layer
                # Note that we don't need to move these metrics to the device here,
                # because they will be moved to the device in the training_step method
                setattr(
                    self,
                    f"train_layer_coverages_{layer_idx}",
                    MeanMetric(),
                )
                setattr(
                    self,
                    f"train_layer_id_entropy_{layer_idx}",
                    MeanMetric(),
                )

        self.val_loss = MeanMetric()
        self.val_first_residuals_norm_ratio = MeanMetric()
        self.val_last_residuals_norm_ratio = MeanMetric()
        self.val_mse = MeanMetric()
        self.val_frac_unique_ids = MeanMetric()

        self.test_loss = MeanMetric()
        self.test_first_residuals_norm_ratio = MeanMetric()
        self.test_last_residuals_norm_ratio = MeanMetric()
        self.test_mse = MeanMetric()
        self.test_frac_unique_ids = MeanMetric()

        # We set the initialization buffer sizes for each layer to the same value
        self.init_buffer_size = init_buffer_size
        for layer in self.quantization_layer_list:
            layer.init_buffer_size = init_buffer_size

    def _instantiate_quantization_layer_list(
        self,
        quantization_layer: Optional[BaseClusteringModule] = None,
        quantization_layer_list: Optional[nn.ModuleList] = None,
        n_layers: Optional[int] = None,
    ) -> None:
        """
        Instantiate the quantization layers. If quantization_layer_list is provided,
        it is used directly. Otherwise, a list of quantization layers is created using
        the provided quantization_layer and n_layers.

        Args:
            quantization_layer: The quantization layer to use, if
                quantization_layer_list is not provided.
            quantization_layer_list: The list of quantization layers to use.
            n_layers: The number of quantization layers to create, if
                quantization_layer_list is not provided.

        Returns:
            An nn.ModuleList of quantization layers.
        """
        if quantization_layer_list is not None:
            return quantization_layer_list
        else:
            if n_layers is None:
                raise ValueError(
                    "Since a quantization layer list was not provided, n_layers must be provided."
                )
            if quantization_layer is None:
                raise ValueError(
                    "Either quantization_layer or quantization_layer_list must be provided."
                )
            return nn.ModuleList(
                modules=[copy.deepcopy(quantization_layer) for _ in range(n_layers)]
            )

    def forward(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the quantization layers.

        Args:
            embeddings: The input embeddings to quantize.
                Shape (batch_size, n_features)
        Returns:
            cluster_ids: The cluster ids assigned to the input items.
            all_residuals: The residuals at each layer, unless self.track_residuals is
                False, in which case this is None.
            quantized_embeddings: The full quantized embeddings after passing through
                all quantization layers. These are the sum of the quantized embeddings
                from each layer.
                  Shape (batch_size, n_features).
            quantization_loss: The total quantization loss value, summed across layers.
        """
        cluster_ids = []
        current_residuals = embeddings
        all_residuals = [] if self.track_residuals else None
        quantized_embeddings = torch.zeros_like(embeddings)
        quantization_loss = torch.tensor(0.0).to(self.device)

        for idx, layer in enumerate(self.quantization_layer_list):
            if self.normalize_residuals:
                current_residuals = nn.functional.normalize(
                    current_residuals, dim=-1
                )  # normalize along the feature dimension

            # Determine whether to train the current layer
            train_layer = False
            if self.trainer.state.fn == TrainerFn.FITTING:
                # If we are training layer-wise, we only train the current layer.
                if self.train_layer_wise:
                    train_layer = idx == self.current_layer
                # If we are training all layers simultaneously, there are multiple cases
                else:
                    # If the current layer is already initialized, but not all layers
                    # are initialized, we do not train the current layer because the
                    # initialization of subsequent layers could require a special
                    # optimization step that should not be applied to
                    # already-initialized layers.
                    if (
                        self.quantization_layer_list[idx].is_initialized
                        and not self.quantization_layer_list[-1].is_initialized
                    ):
                        train_layer = False
                    # Otherwise, we always train the first layer, and train subsequent
                    # layers as long as the previous layer produced valid quantized
                    # embeddings, meaning it has been initialized or is currently in
                    # its initialization step.
                    elif idx == 0:
                        train_layer = True
                    elif (
                        self.quantization_layer_list[idx - 1].is_initialized
                        or self.quantization_layer_list[idx - 1].is_initial_step
                    ):
                        train_layer = True

            if train_layer:
                # We call model step inside forward because we need to get the
                # quantization layer's loss, which is computed in the model step
                layer_ids, layer_embeddings, layer_loss = layer.model_step(
                    current_residuals
                )
                quantization_loss += layer_loss
            else:
                layer_ids, layer_embeddings = layer.predict_step(current_residuals)

            cluster_ids.append(layer_ids)  # batch_size
            quantized_embeddings = quantized_embeddings + layer_embeddings
            current_residuals = current_residuals - layer_embeddings
            if self.track_residuals:
                all_residuals.append(current_residuals)

        cluster_ids = torch.stack(cluster_ids, dim=-1)  # batch_size x n_layers
        all_residuals = (
            torch.stack(all_residuals, dim=-1) if self.track_residuals else None
        )

        return cluster_ids, all_residuals, quantized_embeddings, quantization_loss

    def model_step(
        self, model_input: ItemData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and compute the loss for a single batch.

        Args:
            model_input: ItemData consisting of the batch of input features.

        Returns:
            cluster_ids: The cluster ids assigned to the input items.
                    Shape (batch_size, n_layers)
            all_residuals: The residuals at each layer, unless self.track_residuals is
                False, in which case this is None.
                    Shape (batch_size, n_features, n_layers)
            quantization_loss: The cumulative loss from the quantization layers.
            reconstruction_loss: The reconstruction loss.
        """
        input_embeddings = model_input.transformed_features["input_embedding"].to(
            self.device
        )
        normalized_input_embeddings = self.normalization_layer(input_embeddings)
        encoded_embeddings = self.encoder(normalized_input_embeddings)
        (
            cluster_ids,
            all_residuals,
            quantized_embeddings,
            quantization_loss,
        ) = self.forward(encoded_embeddings)

        if (
            self.trainer.state.fn != TrainerFn.PREDICTING
            and self.reconstruction_loss_function is not None
            and self.quantization_layer_list[-1].is_initialized
        ):
            # Compute the reconstruction loss
            reconstructed_embeddings = self.decoder(quantized_embeddings)
            reconstruction_loss = self.reconstruction_loss_function(
                reconstructed_embeddings, normalized_input_embeddings
            )
        else:
            reconstruction_loss = torch.tensor(0.0).to(self.device)

        return (
            cluster_ids,
            all_residuals,
            quantization_loss,
            reconstruction_loss,
        )

    def training_step(self, batch: Tuple[ItemData]) -> torch.Tensor:
        """
        Perform a single training step on a batch of data.

        Args:
            batch: A batch of data of ItemData type wrapped in a Tuple.

        Returns:
            loss: The loss value.
        """
        # Lightning wraps the batch in a tuple for training, we get the batch from
        # position 0. This behavior only happens for training_step.
        model_input: ItemData = batch[0]
        (
            cluster_ids,
            all_residuals,
            quantization_loss,
            reconstruction_loss,
        ) = self.model_step(model_input)

        loss = (
            self.quantization_loss_weight * quantization_loss
            + self.reconstruction_loss_weight * reconstruction_loss
        )
        self.train_loss(loss)
        self.train_quantization_loss(quantization_loss)
        self.train_reconstruction_loss(reconstruction_loss)
        train_dict_to_log = {
            "train/loss": self.train_loss,
            "train/quantization_loss": self.train_quantization_loss,
            "train/reconstruction_loss": self.train_reconstruction_loss,
        }

        with torch.no_grad():
            if self.verbose and self.global_step % self.trainer.log_every_n_steps == 0:
                # Compute verbose ID statistics
                (
                    train_first_residuals_norm_ratio,
                    train_last_residuals_norm_ratio,
                    first_centroids_norm,
                    last_centroids_norm,
                    train_frac_unique_ids,
                    train_mse,
                    train_layer_coverages,
                    train_layer_id_entropies,
                ) = self._compute_output_stats(
                    cluster_ids=cluster_ids,
                    all_residuals=all_residuals,
                    input_embeddings=model_input.transformed_features[
                        "input_embedding"
                    ],
                )
                # Update the metrics
                self.train_first_residuals_norm_ratio(train_first_residuals_norm_ratio)
                self.train_last_residuals_norm_ratio(train_last_residuals_norm_ratio)
                self.first_centroids_norm(first_centroids_norm)
                self.last_centroids_norm(last_centroids_norm)
                self.train_frac_unique_ids(train_frac_unique_ids)
                self.train_mse(train_mse)
                for layer_idx in range(self.n_layers):
                    layer_frac_unique_metric = getattr(
                        self, f"train_layer_coverages_{layer_idx}"
                    )
                    layer_id_entropy_metric = getattr(
                        self, f"train_layer_id_entropy_{layer_idx}"
                    )
                    layer_frac_unique_metric(train_layer_coverages[layer_idx])
                    layer_id_entropy_metric(train_layer_id_entropies[layer_idx])

                train_dict_to_log.update(
                    {
                        "train/last_residuals_norm_ratio": self.train_last_residuals_norm_ratio,
                        "train/first_residuals_norm_ratio": self.train_first_residuals_norm_ratio,
                        "train/first_centroids_norm": self.first_centroids_norm,
                        "train/last_centroids_norm": self.last_centroids_norm,
                        "train/frac_unique_ids": self.train_frac_unique_ids,
                        "train/mse": self.train_mse,
                    }
                )
                train_dict_to_log.update(
                    {
                        f"train/layer_{layer_idx}/frac_layer_coverages": getattr(
                            self,
                            f"train_layer_coverages_{layer_idx}",
                        )
                        for layer_idx in range(self.n_layers)
                    }
                )
                train_dict_to_log.update(
                    {
                        f"train/layer_{layer_idx}/id_entropy": getattr(
                            self,
                            f"train_layer_id_entropy_{layer_idx}",
                        )
                        for layer_idx in range(self.n_layers)
                    }
                )

        self.log_dict(
            train_dict_to_log,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # If a training loop function is passed, we call it with the module and the loss
        # Otherwise we use the automatic optimization provided by Lightning
        if self.training_loop_function is not None:
            if self.train_layer_wise:
                layer_to_check = self.current_layer
            else:
                layer_to_check = -1
            is_initialized = self.quantization_layer_list[layer_to_check].is_initialized

            self.training_loop_function(
                self,
                loss=loss,
                world_size=self.trainer.world_size,
                is_initialized=is_initialized,
            )

        if (
            self.train_layer_wise
            and self.global_step % self.steps_per_layer == 0
            and (
                self.quantization_layer_list[self.current_layer].is_initialized
                or self.current_layer < 0
            )
            and self.current_layer < self.n_layers - 1
        ):
            self.log_if_true(
                f"Finished training layer {self.current_layer} of {self.n_layers}",
                self.verbose,
            )
            self.current_layer += 1

        return loss

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        if hasattr(self, "train_loss"):
            self.train_loss.reset()

        self.current_layer = 0
        for layer in self.quantization_layer_list:
            layer.on_train_start()

        if self.train_layer_wise:
            total_steps = self.trainer.max_steps
            if self.reconstruction_loss_function is None:
                eff_n_layers = self.n_layers
            else:
                eff_n_layers = self.n_layers + 1
                self.current_layer = -1
            self.steps_per_layer = total_steps // eff_n_layers
            self.log_if_true(
                f"Training layers one-at-a-time, each for {self.steps_per_layer} steps."
                " Ensure that early stopping callbacks are disabled.",
                self.verbose,
            )
        else:
            self.log_if_true("Training all layers simultaneously", self.verbose)

        if self.verbose:
            self.train_first_residuals_norm_ratio.reset()
            self.train_last_residuals_norm_ratio.reset()
            self.train_frac_unique_ids.reset()
            self.first_centroids_norm.reset()
            self.last_centroids_norm.reset()
            self.train_mse.reset()
            for layer_idx in range(self.n_layers):
                layer_frac_unique_metric = getattr(
                    self, f"train_layer_coverages_{layer_idx}"
                )
                layer_id_entropy_metric = getattr(
                    self, f"train_layer_id_entropy_{layer_idx}"
                )
                layer_frac_unique_metric.reset()
                layer_id_entropy_metric.reset()

    def _compute_output_stats(
        self,
        cluster_ids: torch.Tensor,
        all_residuals: torch.Tensor,
        input_embeddings: torch.Tensor,
    ) -> Tuple[MeanMetric, MeanMetric, MeanMetric, MeanMetric, MeanMetric]:
        """
        Compute output statistics for the model.

        Args:
            cluster_ids: The cluster ids assigned to the input items.
                    Shape (batch_size, n_layers)
            all_residuals: The residuals at each layer, unless self.track_residuals is
                False, in which case this is None.
                    Shape (batch_size, n_features, n_layers)
            input_embeddings: The cluster embeddings. These are returned for
                debugging purposes.
                    Shape (batch_size, n_features)

        Returns:
            A tuple containing:
                - first_residuals_norm_ratio: The ratio of the norm of the first
                    residuals to the norm of the input embeddings. Note that if
                    self.normalize_residuals is True, this metric is uninformative.
                - last_residuals_norm_ratio: The ratio of the norm of the last
                    residuals to the norm of the input embeddings. Note that if
                    self.normalize_residuals is True, this metric is uninformative.
                - first_centroids_norm: The norm of the centroids of the first
                    quantization layer.
                - last_centroids_norm: The norm of the centroids of the last
                    quantization layer.
                - frac_unique_ids: # distinct item ID sequences / batch_size.
                - mse: The mean squared error of the last residuals (target is 0).
                - layer_coverages: A list containing, for each quantization layer,
                    # distinct IDs / # clusters in that layer.
                - layer_id_entropies: A list containing, for each quantization layer,
                    the batch entropy of the cluster ids in that layer.
        """
        input_embedding_norm = torch.linalg.matrix_norm(input_embeddings)
        first_residuals_norm_ratio = (
            torch.linalg.matrix_norm(all_residuals[:, :, 0]) / input_embedding_norm
        )
        last_residuals_norm = torch.linalg.matrix_norm(all_residuals[:, :, -1])
        last_residuals_norm_ratio = last_residuals_norm / input_embedding_norm
        mse = last_residuals_norm**2 / all_residuals[:, :, -1].numel()

        first_centroids_norm = torch.linalg.matrix_norm(
            self.quantization_layer_list[0].get_centroids()
        )
        last_centroids_norm = torch.linalg.matrix_norm(
            self.quantization_layer_list[-1].get_centroids()
        )

        frac_unique_ids = (
            torch.unique(cluster_ids, dim=0).shape[0] / cluster_ids.shape[0]
        )

        layer_coverages = []
        layer_id_entropies = []
        for layer_idx in range(self.n_layers):
            _, cluster_counts = torch.unique(
                cluster_ids[:, layer_idx], return_counts=True
            )
            cluster_counts = (cluster_counts / cluster_ids.shape[0]).to(self.device)
            entropy = Categorical(probs=cluster_counts).entropy().to(self.device)
            layer_coverages.append(
                cluster_counts.shape[0]
                / self.quantization_layer_list[layer_idx].n_clusters
            )
            layer_id_entropies.append(entropy)

        return (
            first_residuals_norm_ratio,
            last_residuals_norm_ratio,
            first_centroids_norm,
            last_centroids_norm,
            frac_unique_ids,
            mse,
            layer_coverages,
            layer_id_entropies,
        )

    def eval_step(
        self,
        batch: ItemData,
        loss_to_aggregate: MeanMetric,
        first_residuals_norm_ratio_metric: MeanMetric,
        last_residuals_norm_ratio_metric: MeanMetric,
        frac_unique_ids_metric: MeanMetric,
        mse_metric: MeanMetric,
    ):
        """
        Perform a single evaluation step on a batch of data.

        Args:
            batch: A batch of data of ItemData type.
            loss_to_aggregate: The metric for the loss.
            first_residuals_norm_ratio_metric: The metric for the first residuals norm ratio.
            last_residuals_norm_ratio_metric: The metric for the last residuals norm ratio.
            frac_unique_ids_metric: The metric for the fraction of unique ids.
            mse_metric: The metric for the mean squared error.
        """
        (
            cluster_ids,
            all_residuals,
            loss,
        ) = self.model_step(batch)
        loss_to_aggregate(loss)

        (
            first_residuals_norm_ratio,
            last_residuals_norm_ratio,
            _,
            _,
            frac_unique_ids,
            mse,
            _,
            _,
        ) = self._compute_output_stats(
            cluster_ids=cluster_ids,
            all_residuals=all_residuals,
            input_embeddings=batch.transformed_features["input_embedding"],
        )
        last_residuals_norm_ratio_metric(last_residuals_norm_ratio)
        first_residuals_norm_ratio_metric(first_residuals_norm_ratio)
        frac_unique_ids_metric(frac_unique_ids)
        mse_metric(mse)

    def validation_step(self, batch: ItemData, batch_idx: int):
        """
        Perform a single validation step on a batch of data.

        Args:
            batch: A batch of data of ItemData type.
            batch_idx: The index of the batch.
        """
        self.eval_step(
            batch,
            self.val_loss,
            self.val_first_residuals_norm_ratio,
            self.val_last_residuals_norm_ratio,
            self.val_frac_unique_ids,
            self.val_mse,
        )

        val_dict_to_log = {
            "val/loss": self.val_loss,
            "val/first_residuals_norm_ratio": self.val_first_residuals_norm_ratio,
            "val/last_residuals_norm_ratio": self.val_last_residuals_norm_ratio,
            "val/frac_unique_ids": self.val_frac_unique_ids,
            "val/mse": self.val_mse,
        }
        self.log_dict(
            val_dict_to_log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_validation_start(self):
        """Lightning hook that is called when validation begins."""
        self.val_loss.reset()
        self.val_first_residuals_norm_ratio.reset()
        self.val_last_residuals_norm_ratio.reset()
        self.val_frac_unique_ids.reset()
        self.val_mse.reset()

    def test_step(self, batch: ItemData, batch_idx: int) -> None:
        """
        Perform a single test step on a batch of data.

        Args:
            batch: A batch of data of ItemData type.
            batch_idx: The index of the batch.
        """
        self.eval_step(
            batch,
            self.test_loss,
            self.test_first_residuals_norm_ratio,
            self.test_last_residuals_norm_ratio,
            self.test_frac_unique_ids,
            self.test_mse,
        )

        test_dict_to_log = {
            "test/loss": self.test_loss,
            "test/first_residuals_norm_ratio": self.test_first_residuals_norm_ratio,
            "test/last_residuals_norm_ratio": self.test_last_residuals_norm_ratio,
            "test/frac_unique_ids": self.test_frac_unique_ids,
            "test/mse": self.test_mse,
        }
        self.log_dict(
            test_dict_to_log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_test_start(self):
        """Lightning hook that is called when testing begins."""
        self.test_loss.reset()
        self.test_first_residuals_norm_ratio.reset()
        self.test_last_residuals_norm_ratio.reset()
        self.test_frac_unique_ids.reset()
        self.test_mse.reset()

    def predict_step(self, batch: ItemData) -> OneKeyPerPredictionOutput:
        """
        Perform a single prediction step on a batch of data.

        Save the cluster ids assigned to the input items and the corresponding item ids
        in a OneKeyPerPredictionOutput object.

        Args:
            batch: A batch of data of ItemData type.

        Returns:
            model_output: A OneKeyPerPredictionOutput object containing the item
                ids as keys and the cluster ids as predictions.
        """
        cluster_ids, _, _, _ = self.model_step(batch)

        item_ids = [
            item_id.item() if isinstance(item_id, torch.Tensor) else item_id
            for item_id in batch.item_ids
        ]

        model_output = OneKeyPerPredictionOutput(
            keys=item_ids,
            predictions=cluster_ids,
            key_name="item_id",
            prediction_name="cluster_ids",
        )
        return model_output

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            A dictionary containing the optimizer and learning rate scheduler.
        """
        if self.optimizer is not None:
            optimizer = self.optimizer(params=self.trainer.model.parameters())
            if self.scheduler is not None:
                scheduler = self.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/loss",
                        "interval": "step",
                        "frequency": 1,
                    },
                }
            return {"optimizer": optimizer}

    def on_load_checkpoint(self, checkpoint):
        """
        Lightning hook that is called to load the model state from a checkpoint.

        Args:
            checkpoint: The checkpoint to load the model state from.
        """
        self.current_layer = checkpoint["current_layer"]
        for idx, layer in enumerate(self.quantization_layer_list):
            layer.is_initialized = checkpoint["layers_initialized"][idx]
        return super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        """
        Lightning hook that is called to save the model state to a checkpoint.

        Args:
            checkpoint: The checkpoint to save the model state to.
        """
        checkpoint["current_layer"] = self.current_layer
        checkpoint["layers_initialized"] = [
            layer.is_initialized for layer in self.quantization_layer_list
        ]
        # We do not save the input embedding cache as this can be very large
        return super().on_save_checkpoint(checkpoint)

    def log_if_true(self, message: str, condition: bool) -> None:
        """Log a message if condition is True."""
        if condition:
            logging.info(f"Device {self.device}: {message}")
