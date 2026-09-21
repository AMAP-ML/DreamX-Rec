from typing import Any, Dict, Optional, Union

import torch
import transformers
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.aggregation import BaseAggregator

from src.components.eval_metrics import Evaluator
from src.utils.pylogger import RankedLogger

command_line_logger = RankedLogger(__name__, rank_zero_only=True)


class BaseModule(LightningModule):
    def __init__(
        self,
        model: Union[torch.nn.Module, transformers.PreTrainedModel],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_function: torch.nn.Module,
        evaluator: Evaluator,
        training_loop_function: callable = None,
        test_with_evaluator: Optional[bool] = None
    ) -> None:
        """
        Args:
            model: The model to train.
            optimizer: The optimizer to use for the model.
            scheduler: The scheduler to use for the model.
            loss_function: The loss function to use for the model.
            evaluator: The evaluator to use for the model.
            training_loop_function: The training loop function to use for the model, in case it is different than the default one.
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.evaluator = evaluator
        self.training_loop_function = training_loop_function
        # We use setters to set the prediction key and name.
        self._prediction_key_name = None
        self._prediction_name = None

        if self.training_loop_function is not None:
            self.automatic_optimization = False

        if self.evaluator:  # For inference, evaluator is not set.
            for metric_name, metric_object in self.evaluator.metrics.items():
                setattr(self, metric_name, metric_object)

            # for averaging loss across batches
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()
            self.test_loss = MeanMetric()
            self.rec_loss = MeanMetric()

            # Create a separate evaluator for test metrics to avoid interference with validation metrics
            self.test_evaluator = None

        # Whether to use the separate test evaluator
        self.test_with_evaluator = test_with_evaluator

    @property
    def prediction_key_name(self) -> Optional[str]:
        return self._prediction_key_name

    @prediction_key_name.setter
    def prediction_key_name(self, value: str) -> None:
        command_line_logger.debug(f"Setting prediction_key_name to {value}")
        self._prediction_key_name = value

    @property
    def prediction_name(self) -> Optional[str]:
        return self._prediction_name

    @prediction_name.setter
    def prediction_name(self, value: str) -> None:
        command_line_logger.debug(f"Setting prediction_name to {value}")
        self._prediction_name = value

    def forward(
        self,
        **kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Inherit from this class and implement the forward method."
        )

    def model_step(
        self,
        model_input: Any,
        label_data: Optional[Any] = None,
    ):
        raise NotImplementedError(
            "Inherit from this class and implement the model_step method."
        )

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.evaluator.reset()
        self.train_loss.reset()
        self.test_loss.reset()
        self.rec_loss.reset()
        print("on_train_start")

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when a validation epoch starts."""
        self.val_loss.reset()
        self.evaluator.reset()
        print("on_validation_epoch_start: ", self.global_step)

    def on_test_epoch_start(self):
        self.test_loss.reset()
        self.evaluator.reset()
        print("on_test_epoch_start")
        if hasattr(self, '_compute_all_items_cache'):
            # try:
            command_line_logger.info("Computing and caching all items data after training epoch...")
            self._compute_all_items_cache()
            command_line_logger.info("Successfully cached all items data")
            # except Exception as e:
            #     command_line_logger.warning(f"Failed to compute items cache: {str(e)}")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.log("val/loss", self.val_loss, sync_dist=False, prog_bar=True, logger=True)
        self.log_metrics("val")
        print("global train step:", self.global_step)

        # 打印 IntRREncoderDecoder 模型的 temp_nn 参数
        if hasattr(self, 'temp_nn'):
            command_line_logger.info(f"Logging model parameter temp_nn: {self.temp_nn}")
        else:
            command_line_logger.info("Model does not have temp_nn parameter")
        if hasattr(self, '_compute_all_items_cache'):
            # try:
            command_line_logger.info("Computing and caching all items data after training epoch...")
            self._compute_all_items_cache()
            command_line_logger.info("Successfully cached all items data")
            # except Exception as e:
            #     command_line_logger.warning(f"Failed to compute items cache: {str(e)}")
        # Also compute test metrics during validation
        # Note: This requires test data to be available during training
        # You may need to adjust your data module to make test data available during training
        if self.test_with_evaluator:
            command_line_logger.info("Running test evaluation during validation...")
            try:
                # Check if trainer and datamodule are available
                if not hasattr(self, 'trainer') or self.trainer is None:
                    command_line_logger.warning("Trainer not available during validation")
                    return

                if not hasattr(self.trainer, 'datamodule') or self.trainer.datamodule is None:
                    command_line_logger.warning("Datamodule not available during validation")
                    return

                # Check if test_dataloader method exists and is callable
                if not hasattr(self.trainer.datamodule, 'test_dataloader'):
                    command_line_logger.warning("test_dataloader method not found in datamodule")
                    return

                if not callable(getattr(self.trainer.datamodule, 'test_dataloader')):
                    command_line_logger.warning("test_dataloader is not callable")
                    return

                command_line_logger.info("Running test evaluation during validation...")

                # Reset test metrics
                self.test_loss.reset()
                # Initialize separate test evaluator to avoid interference with validation metrics
                self._initialize_test_evaluator()

                # Run test evaluation
                test_dataloader = self.trainer.datamodule.test_dataloader()
                if test_dataloader is not None:
                    # Handle the case where test_dataloader returns a tuple with the actual dataloader
                    if isinstance(test_dataloader, tuple):
                        actual_dataloader = test_dataloader[0]
                    else:
                        actual_dataloader = test_dataloader

                    command_line_logger.info(f"Test dataloader found, starting test evaluation...")
                    batch_count = 0
                    for batch in actual_dataloader:
                        # Move batch to the same device as the model
                        batch = self._move_batch_to_device(batch)
                        self._test_step_with_separate_evaluator(batch, batch_count)
                        if batch_count % 100 == 0:  # Print progress every 100 batches
                            command_line_logger.debug(f"Processed {batch_count} test batches")
                        batch_count += 1

                    command_line_logger.info(f"Completed test evaluation with {batch_count} batches")

                    # Log test metrics using the separate test evaluator
                    self.log("test/loss", self.test_loss, sync_dist=False, prog_bar=True, logger=True)
                    self._log_test_metrics("test")
                    command_line_logger.info("Test metrics computed and logged successfully")
                else:
                    command_line_logger.warning("test_dataloader returned None")
            except Exception as e:
                command_line_logger.error(f"Error during test in validation: {str(e)}")
                import traceback
                command_line_logger.error(traceback.format_exc())
                # This is not an error condition, just skip test evaluation
                pass
        print("on_validation_epoch_end")


    def on_test_epoch_end(self) -> None:
        self.log(
            "test/loss", self.test_loss, sync_dist=False, prog_bar=True, logger=True
        )
        self.log_metrics("test")
        print("on_test_epoch_end")



    def on_exception(self, exception):
        self.trainer.should_stop = True  # stop all workers
        self.trainer.logger.finalize(status="failure")

    def log_metrics(
        self,
        prefix: str,
        on_step=False,
        on_epoch=True,
        # We use sync_dist=False by default because, if using retrieval metrics, those are already synchronized. Change if using
        # different metrics than the default ones.
        sync_dist=False,
        logger=True,
        prog_bar=False,
        call_compute=False,
    ) -> Dict[str, Any]:

        metrics_dict = {
            f"{prefix}/{metric_name}": metric_object.compute()
            if call_compute
            else metric_object
            for metric_name, metric_object in self.evaluator.metrics.items()
        }

        # Print metrics to console for debugging
        command_line_logger.info(f"Logging {prefix} metrics: {list(metrics_dict.keys())}")

        # Also log to CSV file directly if needed
        if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, 'loggers'):
            for logger_instance in self.trainer.loggers:
                logger_type = type(logger_instance).__name__
                command_line_logger.debug(f"Logging to {logger_type}: {prefix} metrics")

        self.log_dict(
            metrics_dict,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
            logger=logger,
            prog_bar=prog_bar,
        )

        return metrics_dict

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # if self.hparams.compile and stage == "fit":
        #     self.net = torch.compile(self.net)
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
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

    def _move_batch_to_device(self, batch: Any) -> Any:
        """
        Move batch data to the same device as the model.

        Args:
            batch: The batch data which can be a tensor, list, tuple, or dict

        Returns:
            The batch data moved to the model's device
        """
        if batch is None:
            return batch

        # Handle different batch types
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            # Recursively move each item in the list/tuple
            moved_items = []
            for item in batch:
                if hasattr(item, 'to') and hasattr(item, 'device'):
                    # This is a tensor-like object
                    moved_items.append(item.to(self.device))
                elif hasattr(item, '__dict__'):
                    # This might be a custom data class, try to move its attributes
                    moved_item = self._move_object_to_device(item)
                    moved_items.append(moved_item)
                else:
                    # Keep non-tensor items as is
                    moved_items.append(item)
            return type(batch)(moved_items)
        elif isinstance(batch, dict):
            # Move each value in the dictionary
            return {key: self._move_batch_to_device(value) for key, value in batch.items()}
        elif hasattr(batch, '__dict__'):
            # Handle custom objects with attributes
            return self._move_object_to_device(batch)
        else:
            # Return as is for non-tensor types
            return batch

    def _move_object_to_device(self, obj: Any) -> Any:
        """
        Move attributes of a custom object to the model's device.

        Args:
            obj: The object whose tensor attributes should be moved

        Returns:
            The object with tensor attributes moved to the model's device
        """
        # Create a copy of the object to avoid modifying the original
        import copy
        moved_obj = copy.copy(obj)

        # Move tensor attributes to the correct device
        for attr_name in dir(moved_obj):
            if not attr_name.startswith('_'):  # Skip private attributes
                try:
                    attr_value = getattr(moved_obj, attr_name)
                    if hasattr(attr_value, 'to') and hasattr(attr_value, 'device'):
                        setattr(moved_obj, attr_name, attr_value.to(self.device))
                    elif isinstance(attr_value, (list, tuple, dict)):
                        setattr(moved_obj, attr_name, self._move_batch_to_device(attr_value))
                except (AttributeError, TypeError):
                    # Skip attributes that can't be accessed or moved
                    continue

        return moved_obj

    def _initialize_test_evaluator(self):
        """
        Initialize a separate evaluator for test metrics to avoid interference with validation metrics.
        """
        if self.evaluator is not None:
            # Create a deep copy of the evaluator for test metrics
            import copy
            self.test_evaluator = copy.deepcopy(self.evaluator)
            # Reset the test evaluator to start fresh
            self.test_evaluator.reset()
            # Move to correct device
            if hasattr(self.test_evaluator, 'to'):
                self.test_evaluator.to(self.device)
            command_line_logger.debug("Initialized separate test evaluator")

    def _test_step_with_separate_evaluator(self, batch: Any, batch_idx: int) -> None:
        """
        Perform a test step using the separate test evaluator.
        """
        # Temporarily replace the evaluator with the test evaluator
        original_evaluator = self.evaluator
        if self.test_evaluator is not None:
            self.evaluator = self.test_evaluator

        try:
            # Call the regular eval_step but with test_loss
            # This will now use the test_evaluator instead of the original evaluator
            self.eval_step(batch, self.test_loss)
        finally:
            # Always restore the original evaluator
            self.evaluator = original_evaluator

    def _log_test_metrics(self, prefix: str) -> Dict[str, Any]:
        """
        Log test metrics using the separate test evaluator.
        """
        if self.test_evaluator is None:
            command_line_logger.warning("Test evaluator not initialized, falling back to regular evaluator")
            return self.log_metrics(prefix)

        metrics_dict = {
            f"{prefix}/{metric_name}": metric_object.compute()
            if hasattr(metric_object, 'compute')
            else metric_object
            for metric_name, metric_object in self.test_evaluator.metrics.items()
        }

        # Print metrics to console for debugging
        command_line_logger.info(f"Logging {prefix} metrics: {list(metrics_dict.keys())}")

        # Also log to CSV file directly if needed
        if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, 'loggers'):
            for logger_instance in self.trainer.loggers:
                logger_type = type(logger_instance).__name__
                command_line_logger.debug(f"Logging to {logger_type}: {prefix} metrics")

        self.log_dict(
            metrics_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            logger=True,
            prog_bar=False,
        )

        return metrics_dict

    def eval_step(self, batch: Any, loss_to_aggregate: BaseAggregator):
        raise NotImplementedError("eval_step method must be implemented.")

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data of data (tuple) where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        """
        self.eval_step(batch, self.val_loss)

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data of data (tuple) where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        """
        self.eval_step(batch, self.test_loss)
