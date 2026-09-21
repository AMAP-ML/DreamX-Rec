import time
from typing import Dict, Optional, Tuple

import torch
import transformers
from torchmetrics.aggregation import BaseAggregator
from torchmetrics import SumMetric

from src.components.eval_metrics import RetrievalEvaluator
from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import SharedKeyAcrossPredictionsOutput
from src.models.components.network_blocks.embedding_aggregator import (
    EmbeddingAggregator,
)
from src.models.modules.base_module import BaseModule


class TransformerBaseModule(BaseModule):
    def __init__(
        self,
        huggingface_model: transformers.PreTrainedModel,
        postprocessor: torch.nn.Module,
        aggregator: EmbeddingAggregator,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_function: torch.nn.Module,
        evaluator: RetrievalEvaluator,
        weight_tying: bool,
        compile: bool,
        training_loop_function: callable = None,
        feature_to_model_input_map: Dict[str, str] = {},
        decoder: torch.nn.Module = None,
        test_with_evaluator: Optional[bool] = None
    ) -> None:

        super().__init__(
            model=huggingface_model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            evaluator=evaluator,
            training_loop_function=training_loop_function,
            test_with_evaluator=test_with_evaluator
        )

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # we remove the nn.Modules as they are already checkpointed to avoid doing it twice

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "huggingface_model",
                "postprocessor",
                "aggregator",
                "decoder",
                "loss_function",
            ],
        )

        self.encoder = huggingface_model
        self.embedding_post_processor = postprocessor
        self.decoder = decoder
        self.aggregator = aggregator
        self.feature_to_model_input_map = feature_to_model_input_map
        self.train_samples = SumMetric()

    def get_embedding_table(self):
        if self.hparams.weight_tying:  # type: ignore
            return self.encoder.get_input_embeddings().weight
        else:
            return self.decoder.weight

    def training_step(
        self,
        batch: Tuple[Tuple[SequentialModelInputData, SequentialModuleLabelData]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data of data (tuple). Because of lightning, the tuple is wrapped in another tuple,
        and the actual batch is at position 0. The batch is a tuple of data where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        start = time.time()
        # Lightning wraps it in a tuple for training, we get the batch from position 0.
        # this behavior only happens for training_step.
        batch = batch[0]
        # Batch is a tuple of model inputs and labels.
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]
        # Batch will be a tuple of model inputs and labels. We use the index here to access them.
        model_output, loss, rec_loss = self.model_step(
            model_input=model_input, label_data=label_data
        )

        # update and log train samples
        batch_size = model_input.mask.shape[0]
        self.train_samples(batch_size)
        self.log(
            "train/samples",
            self.train_samples,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # update and log metrics. Will only be logged at the interval specified in the logger config
        self.train_loss(loss)
        # checks logging interval and logs the loss
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.rec_loss(rec_loss)
        self.log(
            "train/rec_loss",
            self.rec_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # If a training loop function is passed, we call it with the module and the loss.
        # otherwise we use the automatic optimization provided by lightning
        if self.training_loop_function is not None:
            self.training_loop_function(self, loss)

        end = time.time()
        self.log("train/train_step_time", end - start, on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size)

        return loss

    def eval_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        loss_to_aggregate: BaseAggregator,
    ):
        """Perform a single evaluation step on a batch of data from the validation or test set.
        The method will update the metrics and the loss that is passed.
        """
        # Batch is a tuple of model inputs and labels.
        start = time.time()
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]

        batch_size = model_input.mask.shape[0]
        model_output_before_aggregation, loss = self.model_step(
            model_input=model_input, label_data=label_data
        )

        model_output_after_aggregation = self.aggregator(
            model_output_before_aggregation, model_input.mask
        )

        # Updates metrics inside evaluator.
        self.evaluator(
            query_embeddings=model_output_after_aggregation,
            key_embeddings=self.get_embedding_table().to(
                model_output_after_aggregation.device
            ),
            # TODO: (lneves) hardcoded for now, will need to change for multiple features
            labels=list(label_data.labels.values())[0].to(
                model_output_after_aggregation.device
            ),
        )
        loss_to_aggregate(loss)
        end = time.time()
        # 只记录 epoch 级别的平均值，不记录每个 step，与训练的日志频率保持一致
        self.log("val/eval_step_time", end - start, on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size)

    def predict_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        batch_idx: int,
    ):
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data of data (tuple) where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        """
        model_input: SequentialModelInputData = batch[0]
        model_output_before_aggregation, _ = self.model_step(model_input=model_input)

        model_output_after_aggregation = self.aggregator(
            model_output_before_aggregation, model_input.mask
        )
        # TODO(lneves): Currently passing batch idx, change it to user_id and allow for the user to specify the key and prediction names.
        model_output = SharedKeyAcrossPredictionsOutput(
            key=batch_idx,
            predictions=model_output_after_aggregation,
            key_name=self.prediction_key_name,
            prediction_name=self.prediction_name,
        )
        return model_output