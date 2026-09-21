from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)
from src.utils.file_utils import (
    get_last_modified_file,
    has_no_extension,
    list_subfolders,
)
from src.utils.logging_utils import finalize_loggers
from src.utils.restart_job_utils import get_attribute_from_metadata_file
from src.utils.utils import has_class_object_inside_list

command_line_logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class PipelineModules:
    cfg: DictConfig
    datamodule: LightningDataModule
    model: LightningModule
    # We use the plural form to match the names used by lightning
    callbacks: List[Callback]
    loggers: List[Logger]
    trainer: Trainer


def update_cfg_with_most_recent_checkpoint_path(cfg: DictConfig) -> str:
    """
    Updates the configuration with the most recent checkpoint path if the job is a retry, a checkpoint callback exists,
    and a checkpoint file is found.

    This function is useful for resuming training from the most recent checkpoint when a job is restarted.
    It checks if the current run is part of a retry (using restart metadata), and if so, it looks for the
    most recently modified checkpoint file in the checkpoint directory to use instead of the initial checkpoint.

    Args:
        cfg (DictConfig): The configuration dictionary containing training parameters.

    Returns:
        DictConfig: The updated configuration dictionary with the most recent checkpoint path.
    """

    ckpt_path = cfg.get("ckpt_path", None)

    if (
        ckpt_path is not None
        and has_no_extension(ckpt_path)
        and cfg.get("should_retrieve_latest_ckpt_path", False)
    ):
        # If a path to a folder is passed, we assume it contains folders with versions of checkpoints.
        # We expect those folders to be named using a timestamp.
        checkpoint_folders = list_subfolders(ckpt_path)
        if len(checkpoint_folders) > 0:
            # We sort them in reverse order to get the most recent one.
            checkpoint_folders.sort(reverse=True)
            # We take the first one, which is the most recent one.
            latest_ckpt_folder = checkpoint_folders[0]
            last_modified = get_last_modified_file(
                folder_path=latest_ckpt_folder, suffix="*.ckpt"
            )
            if len(last_modified) > 0:
                ckpt_path = last_modified
                command_line_logger.info(
                    f"Found most recent checkpoint path: {ckpt_path}. Starting job from this checkpoint."
                )

    # If there is a checkpoint callback running, and the restart_metadata file shows we are not on the first run,
    # we check if there are checkpoints in the checkpoint folder and restart from there instead of the initial checkpoint.
    if (
        cfg.get("callbacks")  # has callbacks
        and cfg.callbacks.get("model_checkpoint")  # has checkpoint callback
        and cfg.callbacks.get("restart_job")  # has retry callback
        and get_attribute_from_metadata_file(
            f"{cfg.callbacks.restart_job.metadata_dir}/restart_metadata.json",
            "current_run",
        )
        > 0  # current run is part of a retry
    ):
        checkpoint_folder = cfg.callbacks.model_checkpoint.dirpath
        # We check if there are files with the extension .ckpt in the checkpoint folder. If so, we get the latest one.
        last_modified = get_last_modified_file(
            folder_path=checkpoint_folder, suffix="*.ckpt"
        )
        if len(last_modified) > 0:
            ckpt_path = last_modified
            command_line_logger.info(
                f"Found most recent checkpoint path: {ckpt_path}. Starting job from this checkpoint."
            )

    cfg.ckpt_path = ckpt_path
    return cfg


def initialize_pipeline_modules(
    cfg: DictConfig,
) -> PipelineModules:
    """
    Initialize and instantiate various objects required for running pipelines.

    Args:
        cfg (DictConfig): Configuration object containing parameters for data, model, callbacks, logger, and trainer.

    Returns:
        PipelineModules: A dataclass containing the instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    command_line_logger.info(
        f"Instantiating datamodule <{cfg.data_loading.datamodule._target_}>"
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data_loading.datamodule
    )

    command_line_logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    command_line_logger.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    command_line_logger.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    command_line_logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    cfg = update_cfg_with_most_recent_checkpoint_path(cfg)

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        # The default behavior for lightning it to set `enable_checkpointing` and
        # `enable_model_summary` to True, which might be misleading when we are trying to
        # debug. We change the default to False, but this can be overriden by either
        # setting the parameters in the config file or passing the callbacks as part
        # of the callbacks yaml.
        enable_checkpointing=cfg.trainer.get(
            "enable_checkpointing",
            has_class_object_inside_list(callbacks, ModelCheckpoint),
        ),
        enable_model_summary=cfg.trainer.get(
            "enable_model_summary",
            has_class_object_inside_list(callbacks, ModelSummary),
        ),
        logger=loggers,
    )

    pipeline_modules = PipelineModules(
        cfg=cfg,
        datamodule=datamodule,
        model=model,
        callbacks=callbacks,
        loggers=loggers,
        trainer=trainer,
    )

    return pipeline_modules


@contextmanager
def pipeline_launcher(cfg: DictConfig):
    """
    Launches the pipeline with the given configuration and logger.
    Args:
        cfg (DictConfig): Configuration object containing pipeline settings.
        log (RankedLogger): Logger object for logging information.
    Yields:
        PipelineModules: A dataclass containing the instantiated objects.
    Raises:
        Exception: Propagates any exception that occurs during pipeline initialization.
    Notes:
        - If the configuration contains a logger, hyperparameters will be logged.
        - Ensures that loggers are finalized and profiler output is saved even if the task fails.
    """

    pipeline_modules = None
    try:
        pipeline_modules: PipelineModules = initialize_pipeline_modules(cfg)
        # Log hyperparameters if loggers are present
        if len(pipeline_modules.loggers) > 0:
            command_line_logger.info("Logging hyperparameters!")
            log_hyperparameters(cfg, pipeline_modules.model, pipeline_modules.trainer)
        yield pipeline_modules
    except Exception as ex:
        raise ex
    finally:
        # We add the try catch to make sure the loggers are finalized even if the task fails.
        if pipeline_modules is not None:
            finalize_loggers(pipeline_modules.trainer)
