import json
import os
from importlib.util import find_spec
from typing import Any

from dotenv import load_dotenv
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# logging constants
END_RUN = "end_run"


def convert_dict_to_json_string(data: dict) -> str:
    return json.dumps(data, indent=4)


@rank_zero_only
def login_wandb():
    """
    If WANDB_API_KEY is set in the environment, login to wandb.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Now you can access the WANDB_API_KEY
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        import wandb

        wandb.login(key=wandb_api_key, relogin=True)


@rank_zero_only
def finalize_loggers(trainer: Any, status=END_RUN) -> None:
    """
    Finalize loggers after training is done.

    :param trainer: The Lightning trainer.
    """
    [
        logger.finalize(status)
        for logger in trainer.loggers
        if hasattr(logger, "finalize")
    ]

    if find_spec(
        "wandb"
    ):  # check if wandb is installed. If so, close connection to wandb.
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def log_hyperparameters(
    cfg: DictConfig, model: LightningModule, trainer: Trainer
) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}
    # We resolve the configs to get the actual paths for logging.
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["paths"] = cfg["paths"]
    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data_loading"] = cfg["data_loading"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
