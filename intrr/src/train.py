from typing import Any, Dict, Optional, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
from src.utils import RankedLogger, extras
from src.utils.custom_hydra_resolvers import *
from src.utils.launcher_utils import pipeline_launcher
from src.utils.restart_job import LocalJobLauncher

command_line_logger = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("medium")


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # Pipeline launcher initializes the modules needed for the pipeline to run.
    # It also serves as a context manager, so all resources are properly closed after the pipeline is done.
    with pipeline_launcher(cfg) as pipeline_modules:

        if cfg.get("train"):
            command_line_logger.info("Starting training!")
            pipeline_modules.trainer.fit(
                model=pipeline_modules.model,
                datamodule=pipeline_modules.datamodule,
                ckpt_path=cfg.get("ckpt_path"),
            )
        train_metrics = pipeline_modules.trainer.callback_metrics

        if cfg.get("test"):
            command_line_logger.info("Starting testing!")
            ckpt_path = None
            # Check if a checkpoint callback is available and if it has a best model path.
            # Note that if multiple checkpoint callbacks are used, only the first one will be used
            # to determine the best model path for testing.
            checkpoint_callback = getattr(
                pipeline_modules.trainer, "checkpoint_callback", None
            )
            if checkpoint_callback:
                ckpt_path = getattr(checkpoint_callback, "best_model_path", None)
                if ckpt_path == "":
                    ckpt_path = None
            if not ckpt_path:
                command_line_logger.warning(
                    "Best checkpoint not found! Using current weights for testing..."
                )
            pipeline_modules.trainer.test(
                model=pipeline_modules.model,
                datamodule=pipeline_modules.datamodule,
                ckpt_path=ckpt_path,
            )
            command_line_logger.info(f"Best ckpt path: {ckpt_path}")

        test_metrics = pipeline_modules.trainer.callback_metrics

        # merge train and test metrics
        metric_dict = {**train_metrics, **test_metrics}

        command_line_logger.info(f"Metrics: {metric_dict}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    job_launcher = LocalJobLauncher(cfg=cfg)
    job_launcher.launch(function_to_run=train)


if __name__ == "__main__":
    main()
