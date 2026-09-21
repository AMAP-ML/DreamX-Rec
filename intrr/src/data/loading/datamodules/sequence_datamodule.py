"""Wrapper around a LightningDataModule."""

import logging
from functools import partial
from typing import Any, Dict, List, Optional

from lightning import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from src.data.loading.components.custom_dataloader import DataloaderWithIterationRetry
from src.data.loading.components.interfaces import BaseDataloaderConfig
from src.data.loading.utils import assign_files_to_workers
from src.utils.file_utils import list_files


class SequenceDataModule(LightningDataModule):
    """A LightningDataModule that encapsulates data splitting, preprocessing,
    parallelization and batching.
    """

    def __init__(
        self,
        train_dataloader_config: Optional[BaseDataloaderConfig] = None,
        val_dataloader_config: Optional[BaseDataloaderConfig] = None,
        test_dataloader_config: Optional[BaseDataloaderConfig] = None,
        predict_dataloader_config: Optional[BaseDataloaderConfig] = None,
    ) -> None:
        """Construct a SequenceDataModule using the provided config files.

        The attributes `map_train_files_per_device`,
        `map_val_files_per_device`, and `map_test_files_per_device` are
        initialized as None, and are later modified by `setup()` to contain
        mappings from device indices to lists of data files assigned to that
        device.

        :param train_dataloader_config: Training dataloader configuration passed
            by Hydra.
        :param val_dataloader_config: Validation dataloader configuration passed
            by Hydra.
        :param test_dataloader_config: Test dataloader configuration passed
            by Hydra.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.stage_to_config = {
            TrainerFn.FITTING: train_dataloader_config,
            TrainerFn.VALIDATING: val_dataloader_config,
            TrainerFn.TESTING: test_dataloader_config,
            TrainerFn.PREDICTING: predict_dataloader_config,
        }

        self.stage_to_file_map: Dict[TrainerFn, Dict[int, List[str]]] = dict()

    def _get_partial_collate_fn(
        self, dataloader_config: BaseDataloaderConfig
    ) -> callable:
        """Prepare the collate function for the dataloader.

        :param config: The dataloader configuration.
        """
        # We need to set the collate function here because we can't pickle
        # lambda functions but the collate fn needs to receive just the batch.

        partial_collate_fn = partial(
            dataloader_config.collate_fn,
            labels=dataloader_config.labels,
            sequence_length=dataloader_config.sequence_length,
            masking_token=dataloader_config.masking_token,
            padding_token=dataloader_config.padding_token,
            oov_token=dataloader_config.get("oov_token", None),
        )
        return partial_collate_fn

    def get_file_suffix_from_config(self, config) -> str:
        return (
            config.dataset_config.file_format
            if getattr(config.dataset_config, "file_format", None)
            else config.dataset_config.data_iterator.get_file_suffix()
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Assign data files to GPUs.

        Note that `self.trainer.world_size` is the total number of GPUs, and is
        equal to (# nodes) x (# GPUs per node).
        This method is called by Lightning before `trainer.fit()`,
        `trainer.validate()`, `trainer.test()`, and `trainer.predict()`, so be
        careful not to execute things like random split twice! Also, it is
        called after `self.prepare_data()` and there is a barrier in between
        which ensures that all the processes proceed to `self.setup()` once the
        data is prepared and available for use.

        :param stage: Unused parameter. Lightning implementation of setup() uses
            stage to determine which dataset splits (train, val, test) to set
            up, but we choose to set up all splits on each call to setup() here.

        :raise AttributeError: If `self.trainer` is not initialized.
        """
        if not hasattr(self, "trainer") or self.trainer is None:
            raise AttributeError(
                f"self.trainer must be initialized before call to setup()."
            )

        for stage, config in self.stage_to_config.items():
            if (
                config is None
            ):  # config is None when we don't want to set up the stages. ie. For inference, we only initialize the predict stage.
                self.stage_to_file_map[stage] = {}
            else:
                # If the stage has not been initialized yet, we assign files to workers based on the suffix passed by the config.
                if stage not in self.stage_to_file_map:
                    list_of_files = list_files(
                        folder_path=config.data_folder,
                        suffix=f"*{self.get_file_suffix_from_config(config)}",
                    )
                    if hasattr(config, "limit_files") and config.limit_files:
                        list_of_files = list_of_files[: config.limit_files]

                    self.stage_to_file_map[stage], _ = assign_files_to_workers(
                        list_of_files=list_of_files,
                        total_workers=self.trainer.world_size,
                        assign_by_size=config.assign_files_by_size,
                        should_shuffle_rows=config.should_shuffle_rows
                        if hasattr(config, "should_shuffle_rows")
                        else False,
                        assign_all_files_per_worker=config.assign_all_files_per_worker
                        if hasattr(config, "assign_all_files_per_worker")
                        else False,
                    )

    def get_dataloader(self, stage: TrainerFn) -> DataLoader[Any]:
        """Construct a DataLoader on a single GPU using config `curr_config`.

        The single GPU is managed by Lightning and corresponds to
        `self.trainer.global_rank`.

        :param curr_config: Config that determines dataloader properties.
        :param map_files_idx_per_device: Map from GPUs to file indices.

        :return: DataLoader running on one GPU that processes the files
            assigned to that GPU, according to `map_files_per_device`.

        :raise AttributeError: If `self.trainer` is not initialized or if setup was not called.
        """
        if not hasattr(self, "trainer"):
            raise AttributeError(
                f"self.trainer must be initialized before call to get_dataloader()."
            )

        if not self.stage_to_file_map[stage]:
            raise AttributeError(f"Stage {stage} must initialize file map.")
        curr_config = self.stage_to_config[stage]

        assign_all_files_per_worker = (
            curr_config.assign_all_files_per_worker
            if hasattr(curr_config, "assign_all_files_per_worker")
            else False
        )

        # We initialize the dataset with the parameters passed on the config.
        dataset = curr_config.dataset_class(
            dataset_config=curr_config.dataset_config,
            data_folder=curr_config.data_folder,
            should_shuffle_rows=curr_config.should_shuffle_rows,
            batch_size=curr_config.batch_size_per_device,
            is_for_training=stage == TrainerFn.FITTING,
            assign_all_files_per_worker=assign_all_files_per_worker,
        )  # type: ignore

        device_file_list = self.stage_to_file_map[stage].get(
            self.trainer.global_rank, []
        )

        dataset.set_list_of_files(list_of_files=device_file_list)
        # set the number of total GPUs and GPU index
        dataset.set_distributed_params(
            total_workers=self.trainer.world_size,
            global_worker_id=self.trainer.global_rank,
        )

        # Any additional parameters for the masking function should be added to
        # the config and passed there. This is required because we can't pickle
        # lambda functions but the collate fn needs to receive just the batch.
        collate_fn_partial = self._get_partial_collate_fn(curr_config)

        if curr_config.num_workers == 0:
            persistent_workers = False
            logging.warning(
                "num_workers is set to 0, persistent_workers will be set to"
                " False as persistent workers require num_workers > 0"
            )
        else:
            persistent_workers = curr_config.persistent_workers

        return (
            DataloaderWithIterationRetry(
                dataset=dataset,
                batch_size=curr_config.batch_size_per_device
                if curr_config.dataset_config.iterate_per_row
                else None,
                num_workers=curr_config.num_workers,  # num workers per GPU
                pin_memory=curr_config.pin_memory,
                persistent_workers=persistent_workers,
                drop_last=curr_config.drop_last
                if curr_config.dataset_config.iterate_per_row
                else False,
                collate_fn=collate_fn_partial,
                timeout=curr_config.timeout,
            ),
        )  # type: ignore

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self.get_dataloader(stage=TrainerFn.FITTING)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self.get_dataloader(stage=TrainerFn.VALIDATING)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self.get_dataloader(stage=TrainerFn.TESTING)

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return self.get_dataloader(stage=TrainerFn.PREDICTING)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`,
        `trainer.validate()`, `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the
        datamodule state.

        :return: A dictionary containing the datamodule state that you want to
            save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule
        state given datamodule `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


class ItemDataModule(SequenceDataModule):
    """A LightningDataModule that encapsulates data splitting, preprocessing,
    parallelization and batching individual (non-sequential) item features.
    """

    def __init__(
        self,
        train_dataloader_config: Optional[BaseDataloaderConfig] = None,
        val_dataloader_config: Optional[BaseDataloaderConfig] = None,
        test_dataloader_config: Optional[BaseDataloaderConfig] = None,
        predict_dataloader_config: Optional[BaseDataloaderConfig] = None,
    ) -> None:
        """Construct a SequenceDataModule using the provided config files.

        The attributes `map_train_files_per_device`,
        `map_val_files_per_device`, and `map_test_files_per_device` are
        initialized as None, and are later modified by `setup()` to contain
        mappings from device indices to lists of data files assigned to that
        device.

        :param train_dataloader_config: Training dataloader configuration passed
            by Hydra.
        :param val_dataloader_config: Validation dataloader configuration passed
            by Hydra.
        :param test_dataloader_config: Test dataloader configuration passed
            by Hydra.
        :param predict_dataloader_config: Prediction dataloader configuration
            passed by Hydra.
        """
        super().__init__(
            train_dataloader_config=train_dataloader_config,
            val_dataloader_config=val_dataloader_config,
            test_dataloader_config=test_dataloader_config,
            predict_dataloader_config=predict_dataloader_config,
        )

    def get_dataloader(self, stage: TrainerFn) -> DataLoader[Any]:
        """Construct a DataLoader on a single GPU using config `curr_config`.

        The single GPU is managed by Lightning and corresponds to
        `self.trainer.global_rank`.

        :param curr_config: Config that determines dataloader properties.
        :param map_files_idx_per_device: Map from GPUs to file indices.

        :return: DataLoader running on one GPU that processes the files
            assigned to that GPU, according to `map_files_per_device`.

        :raise AttributeError: If `self.trainer` is not initialized or if setup was not called.
        """
        if not hasattr(self, "trainer"):
            raise AttributeError(
                f"self.trainer must be initialized before call to get_dataloader()."
            )

        if not self.stage_to_file_map[stage]:
            raise AttributeError(f"Stage {stage} must initialize file map.")
        curr_config = self.stage_to_config[stage]

        assign_all_files_per_worker = (
            curr_config.assign_all_files_per_worker
            if hasattr(curr_config, "assign_all_files_per_worker")
            else False
        )

        assert (
            stage == TrainerFn.FITTING or not assign_all_files_per_worker
        ), "Automatic file assignment should only be disabled for training."

        # We initialize the dataset with the parameters passed on the config.
        dataset = curr_config.dataset_class(
            dataset_config=curr_config.dataset_config,
            data_folder=curr_config.data_folder,
            should_shuffle_rows=curr_config.should_shuffle_rows,
            batch_size=curr_config.batch_size_per_device,
            is_for_training=stage == TrainerFn.FITTING,
            assign_all_files_per_worker=assign_all_files_per_worker,
        )  # type: ignore

        device_file_list = self.stage_to_file_map[stage].get(
            self.trainer.global_rank, []
        )

        dataset.set_list_of_files(list_of_files=device_file_list)
        # set the number of total GPUs and GPU index
        dataset.set_distributed_params(
            total_workers=self.trainer.world_size,
            global_worker_id=self.trainer.global_rank,
        )

        if curr_config.num_workers == 0:
            persistent_workers = False
            logging.warning(
                "num_workers is set to 0, persistent_workers will be set to"
                " False as persistent workers require num_workers > 0"
            )
        else:
            persistent_workers = curr_config.persistent_workers

        return (
            DataloaderWithIterationRetry(
                dataset=dataset,
                batch_size=curr_config.batch_size_per_device
                if curr_config.dataset_config.iterate_per_row
                else None,
                num_workers=curr_config.num_workers,  # num workers per GPU
                pin_memory=curr_config.pin_memory,
                persistent_workers=persistent_workers,
                drop_last=curr_config.drop_last
                if curr_config.dataset_config.iterate_per_row
                else False,
                collate_fn=curr_config.collate_fn,
                timeout=curr_config.timeout,
            ),
        )  # type: ignore
