from typing import List

from torch.utils.data import IterableDataset, get_worker_info

from src.data.loading.components.interfaces import BaseDatasetConfig
from src.utils.pylogger import RankedLogger
import logging
command_line_logger = RankedLogger(__name__, rank_zero_only=True)
command_line_logger.setLevel(logging.DEBUG)

class BaseDataset:
    def __init__(
        self,
        dataset_config: BaseDatasetConfig,
        data_folder: str,
        should_shuffle_rows: bool = False,
        batch_size: int = 1,
        is_for_training: bool = True,
        assign_all_files_per_worker: bool = False,
    ):
        """
        Base class for all datasets. This class is used to set up the dataset and provide the list of files to be used.
        Args:
            dataset_config (BaseDatasetConfig): Configuration for the dataset.
            data_folder (str): Path to the folder where the data is stored.
            should_shuffle_rows (bool): Whether to shuffle the rows of the dataset.
            batch_size (int): Batch size to be used for the dataset.
            is_for_training (bool): Whether the dataset is for training or not.
            assign_all_files_per_worker (bool): Whether to assign all files to each worker or not.
                This will enable each worker to access all files. Each worker will locally shuffle the files.
                This would be useful for small datasets. In smaller datasets, if each worker only observes a subset of the files,
                it may not be able to learn the distribution of the data.
        """
        self.dataset_config = dataset_config
        self.should_shuffle_rows = should_shuffle_rows
        self.data_folder = data_folder
        self.list_of_file_paths = []
        self.batch_size = batch_size
        self.is_for_training = is_for_training
        self.assign_all_files_per_worker = assign_all_files_per_worker

    def set_list_of_files(self, list_of_files: List[str]):
        self.list_of_file_paths = list_of_files

    def set_distributed_params(self, total_workers: int, global_worker_id: int):
        # TODO (lneves): figure out how to do this in LightningDataModule
        self.total_workers = total_workers
        self.global_worker_id = global_worker_id

    def get_worker_id_and_num_workers(self):
        worker_info = get_worker_info()

        if worker_info is None:
            # Single-worker setup (no multiprocessing)
            worker_id = 0
            num_workers = 1
        else:
            # Multi-worker setup
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        self.global_dataloader_worker_id = (
            self.global_worker_id * num_workers + worker_id
        )

        return worker_id, num_workers

    def get_list_of_worker_files(self):
        # Get information about worker and then separate only files that belong to this worker
        worker_id, num_workers = self.get_worker_id_and_num_workers()
        if self.assign_all_files_per_worker:
            worker_files = self.list_of_file_paths
        else:
            worker_files = self.list_of_file_paths[worker_id::num_workers]
        command_line_logger.debug(
            f"GPU Worker: {self.global_worker_id}/{self.total_workers} CPU Worker {worker_id} has {len(worker_files)} files"
        )
        return worker_files

    def setup(self):
        pass


class UnboundedSequenceIterable(BaseDataset, IterableDataset):
    """An unbounded dataset is a dataset that we don't know the size of beforehand.
    For training, we will iterate over the dataset infinitely. For evaluation, we will iterate over the dataset once.
    """

    def __init__(
        self,
        dataset_config: BaseDatasetConfig,
        data_folder: str,
        should_shuffle_rows: bool = False,
        batch_size: int = 1,
        is_for_training: bool = True,
        assign_all_files_per_worker: bool = False,
    ):
        super().__init__(
            dataset_config=dataset_config,
            data_folder=data_folder,
            should_shuffle_rows=should_shuffle_rows,
            batch_size=batch_size,
            is_for_training=is_for_training,
            assign_all_files_per_worker=assign_all_files_per_worker,
        )
        self.data_iterator = dataset_config.data_iterator
        self.dataset_to_iterate = None

    def setup(self):
        # We update each worker's data iterator with the files just for that worker.
        self.data_iterator.update_list_of_file_paths(self.get_list_of_worker_files())
        self.data_iterator = (
            # here we use global_dataloader_worker_id as the seed for shuffling
            # this doesn't matter for the case where workers have non-overlapping files
            # but it does matter for the case where workers have all files
            # (e.g. when using assign_all_files_per_worker)
            # the same seed is used for all workers would cause duplicated examples returned by different workers
            self.data_iterator.shuffle(seed=self.global_dataloader_worker_id)
            if self.should_shuffle_rows
            else self.data_iterator
        )
        self.data_iterator.should_shuffle_rows = self.should_shuffle_rows
        # We provide the flexibility to iterate per row, if per row preprocessing is needed, or per batch.
        self.dataset_to_iterate = (
            self.data_iterator.iterrows()
            if self.dataset_config.iterate_per_row
            else self.data_iterator.iter_batches(self.batch_size)
        )

        command_line_logger.debug(
            f"GLOBAL ID {self.global_dataloader_worker_id} GPU Worker: {self.global_worker_id}/{self.total_workers} with {len(self.data_iterator.list_of_file_paths)} files\
                First five files are: {self.data_iterator.list_of_file_paths[:5]}"
        )

    def __iter__(self):
        if self.dataset_to_iterate is None:
            # If it has not been set up, it means it is a forkserver worker. We need to set it up.
            self.setup()
        # If the dataset is for training, we want to keep iterating over the dataset infinitely.
        # On a streaming dataset, we will always be on Epoch 0.
        finished_iteration = False
        while not finished_iteration:

            for row_or_batch in self.dataset_to_iterate:
                for (
                    preprocessing_function
                ) in self.dataset_config.preprocessing_functions:
                    row_or_batch = preprocessing_function(
                        row_or_batch, dataset_config=self.dataset_config
                    )
                    if row_or_batch is None:
                        break
                if row_or_batch:
                    yield row_or_batch
            # if the dataset is not for training, we stop the loop. Otherwise, we continue.
            finished_iteration = not self.is_for_training
            if not finished_iteration:
                self.setup()
        # We reset the dataset to iterate to None, so that it is set up again in the next iteration.
        # This is required for validation when persitent_workers = True.
        self.dataset_to_iterate = None
        return None