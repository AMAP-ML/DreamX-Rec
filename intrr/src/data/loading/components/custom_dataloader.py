import time

from torch.utils.data import _utils
from torch.utils.data.dataloader import (
    DataLoader,
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
)

from src.utils.pylogger import RankedLogger

command_line_logger = RankedLogger(__name__, rank_zero_only=False)


class _MultiProcessingDataLoaderIterWithRetry(_MultiProcessingDataLoaderIter):
    def __init__(self, loader, max_retries=3):
        super().__init__(loader)
        self._max_retries = max_retries

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        # Note that the max wait time will be timeout * (max_retries + 1) due to original call + all retries.
        # This won't fix issues related to corrupted data, but can help with network issues.
        retries = 0
        for _ in range(self._max_retries):
            start_time = time.monotonic()
            status, data = super()._try_get_data(timeout)
            if status:
                return (True, data)
            end_time = time.monotonic()
            # If it took less than the timeout time, it means the issue was an empty queue.
            # If that happens or if we have reached our last retry, we return False. Else, we retry.
            # We need this because original _try_get_data will return False if the queue is empty or
            # if the timeout is reached. We need a way to differentiate and only retry if there is a timeout.
            if end_time - start_time < timeout or retries + 1 == self._max_retries:
                return (False, None)
            command_line_logger.warning(
                f"Retrying after timeout... Retry {retries + 1}/{self._max_retries}"
            )
            retries += 1


class DataloaderWithIterationRetry(DataLoader):
    def __init__(self, max_retries=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_retries = max_retries

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIterWithRetry(
                self, max_retries=self._max_retries
            )
