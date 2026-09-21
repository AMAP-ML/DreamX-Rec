import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

import psutil
from lightning import Trainer

from src.utils.file_utils import (
    file_exists_local_or_remote,
    load_json,
    open_local_or_remote,
)
from src.utils.pylogger import RankedLogger

import torch
import torch.distributed as dist
from lightning.pytorch.strategies.launchers import _SubprocessScriptLauncher
from lightning.pytorch.trainer.connectors.signal_connector import _get_sigkill_signal

command_line_logger = RankedLogger(__name__, rank_zero_only=True)
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class JobCheckpointMetadata:
    """
    A class that stores metadata for job checkpointing and restarts.

    Attributes:
        start_time (str): ISO formatted timestamp of when the job started. Defaults to the current time.
        restarts (List[Dict[str, Any]]): List of dictionaries containing information about previous restarts.
        current_run (int): Counter for the current run number. Starts at 0 and increments with each restart.
        used_ports (List[str]): List of ports that have been used by previous runs.
        world_size (int): The total number of processes participating in the distributed job.
        node_rank (int): The rank of this node in the distributed job.
        master_addr (str): The address of the master node for distributed training.
        original_args (List[str]): The original command-line arguments used to start the job.

    Methods:
        to_dict(): Converts the metadata object to a dictionary representation.
    """

    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    restarts: List[Dict[str, Any]] = field(default_factory=list)
    current_run: int = 0
    used_ports: List[str] = field(default_factory=list)
    world_size: int = 0
    node_rank: int = 0
    master_addr: str = ""
    original_args: List[str] = field(default_factory=lambda: sys.argv)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "restarts": self.restarts,
            "current_run": self.current_run,
            "used_ports": self.used_ports,
            "world_size": self.world_size,
            "node_rank": self.node_rank,
            "master_addr": self.master_addr,
            "original_args": self.original_args,
        }


@dataclass
class RestartMetadata:
    """
    A class to store metadata about a job restart.

    Attributes:
        time (str): The time when the job was restarted.
        exception (str): The exception that caused the job to be restarted.
        run_number (int): The number of times the job has been run.

    Methods:
        to_dict(): Converts the metadata to a dictionary.
    """

    time: str
    exception: str
    run_number: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "exception": self.exception,
            "run_number": self.run_number,
        }


def load_metadata_from_local_or_remote(metadata_path: str) -> JobCheckpointMetadata:
    """
    Loads a JobCheckpointMetadata from a local or remote filepath, if available.

    This function attempts to load and deserialize a JobCheckpointMetadata from the given path.
    If the file doesn't exist or is invalid, it returns an empty JobCheckpointMetadata object and logs a warning.

    Args:
        metadata_path (str): Path to the metadata file, either local or remote.

    Returns:
        JobCheckpointMetadata: The loaded metadata if file exists and is valid, otherwise an empty metadata object.
    """
    command_line_logger.info(f"Trying to load metadata from {metadata_path}")
    if file_exists_local_or_remote(metadata_path):
        try:
            metadata_dict = load_json(metadata_path)
            command_line_logger.info(f"Metadata loaded successfully from {metadata_path}")
            return JobCheckpointMetadata(**metadata_dict)
        except Exception as e:
            command_line_logger.warning(
                f"Failed to load metadata from {metadata_path}: {e}. Creating empty metadata."
            )
            return JobCheckpointMetadata()
    else:
        command_line_logger.warning(
            f"Metadata file not found at {metadata_path}. Creating empty metadata."
        )
        return JobCheckpointMetadata()


def save_metadata_to_local_or_remote(
    metadata: JobCheckpointMetadata, metadata_path: str
) -> None:
    """
    Save job checkpoint metadata to a local or remote file.
    This function serializes the metadata object to JSON and writes it to the specified path.
    It handles both local filesystem paths and remote paths (e.g., Google Cloud Storage 'gs://' paths)
    using the open_local_or_remote utility function.
    Args:
        metadata (JobCheckpointMetadata): The metadata object to save
        metadata_path (str): Path where to save the metadata, can be a local path or a remote path (e.g., gs://...)
    Returns:
        None
    Logs:
        - Info message indicating where metadata is being saved
    """

    command_line_logger.info(
        f"Saving metadata to {metadata_path}. {metadata.to_dict()}"
    )

    # Convert metadata to JSON string
    json_content = json.dumps(metadata.to_dict(), indent=2)

    # Use the open_local_or_remote function which should handle gs:// paths
    with open_local_or_remote(metadata_path, "w") as f:
        f.write(json_content)


def get_attribute_from_metadata_file(metadata_path: str, attribute: str) -> Any:
    """
    Extracts a specified attribute from a metadata file.

    This function loads the metadata from either a local or remote file and returns
    the value of the specified attribute from the loaded metadata object.

    Args:
        metadata_path (str): The path to the metadata file. This can be a local file path
                             or a remote URL.
        attribute (str): The name of the attribute to retrieve from the metadata object.

    Returns:
        Any: The value of the specified attribute from the metadata object.

    Raises:
        AttributeError: If the specified attribute doesn't exist in the metadata object.

    Note:
        This function relies on the `load_metadata_from_local_or_remote` function to handle
        the loading of the metadata file.
    """
    metadata = load_metadata_from_local_or_remote(metadata_path)
    attribute_value = getattr(metadata, attribute, None)
    command_line_logger.info(
        f"Retrieved {attribute}: {attribute_value} from metadata {metadata_path}"
    )
    return attribute_value


def _is_process_running(proc: psutil.Process) -> bool:
    """
    Check if a process is still running.

    This function polls the process to check its status. It returns True if the process is still running (i.e., its return code is None),
    and False if the process has terminated (i.e., its return code is not None).

    Args:
        proc (psutil.Process): The process to check.

    Returns:
        bool: True if the process is still running, False otherwise.
    """
    # Check if the process is running by checking the return code.
    # If the process is still running, poll() will return None.
    # If the process has finished, poll() will return the return code.
    proc.poll()
    return proc.returncode is None


def clean_up_resources(
    trainer: Optional[Trainer] = None, exception: Optional[Exception] = None
) -> None:
    """Clean up distributed processes and CUDA resources."""
    if dist.is_initialized():
        command_line_logger.info("Cleaning up distributed process group")
        dist.destroy_process_group()

    if torch.cuda.is_available():
        command_line_logger.info("Clearing CUDA cache")
        torch.cuda.empty_cache()

    if trainer is not None:
        command_line_logger.info("Tearing down trainer")
        trainer.strategy.on_exception(exception)
        launcher = trainer.strategy.launcher if trainer.strategy is not None else None
        trainer._teardown()
        if isinstance(launcher, _SubprocessScriptLauncher):
            launcher.kill(_get_sigkill_signal())
