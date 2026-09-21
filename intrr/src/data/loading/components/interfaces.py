from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import transformers
from torch.utils.data import IterableDataset

from src.data.loading.components.iterators import RawDataIterator


class BaseDatasetConfig:
    """
    Class to provide base typing for dataset configurations. Should be inherited by all dataset configurations.
    """

    def __init__(self):
        pass

    def get(self, attribute: str, default=None):
        return getattr(self, attribute, default)


class BaseDataloaderConfig:
    """
    Class to provide base typing for dataloader configurations. Should be inherited by all dataloader configurations.
    """

    def __init__(self):
        pass

    def get(self, attribute: str, default=None):
        return getattr(self, attribute, default)


@dataclass
class SequenceDatasetConfig(BaseDatasetConfig):
    """The generic dataset configuration class for datasets of sequence data.

    Parameters:
    ----------
    user_id_field: str
        The user id field name.
    data_iterator: RawDataIterator
        The raw data iterator.
    preprocessing_functions: list[callable]
        The list of preprocessing functions. Should be in the order they must be applied.
    num_placeholder_tokens_map: Optional[dict]
        The number of placeholder tokens map.
    iterate_per_row: bool
        Whether to iterate per row or per batches.
    keep_user_id: bool
        Whether to keep the user id feature in the batches.
    field_type_map: Optional[dict]
        The field type map.
    min_sequence_length: int
        The minimum sequence length. Only works if iterating per row.
    feature_map: Optional[dict]
        maps the feature names to the desired feature names.
    iterate_per_row: bool
        Whether to iterate per row or per batches.
    features_to_consider: list[str]
        List of features to consider. If not specified, consider all features.
    file_format: str
        The file format of the dataset files. If not specified, the data iterator's `get_file_suffix`
        method will be used to determine the file format.
        For example, if the data iterator reads tfrecord files at the first level of the data_folder,
        this can be set to "tfrecord.gz". If we want to retrieve all tfrecord files in subdirectories as well,
        we can set it to "*/*tfrecord.gz".
    """

    user_id_field: str
    data_iterator: RawDataIterator
    preprocessing_functions: list[callable]  # type: ignore
    iterate_per_row: bool = False
    keep_user_id: bool = False
    num_placeholder_tokens_map: Optional[dict] = field(default_factory=dict)
    field_type_map: Optional[dict] = field(default_factory=dict)
    min_sequence_length: int = 10
    feature_map: Optional[dict] = None
    features_to_consider: list[str] = field(default_factory=list)
    file_format: str = None


@dataclass
class SequenceDataloaderConfig(BaseDataloaderConfig):
    """The generic dataloader configuration class for datasets of sequence data.

    Each instance of this class is run on one device.

    Parameters:
    ----------
    dataset_class: IterableDataset
        The dataset class.
    data_folder: str
        Path to the folder containingthe dataset files.
    dataset_config: SequenceDatasetConfig
        The dataset configuration.
    labels: Dict[str, callable]
        A dictionary mapping from feature names to
    batch_size_per_device: list[callable]
        The batch size per dataloader, also per device (GPU).
    num_workers: int
        The number of workers per dataloader, also per device (GPU).
    assign_files_by_size: Optional[dict]
        Whether to assign files to workers by file size to balance computation
        across workers.
    oov_token: Optional[int]
        The token used to represent OOV items.
    masking_token: int
        The token used to represent masked items.
    collate_fn: callable
        Collate function used to construct batches.
    sequence_length: int = 200
        The length of sequences the dataloader should return. If raw sequences
        are shorter, the dataloader will pad them to reach sequence_length.
    padding_token: int = 0
        The token used for padding sequences.
    drop_last: bool = True
        Whether to drop the last batch if it is smaller than
        batch_size_per_device.
    pin_memory: bool = True
        Whether to allocate memory on CPU to ensure data is always available for
        fast transfer to GPU.
    should_shuffle_rows: bool = False
        Whether to shuffle rows between epochs.
    persistent_workers: bool = False
        Whether to maintain worker processes across epochs.
    assign_all_files_per_worker: bool = False
        Whether to assign all files to each worker.
        (NOTE: this should only be activated for training, not for evaluation,
        as it will cause the workers to have overlapping files.)
    """

    dataset_class: IterableDataset
    data_folder: str
    dataset_config: SequenceDatasetConfig
    batch_size_per_device: int
    num_workers: int
    assign_files_by_size: bool
    masking_token: int
    collate_fn: callable  # type: ignore
    labels: Dict[str, callable] = field(default_factory=dict)  # type: ignore
    oov_token: Optional[int] = -1
    sequence_length: int = 200
    padding_token: int = 0
    drop_last: bool = True
    pin_memory: bool = True
    should_shuffle_rows: bool = False
    persistent_workers: bool = False
    timeout: int = 0
    assign_all_files_per_worker: bool = False


@dataclass
class LabelFunctionOutput:
    """Class to unify the output of label functions, making it easier to merge those
    into SequentialModelInputData and SequentialModuleLabelData

    Parameters:
    -----------
    sequence: torch.Tensor
        The sequence tensor of shape (batch_size, sequence_length).
    labels: torch.Tensor
        The labels tensor of shape (num_labels,).
    label_location: torch.Tensor
        The label location tensor of shape (num_labels, 2).
        This is used to indicate the position of the labels in `sequence`.

    """

    sequence: torch.Tensor
    labels: torch.Tensor
    label_location: torch.Tensor = None
    attention_mask: torch.Tensor = None


@dataclass
class SequentialModuleLabelData:
    """The label data class used to wrap the label data for training/testing.

    Parameters
    ----------
    labels: Dict[str, torch.Tensor]
        Dictionary of label_name to label tensor.
        Label tensor is the shape of mask size # long tensor
    label_location: Dict[str, torch.Tensor]
        Dictionary of label_name to label location tensor.
        Label location tensor is the shape of mask_size, 2 as it contains coordinates # long tensor
    """

    labels: Dict[str, torch.Tensor] = field(default_factory=dict)
    label_location: Dict[str, torch.Tensor] = field(default_factory=dict)
    attention_mask: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class SequentialModelInputData:
    """The model input data class used to wrap the model input data for training/testing.

    Parameters
    ----------
    user_id_list: Union[torch.Tensor, List[str], None]
        Tensor or list of user_ids.
    transformed_sequences: Dict[str, torch.Tensor]
        Dictionary of sequence_name to sequence tensor.
        Sequence tensor is (batch_size_per_device x sequence length)
    mask: torch.Tensor
        The mask for the sequence data.
        (batch_size_per_device x sequence length)
    """

    user_id_list: Union[torch.Tensor, List[str], None] = None
    transformed_sequences: Dict[str, torch.Tensor] = field(default_factory=dict)
    mask: torch.Tensor = (
        None  # Single mask if needed as all sequences are padded the same way.
    )


@dataclass
class SemanticIDDatasetConfig(SequenceDatasetConfig):
    """The dataset configuration class used to store the dataset configuration for pipelines
    that use semantic ids.

    Note that this class inherits from SequenceDatasetConfig, thus inherits all of
    its parameters.

    Parameters:
    -----------
    semantic_id_map: Optional[Dict[str, torch.Tensor]]
        The semantic id map from field name to a 2-D tensor.
    keep_user_id: bool
        Whether to keep the user id in the dataset. If set to True, the user id
        will be included in the dataset and can be used for inference or evaluation.
    """

    semantic_id_map: Optional[Dict[str, torch.Tensor]] = None
    keep_user_id: bool = False
    embedding_map: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class TokenizerConfig:
    """The configuration class used to store the tokenizer configuration.

    Parameters:
    ----------
    tokenizer: transformers.PreTrainedTokenizer
        The tokenizer.
    max_length: int
        The maximum length of the tokenized sequences.
    padding: str
        The padding strategy.
    truncation: bool
        Whether to truncate the sequences.
    special_tokens: Optional[Dict[str, str]]
        The special tokens.
    add_special_tokens: bool
        Whether to add special tokens.
    postprocess_eos_token: Optional[bool]
        Whether to postprocess the eos token.
    """

    tokenizer: transformers.PreTrainedTokenizer
    max_length: int
    padding: str
    truncation: bool
    special_tokens: Optional[Dict[str, str]] = field(default_factory=dict)
    add_special_tokens: bool = True
    postprocess_eos_token: Optional[bool] = False


@dataclass
class ItemDatasetConfig(BaseDatasetConfig):
    """The configuration class used to store the item dataset configuration.

    Parameters
    ----------
    item_id_field: str
        The item id field.
    preprocessing_functions: list[callable]
        The preprocessing functions to be applied to the data.
    data_iterator: RawDataIterator
        The data iterator.
    iterate_per_row: bool
        Whether to iterate per row or per batch.
    keep_item_id: bool
        Whether to keep the item id in the data.
    features_to_consider: Optional[List[str]]
        The features to consider.
    feature_map: Optional[Dict[str, str]]
        The map from raw feature name to the key to be used to store the feature in
        ItemTextData.transformed_features.
    field_type_map: Optional[Dict]
        The map from field name to the type of the field.
    embedding_map: Optional[Dict[str, torch.Tensor]]
        The map from field name to the embedding tensor, where the embedding tensor
        is N x d where N is the number of unique IDs in the field and d is the embedding
        dimension.
    num_placeholder_tokens_map: Optional[Dict[str, int]]
        The map of the sparse ID field to the number of placeholder IDs in the field.
    """

    item_id_field: str
    preprocessing_functions: list[callable]  # type: ignore
    data_iterator: RawDataIterator
    iterate_per_row: bool = True
    keep_item_id: bool = True
    features_to_consider: Optional[List[str]] = None
    feature_map: Optional[Dict[str, str]] = None
    field_type_map: Optional[Dict] = None
    embedding_map: Optional[Dict[str, torch.Tensor]] = None
    num_placeholder_tokens_map: Optional[Dict[str, int]] = None


@dataclass
class ItemDataloaderConfig(BaseDataloaderConfig):

    dataset_class: IterableDataset
    data_folder: str
    dataset_config: ItemDatasetConfig
    batch_size_per_device: int
    num_workers: int
    assign_files_by_size: bool

    collate_fn: callable  # type: ignore
    drop_last: bool = True
    pin_memory: bool = True
    should_shuffle_rows: bool = False
    persistent_workers: bool = False
    timeout: int = 0
    oov_token: Optional[int] = 0
    limit_files: Optional[int] = None
    assign_all_files_per_worker: bool = False


@dataclass
class ItemData:
    """The data class used to wrap a batch of item features.

    Parameters
    ----------
    item_ids: Union[torch.Tensor, List[str], None]
        The item ids.
    transformed_features: Dict[str, torch.Tensor]
        The transformed features.
    """

    item_ids: Union[torch.Tensor, List[str], None] = None
    transformed_features: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ItemTextData(ItemData):
    """The data class used to wrap a batch of items with text features for training/testing.

    It is a child class of ItemData, with additional text tokens and text masks.

    Parameters
    ----------
    text_tokens: Optional[torch.Tensor]
        The text tokens.
    text_masks: Optional[torch.Tensor]
        The text masks.
    """

    text_tokens: Optional[torch.Tensor] = None
    text_masks: Optional[torch.Tensor] = None
