from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from src.utils.file_utils import load_json
from src.data.loading.components.interfaces import (
    BaseDatasetConfig,
    SemanticIDDatasetConfig,
)

from src.data.loading.components.interfaces import TokenizerConfig
from src.utils.utils import load_tokenize

# support functions

def convert_bytes_to_string(
    batch_or_row: Dict[str, np.ndarray],
    dataset_config: BaseDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, np.ndarray]:
    # For each feature to apply, cast its np.ndarray of bytes to string.
    for k in batch_or_row:
        if is_feature_in_features_to_apply(features_to_apply, k):
            batch_or_row[k] = batch_or_row[k].astype(str)
    return batch_or_row

def is_feature_in_features_to_apply(features_to_apply: List[str], k: str) -> bool:
    if len(features_to_apply) > 0 and k not in features_to_apply:
        return False
    return True


def filter_features_to_consider(
    batch_or_row: Dict[str, tf.Tensor],
    dataset_config: BaseDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, tf.Tensor]:
    batch_or_row = map_feature_names(batch_or_row, dataset_config)
    features_to_consider = set(dataset_config.features_to_consider)
    if hasattr(dataset_config, "keep_user_id") and dataset_config.keep_user_id:
        if dataset_config.user_id_field not in features_to_consider:
            features_to_consider.add(dataset_config.user_id_field)
    if hasattr(dataset_config, "keep_item_id") and dataset_config.keep_item_id:
        if dataset_config.item_id_field not in features_to_consider:
            features_to_consider.add(dataset_config.item_id_field)
    if len(dataset_config.features_to_consider):
        # Given a batch or row, filter the features to consider.
        return {k: v for k, v in batch_or_row.items() if k in features_to_consider}
    # if not specified, we consider all features
    return batch_or_row


def convert_to_dense_numpy_array(
    batch_or_row: Dict[str, tf.Tensor],
    dataset_config: BaseDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, np.ndarray]:
    # Transform a tfrecord example to a dictionary of numpy arrays, converting sparse tensors to dense numpy arrays.

    for k in batch_or_row:
        if is_feature_in_features_to_apply(features_to_apply, k):
            batch_or_row[k] = tf.sparse.to_dense(batch_or_row[k]).numpy()
    return batch_or_row


def map_feature_names(
    batch_or_row: Dict[str, np.ndarray],
    dataset_config: BaseDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, np.ndarray]:
    # Given a batch or row, map the feature names to the desired feature names.
    if dataset_config.feature_map:
        batch_or_row = {
            v: batch_or_row[k]
            for k, v in dataset_config.feature_map.items()
            if k in batch_or_row
        }
    return batch_or_row


def convert_fields_to_tensors(
    batch_or_row: Dict[str, np.ndarray],
    dataset_config: BaseDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, torch.Tensor]:
    # Given a batch or row, convert all fields to torch tensors. Uses the field type map to determine the dtype, defaulting to torch.long
    # if no dtype is specified.
    for k, v in batch_or_row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            if isinstance(v, int) or isinstance(v, float):
                v = [int(v)]
            batch_or_row[k] = torch.tensor(v, dtype=dataset_config.field_type_map.get(k, torch.long))  # type: ignore
    return batch_or_row


def filter_sequence_length_row(row: Dict[str, torch.Tensor], dataset_config: BaseDatasetConfig, features_to_apply: Optional[List[str]] = [], **kwargs) -> Dict[str, np.ndarray]:  # type: ignore
    # Only works for a row right now. This filters out rows that have fields with sequence length smaller than the min threshold.
    # TODO(lneves): Make this work for a batch as well without creating batches of different sizes.
    for _, tensor in row.items():
        if len(tensor) < dataset_config.min_sequence_length:
            return None
    return row


def filter_empty_feature(row: Dict[str, torch.Tensor], dataset_config: BaseDatasetConfig, features_to_apply: Optional[List[str]] = [], **kwargs) -> Dict[str, np.ndarray]:  # type: ignore
    # Only works for a row right now. This filters out rows that have fields with empty tensors.
    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            if len(v) == 0:
                return None
    return row


def map_sparse_id_to_semantic_id(
    row: Dict[str, torch.Tensor],
    dataset_config: SemanticIDDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    num_hierarchies: Optional[int] = None,
    prefix: str = "",
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Given a row of data, maps the sparse ids to semantic ids
    based on the id_map in the dataset config.
    new column name is prefix + k
    """

    keys_to_process = [k for k in row.keys() if is_feature_in_features_to_apply(features_to_apply, k)]

    for k in  keys_to_process:
        v = row[k]
        if is_feature_in_features_to_apply(features_to_apply, k):
            id_map: torch.Tensor = dataset_config.semantic_id_map.get(k, None)
            # id_map is a D x N tensor
            # where N is the number of unique items in the dataset
            # and D is the number of hierarchies (semantic id digits)
            if id_map is not None:
                # flatten the semantic id sequence
                if num_hierarchies is None:
                    row[prefix + k] = id_map.t()[v].view(-1)
                    print(row[prefix + k].shape)
                else:
                    assert num_hierarchies <= id_map.size(
                        0
                    ), "num_hierarchies must be less than or equal to the number of hierarchies in the semantic id map."
                    row[prefix + k] = id_map[:num_hierarchies].t()[v].view(-1)
            else:
                raise ValueError(f"Semantic id map not found for feature {k}")
    return row


def trim_sequence_row(
    row: Dict[str, Any],
    dataset_config: BaseDatasetConfig,
    sequence_length: int,
    should_trim_left: bool,
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, Any]:
    """
    Trim the sequences in the row to the sequence_length.

    This function handles only rows (not batches) and assumes that the sequences are not
    padded in the first dimension (the dimension to truncate).

    Args:
        row (Dict[str, Any]): A dictionary representing a row of data where each key
            is a feature name and, if the feature is being trimmed, the corresponding
            value is a sequential object to be truncated. The value will be trimmed on
            the side determined by should_trim_left to the specified sequence_length in
            the first dimension.
        dataset_config (BaseDatasetConfig): The dataset configuration object.
        sequence_length (int): The desired length to trim the sequences to.
        should_trim_left (bool): If True, trim the left side of the sequence.
            If False, trim the right side of the sequence.
        features_to_apply (Optional[List[str]]): A list of feature names to apply the
            trimming to. If empty, all features in the row will be trimmed.
    Returns:
        Dict[str, Any]:
            A dictionary identical to the input row, but with the sequences trimmed to
            the specified sequence_length on the specified side. Sequences that are
            shorter than sequence_length will remain unchanged.
    """
    if should_trim_left:
        for k, v in row.items():
            if is_feature_in_features_to_apply(features_to_apply, k):
                v = v[-sequence_length:]
                row[k] = v
    else:
        for k, v in row.items():
            if is_feature_in_features_to_apply(features_to_apply, k):
                v = v[:sequence_length]
                row[k] = v
    return row

def tokenize_text_features(
    batch_or_row: Dict[str, Any],
    features_to_apply: Optional[List[str]] = [],
    tokenizer_config: Optional[TokenizerConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    # Tokenize text features. features_to_apply must contain only text features.
    # This works for both rows and batches.
    tokenize = load_tokenize(config=tokenizer_config)
    batch_or_row_masks = {}
    for k, v in batch_or_row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            k_mask = "_".join([k, "mask"])
            if isinstance(v, np.ndarray) or isinstance(v, list):
                tokenized_seq_list = [tokenize(s) for s in v]
                batch_or_row[k] = torch.stack(
                    [seq["input_ids"].flatten() for seq in tokenized_seq_list]
                )  # seq_length x token_seq_length x 1
                batch_or_row_masks[k_mask] = torch.stack(
                    [seq["attention_mask"].flatten() for seq in tokenized_seq_list]
                )  # seq_length x token_seq_length x 1
            else:
                tokenized_seq = tokenize(v)
                batch_or_row[k] = tokenized_seq[
                    "input_ids"
                ].flatten()  # token_seq_length
                batch_or_row_masks[k_mask] = tokenized_seq[
                    "attention_mask"
                ].flatten()  # token_seq_length

    batch_or_row.update(batch_or_row_masks)
    return batch_or_row


def preprocess_categorical_feature_to_idx(
    batch_or_row: Dict[str, Any],
    features_to_apply: Optional[List[str]] = [],
    mapping_file: Optional[str] = "",
    **kwargs,
) -> Dict[str, Any]:
    # Translate categorical features to indices by looking at the mapping provided.
    # features_to_apply must contain name of the categorical features whose mapping is available in the mapping_file.
    # This works for both rows and batches.

    # Load the mapping if a mapping file is provided
    if mapping_file:
        category_to_idx = load_json(mapping_file)
    else:
        raise ValueError("A valid path to the mapping file must be provided.")

    # Helper function to translate feature values to index
    def translate_to_index(value: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(value, list):
            return [
                category_to_idx.get(v, 0) for v in value
            ]  # Translate each element in the list
        else:
            return category_to_idx.get(
                value, 0
            )  # Default to 0 (e.g., '<OOV>') if not found

    # Determine if we are handling a single row or a batch of rows
    is_batch = isinstance(batch_or_row, list)
    # Apply the mapping to the appropriate features
    if is_batch:
        for row in batch_or_row:
            for feature in features_to_apply:
                if feature in row:
                    row[feature] = translate_to_index(row[feature])
    else:
        for feature in features_to_apply:
            if feature in batch_or_row:
                # if it's a sequence feature then process the entire sequence
                batch_or_row[feature] = translate_to_index(batch_or_row[feature])
    return batch_or_row



def map_sparse_id_to_embedding(
    row: Dict[str, Any],
    dataset_config = None,
    features_to_apply: Optional[List[str]] = [],
    sparse_id_field: str = "id",
    embedding_field_to_add: str = "embedding",
    pre_computer: bool = True,
    total_id_num: int = None,
    id_embedding_dim: int = None,
    **kwargs,
) -> Dict[str, Any]:
    # Map sparse id to pre-computed embedding

    embedding_map: torch.Tensor = dataset_config.embedding_map.get(
        sparse_id_field, None
    )
    # embedding_map is an N x d tensor
    # where N is the number of unique items in the dataset
    # and d is the dimension of the embedding
    if embedding_map is not None:
        row[embedding_field_to_add] = embedding_map[row[sparse_id_field]].squeeze()
    else:
        raise ValueError(f"Embedding map not found")
    return row




def squeeze_tensor_in_place(
    batch_or_row: Dict[str, Any],
    features_to_apply: Optional[List[str]] = [],
    **kwargs,
) -> Dict[str, Any]:
    # Squeeze the dimensions of the features to apply
    # This squeeze is done in place, it does not create a new tensor
    for k, v in batch_or_row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            if isinstance(v, torch.Tensor):
                batch_or_row[k] = v.squeeze_()
            elif isinstance(v, list):
                batch_or_row[k] = [
                    item.squeeze_() if isinstance(item, torch.Tensor) else item
                    for item in v
                ]
            else:
                raise ValueError(
                    f"Unsupported type for feature {k}: {type(v)}. Expected torch.Tensor or list."
                )
    return batch_or_row