from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.loading.components.interfaces import (
    LabelFunctionOutput,
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.data.loading.utils import combine_list_of_tensor_dicts, pad_or_trim_sequence
from src.utils.tensor_utils import extract_locations
from src.data.loading.components.interfaces import ItemData

def identity_collate_fn(batch: Any) -> Any:
    """The default collate function that does nothing."""
    return batch


# TODO (dwd) collate_with_emb_causal_duplicate
def collate_with_emb_causal_duplicate_v2(
    # batch can be a list or a dict
    # this function is used to create the generate contiguous sequences as data augmentation to improve the performance
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    sequence_field_name: str,
    embedding_dim: int,
    labels: Dict[str, callable],  # type: ignore
    sequence_length: int = 200,
    masking_token: int = 1,
    padding_token: int = 0,
    oov_token: Optional[
        int
    ] = None,  # If oov_token is passed, we remove it from the sequence
    max_batch_size: int = 128,
    seq_map: Optional[Dict[str, int]] = None,
    use_fixed_start_augmentation: Optional[bool] = None,  # 新增: 是否使用固定起点的递增子序列
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """
        this collate fn is used to create the generate contiguous sequences as data augmentation to improve the performance.
        It does three things
        1. augment the input sequences by creating all possible contiguous sequences
        2. random sample max_batch_size sequences from the augmented sequences to prevent OOM
        3. run regular collate_fn_train
    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we were
        loading the data per row, or a dictionary of tensors, in the case we were loading the data per batch.
    sequence_field_name : str
        The name of the field in the batch that contains the sequence to be augmented.
    embedding_dim : int
        The dimension of the embedding for each item.
    labels : List[Dict[str, callable]]
        The list of functions to apply to generate the labels.
    sequence_length : int
        The length of the sequence to be padded or trimmed to. (not used in this function, passed to collate_fn_train)
    masking_token : int
        The token used for masking. (not used in this function, passed to collate_fn_train)
    padding_token : int
        The token used for padding. (not used in this function, passed to collate_fn_train)
    oov_token : Optional[int]
        If oov_token is passed, we remove it from the sequence. (not used in this function, passed to collate_fn_train)
    max_batch_size : int
        The maximum batch size to be used after the data augmentation.
    use_fixed_start_augmentation : Optional[bool]
        If True, generate fixed-start incremental subsequences [0,1], [0,1,2], ..., [0,1,...,k-1]
        If False or None, generate all possible contiguous subsequences (original behavior)
    """

    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore
    
    # 根据开关选择不同的采样逻辑
    if use_fixed_start_augmentation:
        # 新逻辑: 固定起点的递增子序列
        return _collate_with_fixed_start_augmentation(
            batch=batch,
            sequence_field_name=sequence_field_name,
            embedding_dim=embedding_dim,
            labels=labels,
            sequence_length=sequence_length,
            masking_token=masking_token,
            padding_token=padding_token,
            oov_token=oov_token,
            max_batch_size=max_batch_size,
            seq_map=seq_map,
        )
    else:
        # 原逻辑: 所有可能的连续子序列
        return _collate_with_all_contiguous_augmentation(
            batch=batch,
            sequence_field_name=sequence_field_name,
            embedding_dim=embedding_dim,
            labels=labels,
            sequence_length=sequence_length,
            masking_token=masking_token,
            padding_token=padding_token,
            oov_token=oov_token,
            max_batch_size=max_batch_size,
            seq_map=seq_map,
        )


def _collate_with_fixed_start_augmentation(
    batch: Dict[str, torch.Tensor],
    sequence_field_name: str,
    embedding_dim: int,
    labels: Dict[str, callable],
    sequence_length: int,
    masking_token: int,
    padding_token: int,
    oov_token: Optional[int],
    max_batch_size: int,
    seq_map: Optional[Dict[str, int]],
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """
    新采样逻辑: 固定起点的递增子序列
    
    对于长度为 k 的序列, 生成:
    [0,1], [0,1,2], [0,1,2,3], ..., [0,1,2,...,k-1]
    总共 k-1 个子序列 (最少2个item, 最多k个item)
    
    Example:
        k=5 → [0,1], [0,1,2], [0,1,2,3], [0,1,2,3,4]  (4个子序列)
    """
    # 计算总子序列数 (每个用户生成 k-1 个)
    total_num_seqs = torch.sum(
        torch.tensor([s.shape[0] for s in batch[sequence_field_name]]) // embedding_dim - 1
    )
    
    # if total_num_seqs > max_batch_size:
    #     select_seqs = torch.randint(
    #         low=0,
    #         high=total_num_seqs,
    #         size=(max_batch_size,),
    #     )
    # else:
    select_seqs = torch.arange(total_num_seqs)
    
    new_batch = {field_name: [] for field_name in batch}
    current_idx = 0
    
    for row_index, sequence in enumerate(batch[sequence_field_name]):
        seq_field_emb_dim = seq_map[sequence_field_name]
        num_items = sequence.shape[0] // seq_field_emb_dim
        
        # 生成固定起点的递增子序列: [0,1], [0,1,2], ..., [0,1,...,k-1]
        for end_item_idx in range(2, num_items + 1):  # 从2个item到k个item
            if current_idx in select_seqs:
                end_index = end_item_idx * seq_field_emb_dim
                
                # 对所有字段进行处理
                for field, emb_dim in seq_map.items():
                    new_sequence = batch[field][row_index]
                    
                    # 固定起点为0, 根据 emb_dim 的比例计算终点
                    ratio = emb_dim / seq_field_emb_dim
                    field_end = int(end_index * ratio)
                    
                    new_batch[field].append(new_sequence[0:field_end])
                
                # 处理非 seq_map 中的字段
                for field_name in new_batch:
                    if field_name not in seq_map.keys():
                        new_batch[field_name].append(batch[field_name][row_index])
            
            current_idx += 1
    
    # 调用标准 collate 函数
    model_input_data, model_label_data = collate_fn_train_v2(
        batch=new_batch,
        labels=labels,
        sequence_length=sequence_length,
        masking_token=masking_token,
        padding_token=padding_token,
        oov_token=oov_token,
        seq_map=seq_map
    )
    
    return model_input_data, model_label_data


def _collate_with_all_contiguous_augmentation(
    batch: Dict[str, torch.Tensor],
    sequence_field_name: str,
    embedding_dim: int,
    labels: Dict[str, callable],
    sequence_length: int,
    masking_token: int,
    padding_token: int,
    oov_token: Optional[int],
    max_batch_size: int,
    seq_map: Optional[Dict[str, int]],
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """
    原采样逻辑: 所有可能的连续子序列
    
    对于长度为 k 的序列, 生成所有长度≥2的连续子序列
    总共 (k-1)*k/2 个子序列
    
    Example:
        k=5 → [0,1], [1,2], [2,3], [3,4], [0,1,2], [1,2,3], ..., [0,1,2,3,4]  (10个子序列)
    """

    # For embedding sequences, we need to adjust the sequence length
    # Each item corresponds to embedding_dim elements in the sequence

    # calculating the total number of contiguous sub-sequences in the batch
    total_num_seqs = torch.sum(
        (
            (
                k := torch.tensor([s.shape[0] for s in batch[sequence_field_name]])
                // embedding_dim
            )
            - 1
        )
        * k
        // 2
    )

    if total_num_seqs > max_batch_size:
        select_seqs = torch.randint(
            low=0,
            high=total_num_seqs,
            size=(max_batch_size,),
        )
    else:
        select_seqs = torch.arange(total_num_seqs)

    # print("select_seqs", select_seqs)

    new_batch = {field_name: [] for field_name in batch}
    # print(batch[sequence_field_name])
    current_idx = 0  # 将 current_idx 移到最外层

    for row_index, sequence in enumerate(batch[sequence_field_name]):

        # 先计算 sequence_data 的子序列数量，用于同步其他字段
        seq_field_emb_dim = seq_map[sequence_field_name]
        seq_end_indices = torch.arange(
            2 * seq_field_emb_dim, sequence.shape[0] + 1, seq_field_emb_dim
        )

        for end_index in seq_end_indices:
            seq_start_indices = torch.arange(
                0, end_index - 2 * seq_field_emb_dim + 1, seq_field_emb_dim
            )

            for start_index in seq_start_indices:
                if current_idx in select_seqs:
                    # 对所有字段进行处理
                    for field, emb_dim in seq_map.items():
                        new_sequence = batch[field][row_index]

                        # 根据 emb_dim 的比例计算对应的起止位置
                        ratio = emb_dim / seq_field_emb_dim
                        field_start = int(start_index * ratio)
                        field_end = int(end_index * ratio)

                        new_batch[field].append(
                            new_sequence[field_start:field_end]
                        )

                    # 处理非 seq_map 中的字段
                    for field_name in new_batch:
                        if field_name not in seq_map.keys():
                            new_batch[field_name].append(batch[field_name][row_index])

                current_idx += 1

    # 打印 new_batch
    for field_name in new_batch:
        if len(new_batch[field_name]) == 0:
            print("new_batch:", field_name, new_batch[field_name], batch, select_seqs)
    # Call collate_fn_train and then detach all tensors to prevent serialization issues
    model_input_data, model_label_data = collate_fn_train_v2(
        batch=new_batch,
        labels=labels,
        sequence_length=sequence_length,  # Use original sequence length
        masking_token=masking_token,
        padding_token=padding_token,
        oov_token=oov_token,
        seq_map=seq_map
    )

    return model_input_data, model_label_data

def collate_with_sid_causal_duplicate(
    # batch can be a list or a dict
    # this function is used to create the generate contiguous sequences as data augmentation to improve the performance
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    sequence_field_name: str,
    sid_hierarchy: int,
    labels: Dict[str, callable],  # type: ignore
    sequence_length: int = 200,
    masking_token: int = 1,
    padding_token: int = 0,
    oov_token: Optional[
        int
    ] = None,  # If oov_token is passed, we remove it from the sequence
    max_batch_size: int = 128,
    use_fixed_start_augmentation: Optional[bool] = None,  # 新增: 是否使用固定起点的递增子序列
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """
        this collate fn is used to create the generate contiguous sequences as data augmentation to improve the performance.
        It does three things
        1. augment the input sequences by creating all possible contiguous sequences
        2. random sample max_batch_size sequences from the augmented sequences to prevent OOM
        3. run regular collate_fn_train
    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we were
        loading the data per row, or a dictionary of tensors, in the case we were loading the data per batch.
    sequence_field_name : str
        The name of the field in the batch that contains the sequence to be augmented.
    sid_hierarchy : int
        The length of Semantic IDs
    labels : List[Dict[str, callable]]
        The list of functions to apply to generate the labels.
    sequence_length : int
        The length of the sequence to be padded or trimmed to. (not used in this function, passed to collate_fn_train)
    masking_token : int
        The token used for masking. (not used in this function, passed to collate_fn_train)
    padding_token : int
        The token used for padding. (not used in this function, passed to collate_fn_train)
    oov_token : Optional[int]
        If oov_token is passed, we remove it from the sequence. (not used in this function, passed to collate_fn_train)
    max_batch_size : int
        The maximum batch size to be used after the data augmentation.
    use_fixed_start_augmentation : Optional[bool]
        If True, generate fixed-start incremental subsequences [0,1], [0,1,2], ..., [0,1,...,k-1]
        If False or None, generate all possible contiguous subsequences (original behavior)
    """

    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore
    
    # 根据开关选择不同的采样逻辑
    if use_fixed_start_augmentation:
        # 新逻辑: 固定起点的递增子序列
        return _collate_with_sid_fixed_start_augmentation(
            batch=batch,
            sequence_field_name=sequence_field_name,
            sid_hierarchy=sid_hierarchy,
            labels=labels,
            sequence_length=sequence_length,
            masking_token=masking_token,
            padding_token=padding_token,
            oov_token=oov_token,
            max_batch_size=max_batch_size,
        )
    else:
        # 原逻辑: 所有可能的连续子序列
        return _collate_with_sid_all_contiguous_augmentation(
            batch=batch,
            sequence_field_name=sequence_field_name,
            sid_hierarchy=sid_hierarchy,
            labels=labels,
            sequence_length=sequence_length,
            masking_token=masking_token,
            padding_token=padding_token,
            oov_token=oov_token,
            max_batch_size=max_batch_size,
        )


def _collate_with_sid_fixed_start_augmentation(
    batch: Dict[str, torch.Tensor],
    sequence_field_name: str,
    sid_hierarchy: int,
    labels: Dict[str, callable],
    sequence_length: int,
    masking_token: int,
    padding_token: int,
    oov_token: Optional[int],
    max_batch_size: int,
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """
    新采样逻辑: 固定起点的递增子序列 (SID 版本)
    
    对于长度为 k 的序列, 生成:
    [0,1], [0,1,2], [0,1,2,3], ..., [0,1,2,...,k-1]
    总共 k-1 个子序列 (最少2个item, 最多k个item)
    
    Example:
        k=5 → [0,1], [0,1,2], [0,1,2,3], [0,1,2,3,4]  (4个子序列)
    """
    # 计算总子序列数 (每个用户生成 k-1 个)
    total_num_seqs = torch.sum(
        torch.tensor([s.shape[0] for s in batch[sequence_field_name]]) // sid_hierarchy - 1
    )
    
    # select_seqs = torch.arange(total_num_seqs)
    # if total_num_seqs > max_batch_size:
    #     select_seqs = torch.randint(
    #         low=0,
    #         high=total_num_seqs,
    #         size=(max_batch_size,),
    #     )
    # else:
    select_seqs = torch.arange(total_num_seqs)
    
    new_batch = {field_name: [] for field_name in batch}
    current_idx = 0
    
    for row_index, sequence in enumerate(batch[sequence_field_name]):
        num_items = sequence.shape[0] // sid_hierarchy
        
        # 生成固定起点的递增子序列: [0,1], [0,1,2], ..., [0,1,...,k-1]
        for end_item_idx in range(2, num_items + 1):  # 从2个item到k个item
            if current_idx in select_seqs:
                end_index = end_item_idx * sid_hierarchy
                
                # 固定起点为0, 切片到 end_index
                new_batch[sequence_field_name].append(sequence[0:end_index])
                
                # 处理其他字段
                for field_name in new_batch:
                    if field_name != sequence_field_name:
                        new_batch[field_name].append(batch[field_name][row_index])
            
            current_idx += 1
    
    return collate_fn_train(
        batch=new_batch,
        labels=labels,
        sequence_length=sequence_length,
        masking_token=masking_token,
        padding_token=padding_token,
        oov_token=oov_token,
    )


def _collate_with_sid_all_contiguous_augmentation(
    batch: Dict[str, torch.Tensor],
    sequence_field_name: str,
    sid_hierarchy: int,
    labels: Dict[str, callable],
    sequence_length: int,
    masking_token: int,
    padding_token: int,
    oov_token: Optional[int],
    max_batch_size: int,
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """
    原采样逻辑: 所有可能的连续子序列 (SID 版本)
    
    对于长度为 k 的序列, 生成所有长度≥2的连续子序列
    总共 (k-1)*k/2 个子序列
    
    Example:
        k=5 → [0,1], [1,2], [2,3], [3,4], [0,1,2], [1,2,3], ..., [0,1,2,3,4]  (10个子序列)
    """

    # calculating the total number of contiguous sub-sequences in the batch
    total_num_seqs = torch.sum(
        (
            (
                k := torch.tensor([s.shape[0] for s in batch[sequence_field_name]])
                // sid_hierarchy
            )
            - 1
        )
        * k
        // 2
    )

    if total_num_seqs > max_batch_size:
        select_seqs = torch.randint(
            low=0,
            high=total_num_seqs,
            size=(max_batch_size,),
        )
    else:
        select_seqs = torch.arange(total_num_seqs)

    new_batch = {field_name: [] for field_name in batch}
    current_idx = 0
    for row_index, sequence in enumerate(batch[sequence_field_name]):
        end_indices = torch.arange(
            2 * sid_hierarchy, sequence.shape[0] + 1, sid_hierarchy
        )
        for end_index in end_indices:
            start_indices = torch.arange(
                0, end_index - 2 * sid_hierarchy + 1, sid_hierarchy
            )  # we have a -2 here because we want to have at least two items in the sequence
            for start_index in start_indices:
                if current_idx in select_seqs:
                    new_batch[sequence_field_name].append(
                        sequence[start_index:end_index]
                    )
                    for field_name in new_batch:
                        if field_name != sequence_field_name:
                            new_batch[field_name].append(batch[field_name][row_index])
                current_idx += 1

    return collate_fn_train(
        batch=new_batch,
        labels=labels,
        sequence_length=sequence_length,
        masking_token=masking_token,
        padding_token=padding_token,
        oov_token=oov_token,
    )


def collate_fn_inference_for_sequence(
    # batch can be a list or a dict
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    id_field_name: str,
    sequence_length: int = 200,
    padding_token: int = 0,
    oov_token: Optional[
        int
    ] = None,  # If oov_token is passed, we remove it from the sequence
    **kwargs,
) -> SequentialModelInputData:
    """The collate function passed to inference dataloader for inference with sequential data.
    It handles id_field_name for saving model outputs

    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we were
        loading the data per row, or a dictionary of tensors, in the case we were loading the data per batch.
    sequence_length : int
        The length of the sequence to be padded or trimmed to.
    masking_token : int
        The token used for masking.
    padding_token : int
        The token used for padding.
    oov_token : Optional[int]
        If oov_token is passed, we remove it from the sequence.
    id_field_name : str
        The name of the field that contains the id of the user/item. This is used to
        map the predictions back to the original id.
    """

    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore

    model_input_data = SequentialModelInputData()

    for field_name, field_sequence in batch.items():  # type: ignore
        if field_name in id_field_name:
            # We use the id field as the user_id_list so predictions can be mapped back to the original id.
            model_input_data.user_id_list = field_sequence

        # TODO (lneves): Allow for non-sequential data to be passed as a feature.
        current_sequence = field_sequence  # type: ignore
        if oov_token:
            # removing the oov token # TODO (Clark): in the future we can add special OOV handling
            current_sequence = [
                sequence[sequence != oov_token] for sequence in field_sequence
            ]
        # 1. in-batch padding s.t. all sequences have the same length and in the format of pt tensor
        current_sequence = pad_sequence(
            current_sequence, batch_first=True, padding_value=padding_token
        )

        # 2. padding or trimming the sequence to the desired length for training
        current_sequence = pad_or_trim_sequence(
            padded_sequence=current_sequence,
            sequence_length=sequence_length,
            padding_token=padding_token,
        )
        model_input_data.transformed_sequences[field_name] = current_sequence

        if field_name not in id_field_name and model_input_data.mask is None:
            # if a field is not id, then it means its the real sequence we want calculate attention mask for it
            model_input_data.mask = (current_sequence != padding_token).long()

    return model_input_data  # type: ignore


def collate_fn_train(
    # batch can be a list or a dict
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    labels: Dict[str, callable],  # type: ignore
    sequence_length: int = 200,
    masking_token: int = 1,
    padding_token: int = 0,
    oov_token: Optional[
        int
    ] = None,  # If oov_token is passed, we remove it from the sequence
    data_augmentation_functions: Optional[
        List[Dict[str, callable]]
    ] = None,  # type: ignore
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """The collate function passed to dataloader. It can do training masking and padding for the input sequence.

    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we were
        loading the data per row, or a dictionary of tensors, in the case we were loading the data per batch.
    labels : List[Dict[str, callable]]
        The list of functions to apply to generate the labels.
    sequence_length : int
        The length of the sequence to be padded or trimmed to.
    masking_token : int
        The token used for masking.
    padding_token : int
        The token used for padding.
    oov_token : Optional[int]
        If oov_token is passed, we remove it from the sequence.
    data_augmentation_functions : Optional[List[Dict[str, callable]]]
        The list of functions to apply to augment the data.
    """

    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore

    if data_augmentation_functions:
        for data_augmentation_function in data_augmentation_functions:
            batch = data_augmentation_function(batch)

    model_input_data = SequentialModelInputData()
    model_label_data = SequentialModuleLabelData()

    for field_name, field_sequence in batch.items():  # type: ignore
        # TODO (lneves): Allow for non-sequential data to be passed as a feature.
        current_sequence = field_sequence  # type: ignore
        if oov_token:
            # removing the oov token # TODO (Clark): in the future we can add special OOV handling
            current_sequence = [
                sequence[sequence != oov_token] for sequence in field_sequence
            ]
        # 1. in-batch padding s.t. all sequences have the same length and in the format of pt tensor
        current_sequence = pad_sequence(
            current_sequence, batch_first=True, padding_value=padding_token
        )

        # 2. padding or trimming the sequence to the desired length for training
        current_sequence = pad_or_trim_sequence(
            padded_sequence=current_sequence,
            sequence_length=sequence_length,
            padding_token=padding_token,
        )

        # creating labels if the field is in the labels list
        if field_name in labels:
            label_function = labels[field_name].transform
            label_function_output: LabelFunctionOutput = label_function.transform_label(
                sequence=current_sequence,
                padding_token=padding_token,
                masking_token=masking_token,
            )
            model_label_data.labels[field_name] = label_function_output.labels
            model_label_data.label_location[
                field_name
            ] = label_function_output.label_location
            model_label_data.attention_mask[
                field_name
            ] = label_function_output.attention_mask
            model_input_data.transformed_sequences[
                field_name
            ] = label_function_output.sequence
        else:
            model_input_data.transformed_sequences[field_name] = current_sequence

        # # Currently supports a single masking per sequence
        # # TODO (lneves): Evaluate if this works or if we should have one mask per feature.
        if model_input_data.mask is None:
            model_input_data.mask = (current_sequence != padding_token).long()

    return model_input_data, model_label_data  # type: ignore




def collate_fn_train_v2(
    # batch can be a list or a dict
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    labels: Dict[str, callable],  # type: ignore
    sequence_length: int = 200,
    masking_token: int = 1,
    padding_token: int = 0,
    oov_token: Optional[
        int
    ] = None,  # If oov_token is passed, we remove it from the sequence
    data_augmentation_functions: Optional[
        List[Dict[str, callable]]
    ] = None,  # type: ignore
    seq_map: Optional[Dict[str, int]] = None,
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """The collate function passed to dataloader. It can do training masking and padding for the input sequence.

    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we were
        loading the data per row, or a dictionary of tensors, in the case we were loading the data per batch.
    labels : List[Dict[str, callable]]
        The list of functions to apply to generate the labels.
    sequence_length : int
        The length of the sequence to be padded or trimmed to.
    masking_token : int
        The token used for masking.
    padding_token : int
        The token used for padding.
    oov_token : Optional[int]
        If oov_token is passed, we remove it from the sequence.
    data_augmentation_functions : Optional[List[Dict[str, callable]]]
        The list of functions to apply to augment the data.
    """

    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore

    if data_augmentation_functions:
        for data_augmentation_function in data_augmentation_functions:
            batch = data_augmentation_function(batch)

    model_input_data = SequentialModelInputData()
    model_label_data = SequentialModuleLabelData()

    # 打印batch 内key 和对应shape


    for field_name, field_sequence in batch.items():  # type: ignore
        # print("before padding collate_fn_train begin:", field_name, len(field_sequence))
        # print("before padding collate_fn_train begin:", field_name, field_sequence[0])
        # TODO (lneves): Allow for non-sequential data to be passed as a feature.
        current_sequence = field_sequence  # type: ignore
        if oov_token:
            # removing the oov token # TODO (Clark): in the future we can add special OOV handling
            current_sequence = [
                sequence[sequence != oov_token] for sequence in field_sequence
            ]
        # 1. in-batch padding s.t. all sequences have the same length and in the format of pt tensor

        current_sequence = pad_sequence(
            current_sequence, batch_first=True, padding_value=padding_token
        )
        # print("collate_fn_train begin:", field_name, len(current_sequence))
        # print("collate_fn_train begin:", field_name, current_sequence[0])

        sequence_length_new = sequence_length
        if seq_map is not None and field_name in seq_map:
            # print("seq_map:", seq_map[field_name])
            sequence_length_new = sequence_length * seq_map[field_name]
        # 2. padding or trimming the sequence to the desired length for training
        current_sequence = pad_or_trim_sequence(
            padded_sequence=current_sequence,
            sequence_length=sequence_length_new,
            padding_token=padding_token,
        )
        # print("collate_fn_train:", field_name, current_sequence.shape, current_sequence)

        # creating labels if the field is in the labels list
        if field_name in labels:
            # print("collate_fn_train_v2 source seq: ", field_name, current_sequence.shape, current_sequence[0])
            label_function = labels[field_name].transform
            label_function_output: LabelFunctionOutput = label_function.transform_label(
                sequence=current_sequence,
                padding_token=padding_token,
                masking_token=masking_token,
            )
            model_label_data.labels[field_name] = label_function_output.labels
            # print("collate_fn_train_v2 label: ", field_name, label_function_output.labels.shape, label_function_output.labels)
            model_label_data.label_location[
                field_name
            ] = label_function_output.label_location
            model_label_data.attention_mask[
                field_name
            ] = label_function_output.attention_mask
            model_input_data.transformed_sequences[
                field_name
            ] = label_function_output.sequence

            # print("collate_fn_train_v2 model_input_data: ", field_name, label_function_output.sequence.shape, label_function_output.sequence[0])
            # print("collate_fn_train_v2 label: ", field_name, label_function_output.labels.shape, label_function_output.labels[0])
            # print("collate_fn_train_v2 label_location: ", field_name, label_function_output.label_location.shape, label_function_output.label_location[0])

        else:
            model_input_data.transformed_sequences[field_name] = current_sequence

        # Currently supports a single masking per sequence
        # TODO (lneves): Evaluate if this works or if we should have one mask per feature.
        if field_name == "sequence_data" and  model_input_data.mask is None:
            model_input_data.mask = (current_sequence != padding_token).long()        #  [B, S * M]
            # model_input_data.mask = zero_last_one(model_input_data.mask)
            # model_input_data.mask 的shape 是 [B, S] 将每一行的最后一个1 置为0

            # print("model_input_data.mask shape:", model_input_data.mask.shape, model_input_data.mask)
            # Adjust the mask and transformed sequences for embedding dimensions
            # The mask should have length sequence_length
            # The transformed sequences should have length sequence_length * embedding_dim



    # input_transfor_dict = model_input_data.transformed_sequences
    # input_transfor_dict = process_map_tensors_2d(model_input_data.mask, input_transfor_dict, seq_map, fill_value=padding_token)
    # model_input_data.transformed_sequences = input_transfor_dict
    # Adjust transformed sequences for embedding dimensions
    for field_name, sequence in model_input_data.transformed_sequences.items():
        # print(field_name, sequence.shape)
        batch_size, _ = model_input_data.mask.shape

        if field_name in seq_map:
            batch_size, _ = sequence.shape
            # Reshape to account for embedding dimensions
            model_input_data.transformed_sequences[field_name] = sequence.view(batch_size, sequence_length, -1)
            # print("Reshape:", field_name,  model_input_data.transformed_sequences[field_name].shape, model_input_data.transformed_sequences[field_name])
        # print("model_input_data.mask shape:", model_input_data.mask.shape)
    # for k, v in model_label_data.labels.items():
    #     print("collate_fn_train_v2 label: ", k, v.shape, v)
    return model_input_data, model_label_data  # type: ignore


def zero_last_one(x):
    """
    将二维 tensor 每一行的最后一个 1 置为 0
    假设每行是 [1,1,...,1,0,0,...,0] 格式（1 在前，0 在后）
    """
    # 方法 1：利用 cumsum 找到最后一个 1 的位置
    # 计算每行累积和，最后一个 1 的位置 = cumsum == row_sum
    row_sum = x.sum(dim=1, keepdim=True)  # 每行 1 的个数
    cumsum = x.cumsum(dim=1)  # 累积和
    last_one_mask = (cumsum == row_sum) & (x == 1)  # 最后一个 1 的位置

    # 创建副本并置 0
    result = x.clone()
    result[last_one_mask] = 0
    return result


def process_map_tensors_2d(mask, map_dict, factor_map, fill_value=-1):
    """
    处理二维 batch 数据

    Args:
        mask: (B, L)
        map_dict: dict[str, Tensor] where each tensor is (B, L_key)
        factor_map: dict[str, int]
        fill_value: 填充值

    Returns:
        processed_map: dict[str, Tensor] same shape as input
    """
    B, L = mask.shape
    n_valid = mask.sum(dim=1)  # (B,) 每行有效长度

    processed = {}
    for key, tensor in map_dict.items():
        factor = factor_map.get(key, 1)
        B2, L_key = tensor.shape
        assert B2 == B, f"Batch size mismatch for key '{key}'"

        # 计算每行应保留的长度: (B,)
        keep_len = factor * n_valid  # (B,)

        # 创建结果张量
        result = torch.full_like(tensor, fill_value)

        # 创建索引矩阵: (B, L_key)
        indices = torch.arange(L_key, device=tensor.device).unsqueeze(0).expand(B, -1)  # (B, L_key)

        # 保留条件: indices < keep_len.unsqueeze(1)
        keep_mask = indices < keep_len.unsqueeze(1)  # (B, L_key)

        # 应用保留
        result[keep_mask] = tensor[keep_mask]

        processed[key] = result

    return processed


def collate_fn_items(
    # batch can be a list or a dict
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    item_id_field: str,
    feature_to_input_name: Dict[str, str],  # type: ignore
) -> ItemData:
    """The collate function passed to the item dataloader.

    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we
        loaded the data per row, or a dictionary of tensors, in the case loaded the data
        per batch.
    item_id_field : str
        The name of the field in the batch that contains the item IDs.
    feature_to_input_name : Dict[str, str]
        The mapping from raw feature name to input feature name in ItemData.

    Returns:
    --------
    model_input_data : ItemData
        An ItemData object, which stores a batch of item features via a list of item IDs
        in the field `item_ids` and a dictionary mapping feature names to value tensors
        stacked along the batch dimension.
    """
    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore
        # does not change shape of text tokens

    model_input_data = ItemData()

    for field_name, field_value in batch.items():  # type: ignore
        if field_name == item_id_field:
            model_input_data.item_ids = list(field_value)

        else:
            # In this case, field_value is a list of tensors, each representing the
            # features of a single item. We stack these tensors along the batch
            # dimension to create a single tensor for the batch of items.
            field_value = torch.stack(field_value, dim=0)
            model_input_data.transformed_features[
                feature_to_input_name[field_name]
            ] = field_value

    return model_input_data