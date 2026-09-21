from typing import Dict, Optional, List
import torch
from torch import nn
import torch.nn.functional as F

from src.data.loading.components.interfaces import (
    BaseDatasetConfig,
    SemanticIDDatasetConfig,
)

from src.data.loading.components.pre_processing import is_feature_in_features_to_apply


def map_sparse_id_to_sid_by_intsid(
    row: Dict[str, torch.Tensor],
    dataset_config: SemanticIDDatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    num_hierarchies: Optional[int] = None,
    num_embeddings_per_hierarchy: int = None,
    embedding_field_to_add: str = "embedding",
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Given a row of data, maps the sparse ids to semantic id embedding
    based on the id_map in the dataset config.
    """

    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):

            (output_emb,
             output_similarity,
             output_similarity_softmax,
             output_similarity_onehot,
             output_max_indices
             ) = codebook_ran_module(v, num_hierarchies, num_embeddings_per_hierarchy, False, **kwargs)
            if output_emb is not None:
                row[k + "_" + embedding_field_to_add] = output_emb
            else:
                raise ValueError(f"codebook_ran_module error")

    return row



def codebook_ran_module(
    item_emb: torch.Tensor,
    num_hierarchies: int = None,
    num_embeddings_per_hierarchy: int = None,
    emb_projection_flag: bool = False,
    is_training: bool = True
) -> (torch.Tensor, List[torch.Tensor],  List[torch.Tensor],  List[torch.Tensor],  List[torch.Tensor]):
    """
    功能:通过多层代码本(codebook)对item_emb嵌入进行压缩/重构
    参数:
        item_emb (torch.Tensor): 输入的嵌入向量。
        num_hierarchies (int): 代码本的层数。
        num_embeddings_per_hierarchy (int): 每个层级的代码本大小。
        emb_projection_flag (bool): 是否对嵌入向量进行映射。
    返回:
        output_emb (torch.Tensor): 编码后的嵌入向量。
        output_similarity (list): 每个层级的相似度。
        output_similarity_softmax (list): 经softmax处理后的相似度。
        output_similarity_onehot (list): one-hot形式的相似度。
        output_max_indices (list): 最大索引值。
    """
    # 使用第一层MLP
    if emb_projection_flag:
        item_emb = torch.nn.Linear(item_emb.size(-1), num_embeddings_per_hierarchy)(item_emb)

    # 初始化输出变量
    output_emb = torch.zeros_like(item_emb)
    output_similarity = []
    output_similarity_softmax = []
    output_similarity_onehot = []
    output_max_indices = []

    # 初始化前一层的输出和相似度
    former_codebook_output = None
    former_similarity = None

    # 初始化 codebooks
    codebooks = [
        torch.randn(num_hierarchies, num_embeddings_per_hierarchy)
        for _ in range(num_hierarchies)
    ]

    for i, codebook in enumerate(codebooks):
        if i == 0:
            codebook_i_input = item_emb
        else:
            current_input = torch.cat([former_codebook_output, former_similarity, item_emb], dim=-1)
            # 使用简单的线性变换替代原有的 neuron_layer_variable
            linear_layer = nn.Linear(current_input.size(-1), item_emb.size(-1))
            codebook_i_input = linear_layer(current_input)

        # 计算相似度
        if len(item_emb.shape) == 2:
            similarity = torch.matmul(codebook_i_input, codebook.t())
        else:
            similarity = torch.einsum('bse, ce -> bsc', codebook_i_input, codebook)

        # Softmax处理相似度
        similarity_softmax = F.softmax(similarity, dim=-1)
        max_indices = torch.argmax(similarity_softmax, dim=-1)
        similarity_onehot = F.one_hot(max_indices, num_classes=similarity_softmax.size(-1)).float()

        # 选择对应的codebook向量
        if len(item_emb.shape) == 2:
            selected_code = torch.matmul(similarity_softmax, codebook)
        else:
            selected_code = torch.einsum('bsc, ce -> bse', similarity_softmax, codebook)

        # 更新前一层的状态
        former_similarity = similarity_softmax
        former_codebook_output = selected_code

        # 累加到输出中
        output_emb += selected_code
        output_similarity.append(similarity)
        output_similarity_softmax.append(similarity_softmax)
        output_similarity_onehot.append(similarity_onehot)
        output_max_indices.append(max_indices)

    return output_emb, output_similarity, output_similarity_softmax, output_similarity_onehot, output_max_indices
