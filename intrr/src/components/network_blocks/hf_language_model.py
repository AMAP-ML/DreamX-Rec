from typing import Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from src.models.components.network_blocks.embedding_aggregator import (
    EmbeddingAggregator,
)


class HFLanguageModel(nn.Module):
    def __init__(
        self,
        huggingface_model: PreTrainedModel,
        aggregator: EmbeddingAggregator,
        postprocessor: nn.Module = nn.Identity(),
        return_last_hidden_states: bool = False,
    ):
        """Initialize the HuggingFace language model.

        This is a wrapper around a HuggingFace PreTrainedModel that generates text
        sequence embeddings by passing the last hidden states of the PreTrainedModel
        through an aggregator and postprocessor.

        Args:
            huggingface_model: HuggingFace model to use for language modeling
            aggregator: Aggregator to use to aggregate the embeddings
            postprocessor: Postprocessor to use to process the aggregated embeddings
            return_last_hidden_states: Whether to return the last hidden states
        """
        super(HFLanguageModel, self).__init__()
        self.huggingface_model = huggingface_model
        self.aggregator = aggregator
        self.postprocessor = postprocessor
        self.return_last_hidden_states = return_last_hidden_states

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the HuggingFace language model.

        Args:
            text_ids: Tensor of token ids.
            text_attention_masks: Tensor of attention masks.

        Returns:
            postprocessed_embeddings: Postprocessed embeddings.
            embeddings: Last hidden states if return_last_hidden_states is True.
        """
        # TODO(lcollins2): Generalize this to handle other types of inputs
        outputs: BaseModelOutput = self.huggingface_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        embeddings = outputs.last_hidden_state
        aggregated_embeddings = self.aggregator(embeddings, attention_mask)
        postprocessed_embeddings = self.postprocessor(aggregated_embeddings)
        if self.return_last_hidden_states:
            return postprocessed_embeddings, embeddings
        return postprocessed_embeddings
