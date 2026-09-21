import logging
from typing import Any, Optional, Tuple, Union

import torch
import transformers
from torch import nn
from torchmetrics.aggregation import BaseAggregator
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers.models.t5.modeling_t5 import T5Config, T5LayerNorm
import torch.nn.functional as F
from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.components.network_blocks.mlp import MLP
from src.models.modules.huggingface.transformer_base_module import TransformerBaseModule
from src.utils.utils import (
    delete_module,
    find_module_shape,
    get_parent_module_and_attr,
    reset_parameters,
)

class SemanticIDGenerativeRecommender(TransformerBaseModule):
    """
    This is a base class for the generative recommender model.
    It is used to generate the semantic ID for the given input.
    It does not contain any specific implementation for the encoder or decoder.
    The encoder and decoder are defined in the subclasses.
    """

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        embedding_dim: int,
        should_check_prefix: bool,
        top_k_for_generation: int,
        should_input_projection:bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDGenerativeRecommender module.

        Paremeters:
        codebooks (torch.Tensor): the codebooks for the semantic ID.
            the shape of the codebooks should be (num_hierarchies, num_embeddings).
        num_hierarchies (int): the number of hierarchies in the codebooks.
        num_embeddings_per_hierarchy (int): the number of embeddings per hierarchy.
        embedding_dim (int): the dimension of the embeddings.
        top_k_for_generation (int): the number of top-k candidates for generation.
        should_check_prefix (bool): whether to check if the prefix is valid.
        """
        super().__init__(**kwargs)

        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.embedding_dim = embedding_dim
        self.num_hierarchies = num_hierarchies
        self.should_check_prefix = should_check_prefix
        if codebooks != None:
            self.codebooks = codebooks.t()
            assert (
                self.codebooks.size(1) == num_hierarchies
            ), "codebooks should be of shape (-1, num_hierarchies)"
        else:
            logging.warning(
                "Not using pre-cached codebooks, \
            please make sure that \n \
                            1) dataset is properly pre-processed \n \
                            2) num_hierarchies and  num_embeddings_per_hierarchy are proerly set\
            "
            )

        self.top_k_for_generation = top_k_for_generation
        if should_input_projection:
            self.input_projection = MLP(
                input_dim=2048,
                hidden_dim_list=[512, 256],
                output_dim=128,
                activation=nn.ReLU,
                dropout=0.0
            )

    def _inject_sep_token_between_sids(
        self,
        id_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sep_token: torch.Tensor,
        num_hierarchies: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject a separator token into the ID embeddings and attention mask.

        Parameters:
        id_embeddings (torch.Tensor): The ID embeddings of shape (batch_size, seq_len, emb_dim).
        attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
        sep_token (torch.Tensor): The separator token of shape (1, emb_dim).
        num_hierarchies (int): The number of hierarchies in the codebooks.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified ID embeddings and attention mask.
        id_embeddings: The ID embeddings with the separator token injected of shape (batch_size, seq_len + num_items, emb_dim).
        attention_mask: The attention mask with the separator token injected of shape (batch_size, seq_len + num_items).

        An intuitive example of the input and output:
        input:
        id_embeddings: [[1, 2, 3, 4], [5, 6, 7, 8]]
        attention_mask: [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
        output:
        id_embeddings: [[1, 2, 3, 4, sep_token], [5, 6, 7, 8, sep_token]]
        attention_mask: [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
        """
        batch_size, seq_len, emb_dim = id_embeddings.size()
        item_count_per_sequence = seq_len // num_hierarchies

        reshaped_id_embeddings = id_embeddings.view(
            batch_size, item_count_per_sequence, num_hierarchies, -1
        )
        reshaped_attention_mask = attention_mask.view(
            batch_size, item_count_per_sequence, num_hierarchies
        )
        reshaped_sep_token_for_concat = (
            sep_token.unsqueeze(0)
            .expand(batch_size, item_count_per_sequence, -1)
            .unsqueeze(-2)
        )
        id_embeddings = torch.cat(
            [reshaped_id_embeddings, reshaped_sep_token_for_concat], dim=-2
        )
        attention_mask = torch.cat(
            [reshaped_attention_mask, reshaped_attention_mask[:, :, [-1]]],
            dim=-1,
        )
        id_embeddings = id_embeddings.reshape(batch_size, -1, emb_dim)
        attention_mask = attention_mask.reshape(batch_size, -1)
        return id_embeddings, attention_mask

    def _spawn_embedding_tables(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> torch.nn.Embedding:
        """
        Spawn an embedding table with the given number of embeddings and embedding dimension.

        Parameters:
        num_embeddings (int): the number of embeddings in the table.
        embedding_dim (int): the dimension of the embeddings.
        """
        table = torch.nn.Embedding(
            num_embeddings=num_embeddings,  # type: ignore
            embedding_dim=embedding_dim,  # type: ignore
        )
        return table

    def _is_kv_cache_valid(
        self, kv_cache: Union[Tuple, DynamicCache, EncoderDecoderCache]
    ) -> bool:

        if isinstance(kv_cache, (EncoderDecoderCache, DynamicCache)):
            return len(kv_cache) > 0
        elif isinstance(kv_cache, Tuple):
            return True
        else:
            return False

    def _add_repeating_offset_to_rows(
        self,
        input_sids: torch.Tensor,
        codebook_size: int,
        num_hierarchies: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Adds repeating offsets to each element in each row of input_sids.
        we use a single embedding table for multiple code books.
        for example if each codebook has 300 embeddings and we have 3 codebooks,
        the input sequence will be transformed from [0, 1, 2] -> to [0, 301, 602]

        Parameters:
            input_sids (torch.Tensor): A 2D PyTorch tensor.
            codebook_size (int): The number of elements in the codebook.
            num_hierarchies (int): The number of hierarchy levels.
        """

        if input_sids.ndim != 2:
            raise ValueError("Input tensor must be 2-dimensional.")

        num_rows, num_cols = input_sids.shape
        offsets = (
            torch.arange(num_hierarchies, device=input_sids.device) * codebook_size
        )

        # Calculate how many times the full offset pattern needs to repeat
        num_repeats = (
            num_cols + num_hierarchies - 1
        ) // num_hierarchies  # Integer division to handle cases where num_cols is not a multiple of num_hierarchies

        # Repeat the offsets and slice to match the number of columns
        repeated_offsets = offsets.repeat(num_repeats)[:num_cols]

        # Add the repeated offsets to each row using broadcasting
        input_sids_with_offsets = input_sids + repeated_offsets
        if attention_mask is not None:
            input_sids_with_offsets = input_sids_with_offsets * attention_mask
        return input_sids_with_offsets

    def _check_valid_prefix(
        self, prefix: torch.Tensor, batch_size: int = 100000
    ) -> torch.Tensor:
        """
        Checks if a given prefix is a valid prefix of the codebooks.

        Args:
            prefix: A tensor of shape [batch_size, hierarchy_level].
            batch_size: The size of the batch to process.

        Returns:
            A boolean tensor of shape [batch_size] indicating the validity of each prefix.
        """
        # TODO (clark): this is a temporary solution, we should use a more efficient way to do this
        # like pre-sorting the codebook and implementing a tree strcture

        current_hierarchy = prefix.shape[1]
        num_prefixes = prefix.shape[0]
        results = []

        # Ensure codebooks are on the correct device.  Do this *once* outside the loop.
        if prefix.device != self.codebooks.device:
            self.codebooks = self.codebooks.to(prefix.device)

        # Trim the codebooks to the relevant hierarchy *once* outside the loop.
        trimmed_codebooks = self.codebooks[:, :current_hierarchy]

        for i in range(0, num_prefixes, batch_size):
            # Get the current batch of prefixes.
            batch_prefix = prefix[
                i : i + batch_size
            ]  # Shape: [batch_size, hierarchy_level]

            # Perform the comparison.  Broadcasting is now limited by batch_size.
            # trimmed_codebooks shape: [C, H] -> unsqueezed [C, 1, H]
            # batch_prefix shape   : [b, H] -> unsqueezed [1, b, H]
            # comparison result    : [C, b, H]
            comparison = trimmed_codebooks.unsqueeze(1) == batch_prefix.unsqueeze(0)

            # Reduce along the hierarchy dimension (H). Shape: [C, b]
            all_match = comparison.all(dim=2)

            # Reduce along the codebook dimension (C).  Shape: [b]
            any_match = all_match.any(dim=0)

            # Append the results for this batch.
            results.append(any_match)

        # Concatenate the results from all batches.
        return torch.cat(results)

    def _beam_search_one_step(
        self,
        candidate_logits: torch.Tensor,
        generated_ids: Union[torch.Tensor, None],
        marginal_log_prob: Union[torch.Tensor, None],
        past_key_values: Union[EncoderDecoderCache, None],
        hierarchy: int,
        batch_size: int,
        sumlogp: Optional[bool] = False,
    ):
        """
        Perform one step of beam search.

        Args:
            candidate_logits: The logits for the next token.
            generated_ids: The generated IDs so far.
            marginal_log_prob: The marginal log probabilities.
            past_key_values: The cache for past key values.
            hierarchy: The current hierarchy level.
            batch_size: The size of the batch.

        Returns:
            The updated generated IDs and the marginal probabilities.
        """

        # pruning the beams that cannot be mapped to a valid item
        if self.should_check_prefix:
            if generated_ids is None:
                valid_prefix_mask = self._check_valid_prefix(
                    torch.arange(
                        self.num_embeddings_per_hierarchy,
                        device=candidate_logits.device,
                    ).unsqueeze(1)
                )
                candidate_logits[:, ~valid_prefix_mask] = float("-inf")
            else:
                # we prune all beams with prefixes that cannot be mapped to a valid item
                valid_prefix_mask = self._check_valid_prefix(
                    torch.cat(
                        [
                            generated_ids.reshape(-1, hierarchy).repeat_interleave(
                                self.num_embeddings_per_hierarchy, dim=0
                            ),
                            torch.arange(
                                self.num_embeddings_per_hierarchy,
                                device=candidate_logits.device,
                            )
                            .repeat(self.top_k_for_generation * batch_size)
                            .unsqueeze(1),
                        ],
                        dim=1,
                    )
                ).reshape(-1, self.num_embeddings_per_hierarchy)
                candidate_logits[~valid_prefix_mask] = float("-inf")

        candidate_logits = torch.nn.functional.softmax(candidate_logits, dim=-1)
        proba, indices = torch.sort(candidate_logits, descending=True)

        if generated_ids is None:
            proba_topk, indices_topk = (
                proba[:, : self.top_k_for_generation],
                indices[:, : self.top_k_for_generation],
            )
            generated_ids = indices_topk.unsqueeze(-1)
            # we need to overwrite the cache because we expanded the beam width from bsz to bsz * beam_width
            # real KV cache starts from the first hierarchy rather than 0-th
            # this is because in 0th hierarchy, self-attention doesn't have cache.
            # and kv cache in huggingface has poor support for this corner case
            past_key_values = EncoderDecoderCache(
                self_attention_cache=DynamicCache(),
                cross_attention_cache=DynamicCache(),
            )
            replace_indices = None
            if sumlogp:
                proba_topk = torch.log(proba_topk)
        else:
            # we have beams, generating more beams from the existing beams
            proba, indices = (
                proba[:, : self.num_embeddings_per_hierarchy],
                indices[:, : self.num_embeddings_per_hierarchy],
            )
            proba, indices = proba.reshape(
                -1, self.top_k_for_generation * self.num_embeddings_per_hierarchy
            ), indices.reshape(
                -1, self.top_k_for_generation * self.num_embeddings_per_hierarchy
            )
            # calculating the marginal probability
            if sumlogp:
                proba = torch.log(proba) + marginal_log_prob.repeat_interleave(self.num_embeddings_per_hierarchy, dim=-1)
            else:
                proba = torch.mul(
                    marginal_log_prob.repeat_interleave(
                        self.num_embeddings_per_hierarchy, dim=-1
                    ),
                    proba,
                )

            topk_results = torch.topk(
                torch.nan_to_num(proba, nan=-1), k=self.top_k_for_generation, dim=-1
            )
            proba_topk, indices_topk = topk_results.values, topk_results.indices
            # getting indices of winning beams in the original beams
            replace_indices = (
                (indices_topk // self.num_embeddings_per_hierarchy)
                + torch.arange(indices_topk.size(0), device=proba.device).unsqueeze(1)
                * self.top_k_for_generation
            ).flatten()
            # accordingly update kv cache given the winning beams
            if past_key_values != None:
                past_key_values.reorder_cache(replace_indices)

            indices_topk = torch.gather(indices, 1, indices_topk)

        if replace_indices != None:
            generated_ids = torch.cat(
                [
                    generated_ids.reshape(-1, hierarchy)[replace_indices].reshape(
                        -1, self.top_k_for_generation, hierarchy
                    ),
                    indices_topk.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            generated_ids = indices_topk.unsqueeze(-1)

        return generated_ids, proba_topk, past_key_values

    def eval_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        loss_to_aggregate: BaseAggregator,
    ):
        """Perform a single evaluation step on a batch of data from the validation or test set.
        The method will update the metrics and the loss that is passed.
        """
        import time
        start = time.time()
        
        # Batch is a tuple of model inputs and labels.
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]
        
        batch_size = model_input.mask.shape[0]
        _, loss, _ = self.model_step(model_input=model_input, label_data=label_data)
        # print("eval_step:", model_input.transformed_sequences.keys())
        # print("eval_step model_input.mask shape:", model_input.mask.shape)

        generated_ids, marginal_probs = self.generate(
            attention_mask=model_input.mask,
            **{
                self.feature_to_model_input_map[k]: v
                for k, v in model_input.transformed_sequences.items()
                if k in self.feature_to_model_input_map
            },
        )

        self.evaluator(
            marginal_probs=marginal_probs,
            generated_ids=generated_ids,
            # TODO: (lneves) hardcoded for now, will need to change for multiple features
            labels=label_data.labels['sid_sequence_data'].to(marginal_probs.device) if 'sid_sequence_data' in label_data.labels else label_data.labels['sequence_data'].to(marginal_probs.device),
        )

        loss_to_aggregate(loss)
        
        end = time.time()
        # 只记录 epoch 级别的平均值，不记录每个 step，与训练的日志频率保持一致
        self.log("val/eval_step_time", end - start,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True)

    def _make_deterministic(self, is_training: bool):
        """
        Make the model deterministic by turning off some flags.
        This is needed as the default functions in lightning such as
        on_validation_start on_predict_start cannnot properly set the flags
        for the encoder and decoder.
        (TODO) clark: in the future we can revisit this and make it more generic

        Args:
            is_training (bool): Whether the model is in training mode or not.
        """
        if is_training:
            if self.decoder != None:
                self.decoder.decoder.is_training = True
                self.decoder.decoder.train()
            if self.encoder != None:
                self.encoder.encoder.is_training = True
                self.encoder.encoder.train()
        else:
            if self.decoder != None:
                self.decoder.decoder.is_training = False
                self.decoder.decoder.eval()
            if self.encoder != None:
                self.encoder.encoder.is_training = False
                self.encoder.encoder.eval()

    def on_predict_start(self):
        super().on_predict_start()
        self._make_deterministic(is_training=False)

    def on_predict_end(self):
        super().on_predict_end()
        self._make_deterministic(is_training=True)

    def on_validation_start(self):
        super().on_validation_start()
        self._make_deterministic(is_training=False)

    def on_validation_end(self):
        super().on_validation_end()
        self._make_deterministic(is_training=True)

    def on_test_start(self):
        super().on_test_start()
        self._make_deterministic(is_training=False)

    def on_test_end(self):
        super().on_test_end()
        self._make_deterministic(is_training=True)

    def on_train_start(self):
        super().on_train_start()
        self._make_deterministic(is_training=True)


class SemanticIDEncoderDecoder(SemanticIDGenerativeRecommender):
    """
    This is an in-house implementation of the encoder-decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    We added some additional features and modifications to the original architecture.
    (e.g., constrained beam search, separation tokens, etc)
    """

    def __init__(
        self,
        top_k_for_generation: int = 10,
        codebooks: torch.Tensor = None,
        embedding_dim: int = None,
        num_hierarchies: int = None,
        num_embeddings_per_hierarchy: int = None,
        num_user_bins: Optional[int] = None,
        mlp_layers: Optional[int] = None,
        should_check_prefix: bool = False,
        should_add_sep_token: bool = True,
        prediction_key_name: str = "user_id",
        prediction_value_name: str = "semantic_ids",
        encoder_input_emb: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDEncoderDecoder module.

        Paremeters:
        codebooks (torch.Tensor): the codebooks for the semantic ID.
            the shape of the codebooks should be (num_hierarchies, num_embeddings_per_hierarchy).
        num_hierarchies (int): the number of hierarchies in the codebooks.
        top_k_for_generation (int): the number of top-k candidates for generation.
        num_user_bins (Optional[int]): the number of bins for user in the dataset (this number equals to the number of rows in the embedding table ).
        mlp_layers (Optional[int]): the number of mlp layers in the encoder and decoder.
        embedding_dim (Optional[int]): the dimension of the embeddings.
        should_check_prefix (bool): whether to check if the prefix is valid.
        """

        if num_hierarchies is None or num_embeddings_per_hierarchy is None:
            num_hierarchies, num_embeddings_per_hierarchy = (
                codebooks.shape[0],
                codebooks.max().item() + 1,
            )
        if embedding_dim is None:
            embedding_dim = (
                kwargs["huggingface_model"]
                .encoder.block[0]
                .layer[0]
                .SelfAttention.q.in_features
            )

        super().__init__(
            codebooks=codebooks,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings_per_hierarchy,
            embedding_dim=embedding_dim,
            top_k_for_generation=top_k_for_generation,
            should_check_prefix=should_check_prefix,
            **kwargs,
        )

        self.encoder = SemanticIDEncoderModule(
            encoder=self.encoder,
        )

        # bos_token used to prompt the decoder to generate the first token
        bos_token = torch.nn.Parameter(
            torch.randn(1, self.embedding_dim), requires_grad=True
        )

        self.decoder = SemanticIDDecoderModule(
            decoder=self.decoder,
            bos_token=bos_token,
            decoder_mlp=torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.embedding_dim,
                        self.num_embeddings_per_hierarchy,
                        bias=False,
                    )
                    for _ in range(self.num_hierarchies)
                ]
            ),
        )

        if mlp_layers is not None:
            # bloating the mlp layers in both encoder and decoder
            # TODO (clark): this currently only works for T5
            for name, module in self.named_modules():
                if isinstance(module, transformers.models.t5.modeling_t5.T5LayerFF):
                    parent_module, attr_name = get_parent_module_and_attr(self, name)
                    setattr(
                        parent_module,
                        attr_name,
                        T5MultiLayerFF(
                            config=self.encoder.encoder.config, num_layers=mlp_layers
                        ),
                    )

        # generate embedding tables for each hierarchy
        # here we assume each hierarchy has the same amount of embeddings
        self.item_sid_embedding_table_encoder = self._spawn_embedding_tables(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies,
            embedding_dim=self.embedding_dim,
        )

        # generating user embedding table
        self.user_embedding: torch.nn.Embedding = (
            self._spawn_embedding_tables(
                num_embeddings=num_user_bins,
                embedding_dim=self.embedding_dim,
            )
            if num_user_bins
            else None
        )

        # separation token for the encoder to differentiate between items
        self.sep_token = (
            torch.nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)
            if should_add_sep_token
            else None
        )
        # the key value names for the prediction output
        self.prediction_key_name = prediction_key_name
        self.prediction_value_name = prediction_value_name
        self.encoder_input_emb = encoder_input_emb

    def encoder_forward_pass(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder module.

        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
        """
        if self.encoder_input_emb:
            # input_ids 是 (batch_size, seq_length* embedding_dim )
            #  reshap 为 (batch_size, seq_length, embedding_dim)
            input_ids = input_ids.reshape(
                input_ids.size(0), 30, 2048
            )
            input_ids = self.input_projection(input_ids)
            # attention_mask 是 (batch_size, seq_length * 2048) 转换成  (batch_size, seq_length) 在2048上取max
            attention_mask = attention_mask.view(attention_mask.size(0), 30, 2048).amax(dim=2)


            encoder_output = self.encoder(
                sequence_embedding=input_ids,
                attention_mask=attention_mask,
            )
            return encoder_output, attention_mask

        # we shift the IDs here to match the hierarchy structure
        # so that we can use a single embedding table to store the embeddigns for all hierarchies
        shifted_sids = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds_for_encoder = self.get_embedding_table(table_name="encoder")(
            shifted_sids
        )

        if self.sep_token is not None:
            (
                inputs_embeds_for_encoder,
                attention_mask,
            ) = self._inject_sep_token_between_sids(
                id_embeddings=inputs_embeds_for_encoder,
                attention_mask=attention_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        # we enter this loop if we want to use user_id
        if user_id is not None and self.user_embedding is not None:
            # preprocessing function pad user_id with zeros
            # so we only need to take the first column
            user_id = user_id[:, 0]

            # TODO (clark): here we assume remainder hashing, which is different from LSH hashing used in TIGER.
            user_embeds = self.user_embedding(
                torch.remainder(user_id, self.user_embedding.num_embeddings)
            )

            # prepending the user_id embedding to the input senquence
            inputs_embeds_for_encoder = torch.cat(
                [
                    user_embeds.unsqueeze(1),
                    inputs_embeds_for_encoder,
                ],
                dim=1,
            )
            # prepending 1 to attention mask as we introduce user embedding in the first column
            user_attention_mask = torch.ones(
                attention_mask.size(0), 1, device=attention_mask.device
            )
            attention_mask_for_encoder = torch.cat(
                [
                    user_attention_mask,
                    attention_mask,
                ],
                dim=1,
            )
        else:
            attention_mask_for_encoder = attention_mask

        encoder_output = self.encoder(
            sequence_embedding=inputs_embeds_for_encoder,
            attention_mask=attention_mask_for_encoder,
        )
        return encoder_output, attention_mask_for_encoder

    def decoder_forward_pass(
        self,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # TODO (clark): in the future we should support variable length semantic id
        future_ids: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        attention_mask_for_encoder: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[DynamicCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            future_ids (Optional[torch.Tensor]): The future IDs for the decoder.
            encoder_output (Optional[torch.Tensor]): The output from the encoder.
            attention_mask_for_encoder (Optional[torch.Tensor]): The attention mask for the encoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (Optional[DynamicCache]): The cache for past key values.
        """

        # we generated something before and we need to shift the future_ids
        if future_ids is not None:
            shifted_future_sids = self._add_repeating_offset_to_rows(
                input_sids=future_ids,
                codebook_size=self.num_embeddings_per_hierarchy,
                num_hierarchies=self.num_hierarchies,
                attention_mask=torch.ones_like(future_ids, device=future_ids.device)
                if attention_mask is None
                else attention_mask,
            )
            inputs_embeds_for_decoder = self.get_embedding_table(table_name="decoder")(
                shifted_future_sids
            )

            # we do not have valid kv cache
            # we need to prepend bos token to the decoder input
            if not self._is_kv_cache_valid(kv_cache=past_key_values):
                inputs_embeds_for_decoder = torch.cat(
                    [
                        self.decoder.bos_token.unsqueeze(0).expand(
                            future_ids.size(0), 1, -1
                        ),
                        inputs_embeds_for_decoder,
                    ],
                    dim=1,
                )
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            torch.ones(future_ids.size(0), 1, device=future_ids.device),
                            attention_mask,
                        ],
                        dim=1,
                    )
            else:
                # we have valid kv cache
                # we only need the last token in the decoder input
                inputs_embeds_for_decoder = inputs_embeds_for_decoder[:, -1:, :]
        # this is the beginning of generation, we start from bos token
        else:
            inputs_embeds_for_decoder = self.decoder.bos_token.unsqueeze(0).expand(
                encoder_output.size(0), 1, -1
            )

        decoder_output = self.decoder(
            sequence_embedding=inputs_embeds_for_decoder,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask_for_encoder,
            encoder_output=encoder_output,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        return decoder_output

    def generate(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate the semantic id given the current model in the sequence using beam search.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
        """

        # getting encoder output
        # we only need to do this once because we have decoder
        # to do auto-regressive generation
        encoder_output, encoder_attention_mask = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
        )
        # print("generate: ", encoder_output.shape)

        # initilize cached generated ids to None
        generated_ids = None
        marginal_log_prob = None

        # initialize kv cache
        past_key_values = EncoderDecoderCache(
            self_attention_cache=DynamicCache(), cross_attention_cache=DynamicCache()
        )

        for hierarchy in range(self.num_hierarchies):
            if generated_ids is not None:
                # we generated something before
                # we need to reshape the generated ids so that
                # the number of beams equals to batch size * top_k
                squeezed_generated_ids = generated_ids.reshape(-1, hierarchy).to(
                    encoder_output.device
                )  # shape: (batch_size * top_k, hierarchy)

                repeated_encoder_output = encoder_output.repeat_interleave(
                    self.top_k_for_generation, dim=0
                )
                # shape: (batch_size * top_k, seq_len+1, hidden_dim)
                # +1 because we have user_id token

                repeated_encoder_attention_mask = (
                    encoder_attention_mask.repeat_interleave(
                        self.top_k_for_generation, dim=0
                    )
                )  # shape: (batch_size * top_k, seq_len+1)
            else:
                # we haven't generated anything yet!
                # the number of beams currently equals to batch size
                squeezed_generated_ids = None
                repeated_encoder_output = encoder_output
                repeated_encoder_attention_mask = encoder_attention_mask

            # feeding the decoder with the generated ids
            decoder_output, past_key_values = self.decoder_forward_pass(
                future_ids=squeezed_generated_ids,
                encoder_output=repeated_encoder_output,
                attention_mask_for_encoder=repeated_encoder_attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

            # decoder_output[:, -1, :] is the embedding for the next token
            latest_output_representation = decoder_output[:, -1, :]

            # # calculating the logits for the next token
            candidate_logits = self.decoder.decoder_mlp[hierarchy](
                latest_output_representation
            )  # shape: (batch_size * top_k, num_embeddings in the hierarchy)

            (
                generated_ids,
                marginal_log_prob,
                past_key_values,
            ) = self._beam_search_one_step(
                candidate_logits=candidate_logits,
                generated_ids=generated_ids,
                marginal_log_prob=marginal_log_prob,
                past_key_values=past_key_values,
                hierarchy=hierarchy,
                batch_size=input_ids.size(0),
            )
        return generated_ids, marginal_log_prob

    def forward(
        self,
        attention_mask_encoder: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
        future_ids: Optional[torch.Tensor] = None,
        attention_mask_decoder: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder-decoder model.
        Parameters:
            attention_mask_encoder (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
            future_ids (Optional[torch.Tensor]): The future IDs for the decoder.
            attention_mask_decoder (Optional[torch.Tensor]): The attention mask for the decoder.
        """

        encoder_output, attention_mask_for_encoder = self.encoder_forward_pass(
            attention_mask=attention_mask_encoder,
            input_ids=input_ids,
            user_id=user_id,
        )

        decoder_output = self.decoder_forward_pass(
            future_ids=future_ids,
            attention_mask=attention_mask_decoder,
            encoder_output=encoder_output,
            attention_mask_for_encoder=attention_mask_for_encoder,
            use_cache=False,  # we are not using cache for training
        )
        # print("forward decoder_output:", decoder_output.shape)
        return decoder_output

    def get_embedding_table(self, table_name: str, hierarchy: Optional[int] = None):
        """
        Get the embedding table for the given table name and hierarchy.
        Args:
            table_name: The name of the table to get the embedding for.
            hierarchy: The hierarchy level to get the embedding for.
        """
        # here we assume the encoder and decoder share the same embedding table
        # we can have flexible embedding table in the future
        if table_name == "encoder":
            embedding_table = self.item_sid_embedding_table_encoder
        elif table_name == "decoder":
            embedding_table = self.item_sid_embedding_table_encoder

        if hierarchy is not None:
            return embedding_table(
                torch.arange(
                    hierarchy * self.num_embeddings_per_hierarchy,
                    (hierarchy + 1) * self.num_embeddings_per_hierarchy,
                ).to(self.device)
            )
        return embedding_table

    def predict_step(self, batch: SequentialModelInputData):
        generated_sids, _ = self.model_step(batch)
        ids = [
            id.item() if isinstance(id, torch.Tensor) else id
            for id in batch.user_id_list
        ]
        model_output = OneKeyPerPredictionOutput(
            keys=ids,
            predictions=generated_sids,
            key_name=self.prediction_key_name,
            prediction_name=self.prediction_value_name,
        )
        return model_output

    def model_step(
        self,
        model_input: SequentialModelInputData,
        label_data: Optional[SequentialModuleLabelData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass of the model and calculate the loss if label_data is provided.

        Args:
            model_input: The input data to the model.
            label_data: The label data to the model. Its optional as it is not required for inference.
        """

        # if label_data is None, we are in inference mode and doing free-form generation
        if label_data is None:
            # this is inference stage
            generated_ids, marginal_probs = self.generate(
                attention_mask=model_input.mask,
                **{
                    self.feature_to_model_input_map.get(k, k): v
                    for k, v in model_input.transformed_sequences.items()
                    if k in self.feature_to_model_input_map
                },
            )
            return generated_ids, 0  # returning 0 here because we don't have a loss

        fut_ids = None
        for label in label_data.labels:
            curr_label = label_data.labels[label]
            fut_ids = curr_label.reshape(model_input.mask.size(0), -1)
        # here we pass labels in to the forward function
        # because the decoder is causal and we are doing shifted prediction



        model_output = self.forward(
            attention_mask_encoder=model_input.mask,
            future_ids=fut_ids,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        # we prepended a bos token to the decoder input
        # so we need to remove the last token in the output
        model_output = model_output[:, :-1]

        # the label locations is shared for all semantic id hierarchies
        loss = 0
        for hierarchy in range(self.num_hierarchies):

            input = self.decoder.decoder_mlp[hierarchy](model_output[:, hierarchy])
            # input进行 softmax 然后返回最大 index
            softmax_output = F.softmax(input, dim=-1)
            max_indices = torch.argmax(softmax_output, dim=-1)

            loss += self.loss_function(
                input=input,
                target=fut_ids[:, hierarchy].long(),
            )
        return model_output, loss, 0


class SemanticIDDecoderModule(torch.nn.Module):
    """
    This is an in-house replication of the decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        decoder: transformers.PreTrainedModel,
        decoder_mlp: Optional[torch.nn.Module] = None,
        bos_token: Optional[torch.nn.Parameter] = None,
    ) -> None:
        """
        Initialize the SemanticIDDecoderModule.

        Parameters:
        decoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        decoder_mlp (torch.nn.Module): the mlp layers used to project the decoder output to the embedding table.
        bos_token (Optional[torch.nn.Parameter]):
            the bos token used to prompt the decoder.
            if None, then this means the decoder is used standalone without an encoder.
        """

        super().__init__()
        # some sanity checks
        if bos_token is not None:
            assert decoder.config.is_decoder == True, "Decoder must be a decoder model"
            assert (
                decoder.config.is_encoder_decoder == False
            ), "Decoder must be a standalone decoder model"

        self.decoder = decoder
        # this bos token is prompt for the decoder
        self.bos_token = bos_token
        self.decoder_mlp = decoder_mlp
        # deleting embedding table in the decoder to save space
        delete_module(self.decoder, "embed_tokens")
        delete_module(self.decoder, "shared")
        reset_parameters(self.decoder)

    def forward(
        self,
        attention_mask: torch.Tensor,
        sequence_embedding: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        use_cache: bool = False,
        past_key_values: DynamicCache = DynamicCache(),
    ) -> torch.Tensor:
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            sequence_embedding (torch.Tensor): The input sequence embedding for the decoder.
            encoder_output (torch.Tensor): The output from the encoder.
            encoder_attention_mask (torch.Tensor): The attention mask for the encoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (DynamicCache): The cache for past key values.
        """

        # print("SemanticIDDecoderModule, sequence_embedding shape: ", sequence_embedding.shape)
        decoder_outputs: Seq2SeqModelOutput = self.decoder(
            attention_mask=attention_mask,
            inputs_embeds=sequence_embedding,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        embeddings = decoder_outputs.last_hidden_state

        if use_cache:
            return embeddings, decoder_outputs.past_key_values
        return embeddings


class SemanticIDEncoderModule(torch.nn.Module):
    """
    This is an in-house replication of the encoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        encoder: transformers.PreTrainedModel,
    ) -> None:
        """
        Initialize the SemanticIDEncoderModule module.

        Paremeters:
        encoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        """
        super().__init__()

        self.encoder = encoder
        embedding_table_dim = find_module_shape(self.encoder, "embed_tokens")
        num_embeddings, embedding_dim = embedding_table_dim

        self.num_embeddings_per_hierarchy = num_embeddings
        self.embedding_dim = embedding_dim
        # TODO (clark): take care of chunky position encoding

        # deleting embedding table in the encoder to save space
        delete_module(self.encoder, "embed_tokens")
        delete_module(self.encoder, "shared")
        reset_parameters(self.encoder)

    def forward(
        self,
        attention_mask: torch.Tensor,
        sequence_embedding: torch.Tensor,
    ) -> torch.Tensor:
        encoder_output = self.encoder(
            inputs_embeds=sequence_embedding,
            attention_mask=attention_mask,
        )
        embeddings = encoder_output.last_hidden_state
        return embeddings


# TODO (clark): this is a T5 specific implementation
# this class is used for bloating the mlp layers in the encoder and decoder
# original T5 implementation only has one layer
class T5MultiLayerFF(nn.Module):
    def __init__(self, config: T5Config, num_layers: int):
        """
        Initialize the T5MultiLayerFF module.
        This module is a multi-layer feed-forward network (MLP) used in the T5 model.
        It consists of a series of linear layers with ReLU activation and dropout.
        And it also includes layer normalization and residual connections.
        Parameters:
            config (T5Config): The T5 configuration object.
            num_layers (int): The number of layers in the MLP.
        """
        super().__init__()
        self.mlp = MLP(
            input_dim=config.d_model,
            output_dim=config.d_model,
            hidden_dim_list=[config.d_ff for _ in range(num_layers)],
            activation=nn.ReLU,
            dropout=config.dropout_rate,
        )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the T5MultiLayerFF module.
        Parameters:
            hidden_states (torch.Tensor): The input hidden states for the MLP.
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states
