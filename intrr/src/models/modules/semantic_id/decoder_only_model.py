from typing import Any, Optional, Tuple, Union

import torch
import transformers
from torch import nn
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers.cache_utils import DynamicCache

from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.modules.semantic_id.tiger_generation_model import SemanticIDGenerativeRecommender, T5MultiLayerFF
from src.utils.utils import (
    delete_module,
    get_parent_module_and_attr,
    reset_parameters,
)


class SemanticIDDecoderOnly(SemanticIDGenerativeRecommender):
    """
    A pure decoder-only architecture for semantic ID generation, similar to GPT-style models.
    This model uses a single transformer decoder stack for both encoding input sequences
    and generating output sequences in an autoregressive manner.
    """

    def __init__(
            self,
            codebooks: torch.Tensor,
            num_hierarchies: Optional[int] = None,
            num_embeddings_per_hierarchy: Optional[int] = None,
            embedding_dim: Optional[int] = None,
            top_k_for_generation: int = 10,
            num_user_bins: Optional[int] = None,
            mlp_layers: Optional[int] = None,
            should_check_prefix: bool = True,
            prediction_key_name: str = "user_id",
            prediction_value_name: str = "generated_sids",
            should_add_sep_token: bool = False,
            **kwargs: Any,
    ) -> None:
        """
        Initialize the SemanticIDDecoderOnly model.

        Parameters:
        -----------
        codebooks (torch.Tensor): the codebooks for the semantic ids.
        num_hierarchies (Optional[int]): the number of hierarchies in the semantic ids.
        num_embeddings_per_hierarchy (Optional[int]): the number of embeddings per hierarchy.
        top_k_for_generation (int): the number of top-k candidates for generation.
        num_user_bins (Optional[int]): the number of bins for user in the dataset (this number equals to the number of rows in the embedding table ).
        mlp_layers (Optional[int]): the number of mlp layers in the decoder.
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

        # In decoder-only architecture, we use the same model for both encoding and decoding
        # Check if decoder is a Hugging Face model or a custom model (like HSTU)
        if hasattr(self.decoder, 'config') and hasattr(self.decoder.config, 'is_decoder'):
            # Standard Hugging Face decoder (e.g., T5)
            self.decoder = DecoderModule(
                decoder=self.decoder,
            )
        else:
            # Custom decoder (e.g., HSTU) or other models without config
            self.decoder = HSTUDecoderModule(
                decoder=self.decoder,
            )

        # bos_token used to prompt the decoder to generate the first token
        bos_token = torch.nn.Parameter(
            torch.randn(1, self.embedding_dim), requires_grad=True
        )
        self.bos_token = bos_token

        if mlp_layers is not None:
            # bloating the mlp layers in the decoder
            # TODO (clark): this currently only works for T5
            for name, module in self.named_modules():
                if isinstance(module, transformers.models.t5.modeling_t5.T5LayerFF):
                    parent_module, attr_name = get_parent_module_and_attr(self, name)
                    setattr(
                        parent_module,
                        attr_name,
                        T5MultiLayerFF(
                            config=self.decoder.decoder.config, num_layers=mlp_layers
                        ),
                    )

        # generate embedding tables for each hierarchy
        # here we assume each hierarchy has the same amount of embeddings
        self.item_sid_embedding_table = self._spawn_embedding_tables(
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

        # separation token for the decoder to differentiate between items
        self.sep_token = (
            torch.nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)
            if should_add_sep_token
            else None
        )

        # MLP layers for projecting decoder output to vocabulary space for each hierarchy
        self.decoder_mlp = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.embedding_dim,
                    self.num_embeddings_per_hierarchy,
                    bias=False,
                )
                for _ in range(self.num_hierarchies)
            ]
        )

        # the key value names for the prediction output
        self.prediction_key_name = prediction_key_name
        self.prediction_value_name = prediction_value_name

        # Remove unused encoder parameters to avoid DDP issues

    def _remove_unused_encoder_components(self):
        """
        Remove or disable unused encoder components to avoid DDP unused parameter issues.
        This method removes the encoder and encoder-decoder attention layers that are not used
        in the decoder-only architecture.
        """
        # Remove the entire encoder since it's not used in decoder-only architecture
        if hasattr(self, 'encoder'):
            delattr(self, 'encoder')

        # Remove shared embedding if it exists and is not used
        if hasattr(self, 'shared'):
            delattr(self, 'shared')

        # Remove encoder-decoder attention layers from decoder blocks
        if hasattr(self, 'decoder') and hasattr(self.decoder, 'decoder'):
            decoder_model = self.decoder.decoder
            if hasattr(decoder_model, 'block'):
                for i, block in enumerate(decoder_model.block):
                    # T5 decoder blocks have 3 layers: [SelfAttention, EncDecAttention, FF]
                    # We want to remove the EncDecAttention layer (index 1)
                    if hasattr(block, 'layer') and len(block.layer) > 1:
                        # Check if layer[1] has EncDecAttention
                        if hasattr(block.layer[1], 'EncDecAttention'):
                            # Create a custom layer that skips EncDecAttention
                            original_layer = block.layer[1]

                            # Create a new layer that only does layer norm and skip connection
                            class SkipEncDecAttentionLayer(torch.nn.Module):
                                def __init__(self, layer_norm):
                                    super().__init__()
                                    self.layer_norm = layer_norm

                                def forward(self, hidden_states, *args, **kwargs):
                                    # Just apply layer norm and return (skip connection)
                                    return (self.layer_norm(hidden_states), None, None, None)

                            # Replace with skip layer
                            block.layer[1] = SkipEncDecAttentionLayer(original_layer.layer_norm)

    def forward(
            self,
            attention_mask: torch.Tensor,
            input_ids: torch.Tensor,
            user_id: Optional[torch.Tensor] = None,
            future_ids: Optional[torch.Tensor] = None,
            future_attention_mask: Optional[torch.Tensor] = None,
            use_cache: bool = False,
            past_key_values: Optional[DynamicCache] = None,
            **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, DynamicCache]]:
        """
        Forward pass for the decoder-only model.
        
        In training mode:
        - input_ids: the input sequence (context)
        - future_ids: the target sequence to predict
        - The model concatenates input and future sequences and applies causal masking
        
        In inference mode:
        - input_ids: the input sequence (context)
        - future_ids: the partially generated sequence
        - use_cache: whether to use KV cache for efficient generation
        - past_key_values: cached KV values from previous steps
        """

        # Process input sequence
        inputs_embeds, attention_mask = self._process_input_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_id=user_id,
            future_ids=future_ids
        )

        # # If we have future_ids (training or conditional generation), process them too
        # if future_ids is not None:
        #     future_embeds = self._process_future_sequence(future_ids=future_ids)
        #
        #     # Concatenate input and future embeddings
        #     sequence_embeds = torch.cat([inputs_embeds, future_embeds], dim=1)
        #
        #     # Create combined attention mask
        #     if future_attention_mask is not None:
        #         # Add attention for BOS token (always attend)
        #         # bos_attention = torch.ones(future_attention_mask.size(0), 1, device=future_attention_mask.device, dtype=future_attention_mask.dtype)
        #         # extended_future_mask = torch.cat([bos_attention, future_attention_mask], dim=1)
        #         combined_attention_mask = torch.cat([attention_mask, future_attention_mask], dim=1)
        #     else:
        #         # Create attention mask for future sequence including BOS token
        #         # BOS token (1) + future_ids length
        #         future_mask = torch.ones(future_ids.size(0), future_embeds.size(1), device=future_ids.device, dtype=torch.int32)
        #         combined_attention_mask = torch.cat([attention_mask, future_mask], dim=1)
        # else:
        # Only input sequence (inference with KV cache)
        sequence_embeds = inputs_embeds
        combined_attention_mask = attention_mask

        # Forward pass through decoder
        decoder_output = self.decoder(
            sequence_embedding=sequence_embeds,
            attention_mask=combined_attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        if use_cache:
            hidden_states, past_key_values = decoder_output
            return hidden_states, past_key_values
        else:
            return decoder_output

    def _process_input_sequence(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            user_id: Optional[torch.Tensor] = None,
            future_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process the input sequence for the decoder.
        """
        input_ids, attention_mask = self.concatenate_future_ids(input_ids, attention_mask, future_ids)

        # Shift IDs to match hierarchy structure
        shifted_sids = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds = self.get_embedding_table()(shifted_sids)

        if self.sep_token is not None:
            (
                inputs_embeds,
                attention_mask,
            ) = self._inject_sep_token_between_sids(
                id_embeddings=inputs_embeds,
                attention_mask=attention_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        # Add user embedding if provided
        if user_id is not None and self.user_embedding is not None:
            # Preprocessing function pads user_id with zeros
            # so we only need to take the first column
            user_id = user_id[:, 0]

            # TODO (clark): here we assume remainder hashing, which is different from LSH hashing used in TIGER.
            user_embeds = self.user_embedding(
                torch.remainder(user_id, self.user_embedding.num_embeddings)
            )

            # Prepending the user_id embedding to the input sequence
            inputs_embeds = torch.cat(
                [
                    user_embeds.unsqueeze(1),
                    inputs_embeds,
                ],
                dim=1,
            )
            # No need to modify attention mask here since user token participates in all attention

        return inputs_embeds, attention_mask

    def concatenate_future_ids(self, input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               future_ids: Optional[torch.Tensor] = None):
        """
        将future_ids根据attention_mask的有效位置拼接到input_ids后面

        Args:
            input_ids: [B, S] - 输入token ids
            attention_mask: [B, S] - attention mask，1表示有效位置，0表示padding位置
            future_ids: [B, 4] - 要拼接的future ids

        Returns:
            new_input_ids: [B, S + 4] - 拼接后的input_ids
            new_attention_mask: [B, S + 4] - 拼接后的attention_mask
        """
        if future_ids is None:
            return input_ids, attention_mask
        B, S = input_ids.shape
        _, future_len = future_ids.shape

        # 计算每个序列有效token的结束位置
        valid_lengths = attention_mask.sum(dim=1)  # [B], 每个序列的有效长度

        # 创建新的tensor
        new_input_ids = torch.zeros(B, S + future_len, dtype=input_ids.dtype, device=input_ids.device)
        new_attention_mask = torch.zeros(B, S + future_len, dtype=attention_mask.dtype, device=attention_mask.device)

        # 复制原始数据
        new_input_ids[:, :S] = input_ids
        new_attention_mask[:, :S] = attention_mask

        # 批量处理每个序列
        for i in range(B):
            valid_len = valid_lengths[i].item()
            # 在有效位置后插入future_ids
            new_input_ids[i, valid_len:valid_len + future_len] = future_ids[i]
            new_attention_mask[i, valid_len:valid_len + future_len] = 1

        return new_input_ids, new_attention_mask

    def _process_future_sequence(
            self,
            future_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process the future/target sequence for training.
        """
        # Shift future IDs to match hierarchy structure
        batch_size, seq_len = future_ids.shape
        shifted_future_sids = self._add_repeating_offset_to_rows(
            input_sids=future_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=torch.ones(batch_size, seq_len, device=future_ids.device, dtype=torch.int32),
        )
        future_embeds = self.get_embedding_table()(shifted_future_sids)

        # Prepend BOS token for autoregressive prediction
        bos_tokens = self.bos_token.unsqueeze(0).expand(batch_size, 1, -1)
        future_embeds = torch.cat([bos_tokens, future_embeds], dim=1)

        return future_embeds

    def generate(
            self,
            attention_mask: torch.Tensor,
            input_ids: torch.Tensor,
            user_id: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate semantic IDs given the input context using beam search.
        
        Parameters:
        -----------
        attention_mask (torch.Tensor): The attention mask for the input context.
        input_ids (torch.Tensor): The input IDs for the context.
        user_id (torch.Tensor): The user IDs for the context.
        
        Returns:
        --------
        generated_ids (torch.Tensor): The generated semantic IDs.
        marginal_log_prob (torch.Tensor): The marginal log probabilities of the generated sequences.
        """
        # Initialize generated sequences
        generated_ids = None
        marginal_log_prob = None

        for hierarchy in range(self.num_hierarchies):
            past_key_values = DynamicCache()
            if generated_ids is not None:
                # we generated something before
                # we need to reshape the generated ids so that
                # the number of beams equals to batch size * top_k
                squeezed_generated_ids = generated_ids.reshape(-1, hierarchy).to(
                    input_ids.device
                )  # shape: (batch_size * top_k, hierarchy)

                repeated_input_ids = input_ids.repeat_interleave(
                    self.top_k_for_generation, dim=0
                )
                # shape: (batch_size * top_k, seq_len+1, hidden_dim)
                # +1 because we have user_id token

                repeated_attention_mask = (
                    attention_mask.repeat_interleave(
                        self.top_k_for_generation, dim=0
                    )
                )  # shape: (batch_size * top_k, seq_len+1)
            else:
                # we haven't generated anything yet!
                # the number of beams currently equals to batch size
                squeezed_generated_ids = None
                repeated_input_ids = input_ids
                repeated_attention_mask = attention_mask

            sequence_embeds, combined_attention_mask = self._process_input_sequence(input_ids=repeated_input_ids,
                                                                                    attention_mask=repeated_attention_mask,
                                                                                    user_id=user_id,
                                                                                    future_ids=squeezed_generated_ids)

            repeated_input_ids, repeated_attention_mask = self.concatenate_future_ids(repeated_input_ids,
                                                                                      repeated_attention_mask,
                                                                                      squeezed_generated_ids)
            # sequence_embeds, combined_attention_mask = self.concatenate_future_ids(
            #     input_ids=repeated_input_ids,
            #     attention_mask=repeated_attention_mask,
            #     future_ids=squeezed_generated_ids,
            # )

            decoder_output = self.decoder(
                sequence_embedding=sequence_embeds,
                attention_mask=combined_attention_mask,
                use_cache=False,
                past_key_values=past_key_values,
            )

            batch_size = decoder_output.size(0)
            # 使用 combined_attention_mask 而不是 attention_mask，因为它已经被 repeat_interleave 处理过
            seq_lengths = combined_attention_mask.sum(dim=1)  # [B * top_k], 每个样本的有效长度
            last_indices = (seq_lengths - 1).long()
            latest_output_representation = decoder_output[
                torch.arange(batch_size, device=decoder_output.device), last_indices]  # [B, C]

            # Calculate logits for the next token
            candidate_logits = self.decoder_mlp[hierarchy](
                latest_output_representation
            )

            # Perform beam search step
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

    def get_embedding_table(self, hierarchy: Optional[int] = None):
        """
        Get the embedding table for the given hierarchy.
        """
        if hierarchy is not None:
            return self.item_sid_embedding_table(
                torch.arange(
                    hierarchy * self.num_embeddings_per_hierarchy,
                    (hierarchy + 1) * self.num_embeddings_per_hierarchy,
                ).to(self.device)
            )
        return self.item_sid_embedding_table

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
                },
            )
            return generated_ids, 0  # returning 0 here because we don't have a loss

        # Training mode - calculate loss
        fut_ids = None
        for label in label_data.labels:
            curr_label = label_data.labels[label]
            fut_ids = curr_label.reshape(model_input.mask.size(0), -1)

        # Forward pass for training
        model_output = self.forward(
            attention_mask=model_input.mask,
            future_ids=fut_ids,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )  # [B, S + 4, C]

        batch_size = model_output.size(0)
        seq_lengths = model_input.mask.sum(dim=1)  # [B], 每个样本的有效长度
        last_indices = (seq_lengths - 1).long()

        # 优化：使用高级索引一次性提取所有需要的位置
        batch_idx = torch.arange(batch_size, device=model_output.device).unsqueeze(1)  # [B, 1]
        time_offsets = torch.arange(4, device=model_output.device).unsqueeze(0)  # [1, 4]
        gather_indices = last_indices.unsqueeze(1) + time_offsets  # [B, 4]

        model_output = model_output[batch_idx, gather_indices]  # [B, 4, C]

        # Compute loss for each hierarchy level
        loss = 0
        for hierarchy in range(self.num_hierarchies):
            logits = self.decoder_mlp[hierarchy](model_output[:, hierarchy])
            loss += self.loss_function(
                input=logits,
                target=fut_ids[:, hierarchy].long(),
            )

        rec_loss = 0
        # Check for unused parameters after loss computation but before return
        # This helps debug DDP unused parameter issues
        # if self.training and hasattr(self, 'trainer') and self.trainer.global_step % 100 == 0:
        #     # Only check every 100 steps to avoid too much logging
        #     unused = []
        #     for name, p in self.named_parameters():
        #         if p.requires_grad and p.grad is None:
        #             unused.append(name)
        #
        #     if unused:
        #         print(f"[DDP unused parameters] step={self.trainer.global_step} count={len(unused)}")
        #         for n in unused:
        #             print(" -", n)
        return model_output, loss, rec_loss


class DecoderModule(torch.nn.Module):
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
        decoder_outputs: Seq2SeqModelOutput = self.decoder(
            attention_mask=attention_mask,
            inputs_embeds=sequence_embedding,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        embeddings = decoder_outputs.last_hidden_state

        if use_cache:
            return embeddings, decoder_outputs.past_key_values
        return embeddings


class HSTUDecoderModule(torch.nn.Module):
    """
    Decoder module specifically designed for HSTU (Hierarchical Sequential Transduction Unit) models.
    Unlike the standard DecoderModule which expects Hugging Face PreTrainedModel,
    this module works with custom HSTU architecture.
    """

    def __init__(
            self,
            decoder: nn.Module,
            decoder_mlp: Optional[torch.nn.Module] = None,
            bos_token: Optional[torch.nn.Parameter] = None,
    ) -> None:
        """
        Initialize the HSTUDecoderModule.

        Parameters:
        -----------
        decoder (nn.Module): The HSTU decoder model (e.g., HSTURec).
        decoder_mlp (torch.nn.Module): Optional MLP layers for output projection.
        bos_token (Optional[torch.nn.Parameter]): Optional BOS token for prompting.
        """
        super().__init__()
        self.decoder = decoder
        self.bos_token = bos_token
        self.decoder_mlp = decoder_mlp

    def forward(
            self,
            attention_mask: torch.Tensor,
            sequence_embedding: torch.Tensor,
            use_cache: bool = False,
            past_key_values: Optional[DynamicCache] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[DynamicCache]]]:
        """
        Forward pass for the HSTU decoder module.

        Parameters:
        -----------
        attention_mask (torch.Tensor): The attention mask for the decoder.
            Shape: [batch_size, seq_len]
        sequence_embedding (torch.Tensor): The input sequence embedding.
            Shape: [batch_size, seq_len, d_model]
        use_cache (bool): Whether to use cache (not supported for HSTU yet).
        past_key_values (Optional[DynamicCache]): Cache for past key values (not supported yet).

        Returns:
        --------
        embeddings (torch.Tensor): The output embeddings from the decoder.
            Shape: [batch_size, seq_len, d_model]
        """
        # HSTU forward pass
        embeddings = self.decoder(sequence_embedding, mask=attention_mask)

        # Note: HSTU doesn't support KV caching yet
        if use_cache:
            # Return None for past_key_values since HSTU doesn't support caching
            return embeddings, None
        return embeddings
