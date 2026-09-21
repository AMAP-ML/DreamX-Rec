from typing import Any, Optional, Tuple, Union, Dict, List

import torch
import transformers
from torch import nn
from torchmetrics.aggregation import BaseAggregator
from transformers.cache_utils import DynamicCache
import torch.nn.functional as F
from lightning.pytorch.trainer.states import TrainerFn
from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.modules.semantic_id.decoder_only_model import DecoderModule, HSTUDecoderModule
from src.models.modules.semantic_id.tiger_generation_model import SemanticIDGenerativeRecommender, \
    SemanticIDEncoderModule, T5MultiLayerFF
from src.utils.utils import (
    get_parent_module_and_attr,
    unique_ids_and_sids
)


class IntRRDecoderOnly(SemanticIDGenerativeRecommender):

    def __init__(
            self,
            top_k_for_generation: int = 256,
            codebooks: torch.Tensor = None,
            embedding_dim: int = None,
            num_hierarchies: int = None,
            num_embeddings_per_hierarchy: int = None,
            num_user_bins: Optional[int] = None,
            mlp_layers: Optional[int] = None,
            should_check_prefix: bool = False,
            should_add_sep_token: bool = True,
            prediction_key_name: str = "user_id",
            prediction_value_name: str = "item_ids",
            encoder_input_emb: bool = False,
            pretrained_item_embeddings: Optional[Dict[str, torch.Tensor]] = None,
            total_item_num: Optional[int] = None,
            item_embedding_dim: Optional[int] = None,
            codebook_activation: str = "ReLU",
            temperature: Optional[float] = None,
            codebook_hidden_dim: Optional[int] = None,
            llm_embedding_dim: Optional[int] = None,
            user_vec_mode: Optional[int] = None,
            reconstruction_loss_weight: float = 0.1,
            should_add_bos_token: bool = True,
            beamsearch: bool = False,
            emb_projection_flag: bool = False,
            u_groudtruth: bool = True,
            i_groudtruth: bool = False,
            split_codebook_mlp: bool = False,
            rec_loss_step: Optional[int] = None,
            rec_level: Optional[int] = None,
            **kwargs,
    ) -> None:
        """
        Initialize the IntRRDecoderOnly module for semantic ID generation.

        Args:
            top_k_for_generation: Number of top-k candidates for generation.
            codebooks: Codebook tensor with shape (num_items, num_hierarchies).
            embedding_dim: Dimension of the embeddings.
            num_hierarchies: Number of hierarchies in the codebooks.
            num_embeddings_per_hierarchy: Number of embeddings per hierarchy.
            num_user_bins: Number of bins for user embedding table.
            mlp_layers: Number of MLP layers in encoder and decoder.
            should_check_prefix: Whether to check if prefix is valid during generation.
            should_add_sep_token: Whether to add separation token between items.
            prediction_key_name: Key name for prediction output.
            prediction_value_name: Value name for prediction output.
            encoder_input_emb: Whether to use encoder input embedding.
            pretrained_item_embeddings: Pre-trained item embeddings for initialization.
            total_item_num: Total number of items in the dataset.
            item_embedding_dim: Dimension of item embeddings.
            codebook_activation: Activation function for codebook layers.
            temperature: Temperature for softmax in codebook similarity.
            codebook_hidden_dim: Hidden dimension for codebook MLP.
            llm_embedding_dim: Dimension of LLM embeddings.
            user_vec_mode: User vector computation mode (1, 2, or 3).
            reconstruction_loss_weight: Weight for reconstruction loss (default: 0.1).
            should_add_bos_token: Whether to add BOS token.
            beamsearch: Whether to use beam search for generation.
            emb_projection_flag: Whether to project embeddings before codebook lookup.
            u_groudtruth: Whether to use ground truth for user embeddings during training.
            i_groudtruth: Whether to use ground truth for item embeddings during training.
            split_codebook_mlp: Whether to use separate MLP for input/output tasks.
            rec_loss_step: Step to disable reconstruction loss.
            rec_level: Number of levels to compute reconstruction loss.
            **kwargs: Additional arguments passed to parent class.
        """
        # Initialize prefix validation cache before calling super().__init__
        self._prefix_valid_cache = None
        self._prefix_cache_hierarchy = -1

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

        # Activation function mapping
        self.activation_mapping = {
            "ReLU": nn.ReLU,
            "Softmax": nn.Softmax,
            "Sigmoid": nn.Sigmoid,
            "Tanh": nn.Tanh,
            "LeakyReLU": nn.LeakyReLU,
            "ELU": nn.ELU,
            "GELU": nn.GELU,
            "Swish": nn.SiLU,  # Swish is equivalent to SiLU in PyTorch
            "Mish": nn.Mish
        }

        # Get configured activation function class
        if codebook_activation in self.activation_mapping:
            self.codebook_activation_fn = self.activation_mapping[codebook_activation]
        else:
            raise ValueError(f"Unsupported activation function: {codebook_activation}. "
                             f"Supported functions: {list(self.activation_mapping.keys())}")

        self.encoder = SemanticIDEncoderModule(
            encoder=self.encoder,
        )

        # bos_token used to prompt the decoder to generate the first token
        self.bos_token = torch.nn.Parameter(
            torch.randn(1, self.embedding_dim), requires_grad=True
        ) if should_add_bos_token else None

        # In decoder-only architecture, we use the same model for both encoding and decoding
        # Check if decoder is a Hugging Face model or a custom model (like HSTU)
        if hasattr(self.decoder, 'config') and hasattr(self.decoder.config, 'is_decoder'):
            # Standard Hugging Face decoder (e.g., T5)
            self.decoder = DecoderModule(
                decoder=self.decoder,
            )
        else:
            # Custom decoder (e.g., HSTU)
            self.decoder = HSTUDecoderModule(
                decoder=self.decoder,
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

        self.codebook_emb_dim = self.embedding_dim
        # Initialize codebooks and move to correct device
        self.intrr_codebooks = torch.nn.ParameterList([
            torch.nn.Parameter(
                torch.randn(self.num_embeddings_per_hierarchy, self.codebook_emb_dim)
            )
            for _ in range(self.num_hierarchies)
        ])

        self.emb_projection_flag = emb_projection_flag
        if self.emb_projection_flag:
            self.codebook_project_mlp = torch.nn.Linear(self.embedding_dim, self.codebook_emb_dim)

        self.item_embedding_dim = item_embedding_dim if item_embedding_dim else self.embedding_dim

        self.llm_embedding_dim = llm_embedding_dim if llm_embedding_dim else self.embedding_dim

        # itememb_input_mode defaults to 1
        self.itememb_input_mode = 1
        self.use_input_mlp = True
        if pretrained_item_embeddings and total_item_num:
            self.input_mlp = torch.nn.Linear(2048 + self.item_embedding_dim, self.embedding_dim)
        elif total_item_num:
            self.input_mlp = torch.nn.Linear(self.item_embedding_dim, self.embedding_dim)
        else:
            self.input_mlp = torch.nn.Linear(2048, self.embedding_dim)

        # Store pre-trained item embeddings
        self.pretrained_item_embeddings = pretrained_item_embeddings['id'].to(
            self.device) if pretrained_item_embeddings else None

        self.item_embedding: torch.nn.Embedding = (
            self._spawn_embedding_tables(
                num_embeddings=total_item_num,
                embedding_dim=self.item_embedding_dim,
            )
            if total_item_num
            else None
        )
        self.total_item_num = total_item_num

        self.split_codebook_mlp = split_codebook_mlp

        # Codebook MLP: [prev_emb, item_emb] -> codebook_emb_dim
        self.codebook_linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.codebook_emb_dim * 2, codebook_hidden_dim),
                self.codebook_activation_fn(),
                nn.Linear(codebook_hidden_dim, self.codebook_emb_dim)
            )
            for _ in range(len(self.intrr_codebooks) - 1)
        ])
        # u codebook mlp (when split_codebook_mlp is True)
        if self.split_codebook_mlp:
            self.split_codebook_mlp_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.codebook_emb_dim * 2, codebook_hidden_dim),
                    self.codebook_activation_fn(),
                    nn.Linear(codebook_hidden_dim, self.codebook_emb_dim)
                )
                for _ in range(len(self.intrr_codebooks) - 1)
            ])

        self.temperature = temperature if temperature is not None else 1.0
        if self.temperature == 0:
            self.temp_nn = nn.Parameter(torch.tensor(0.0))

        self.user_vec_mode = user_vec_mode
        if self.user_vec_mode == 2:
            self.user_vec_linear = nn.Linear(2 * self.codebook_emb_dim, self.codebook_emb_dim)

        # Set reconstruction loss weight
        self.reconstruction_loss_weight = reconstruction_loss_weight

        self.should_add_bos_token = should_add_bos_token
        self.beamsearch = beamsearch
        self.u_groudtruth = u_groudtruth
        self.i_groudtruth = i_groudtruth
        self.rec_loss_step = rec_loss_step  # When not None and global_step > rec_loss_step, disable reconstruction_loss
        self.rec_level = rec_level  # When not None, only compute reconstruction loss for first min(rec_level, L) levels

        self._all_items_cache = {'is_valid': False}

    def _compute_reconstruction_loss(
            self,
            output_max_indices: List[torch.Tensor],
            output_similarity: List[torch.Tensor],
            input_sids: torch.Tensor,
            unique_input_ids: torch.Tensor,
            inverse_indices: torch.Tensor,
            attention_mask: torch.Tensor,
            stage: str = "train"
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, S, L = input_sids.shape

        effective_levels = min(self.rec_level, L) if self.rec_level is not None else L

        attention_mask_2d = attention_mask.squeeze(-1)
        valid_true_sids = input_sids[attention_mask_2d.bool()]

        all_level_losses = []
        all_level_accuracies = []

        for level in range(effective_levels):
            level_softmax = output_similarity[level]
            level_probs = level_softmax[inverse_indices]

            if level_probs.dim() == 3:
                level_probs = level_probs.squeeze(1)

            level_targets = valid_true_sids[:, level].long()

            level_preds = torch.argmax(level_probs, dim=-1)
            level_correct = (level_preds == level_targets).float()
            level_accuracy = level_correct.mean()
            all_level_accuracies.append(level_accuracy)

            level_loss = F.cross_entropy(level_probs, level_targets, reduction='none')

            all_level_losses.append(level_loss)

        all_level_losses = torch.stack(all_level_losses, dim=1)
        position_losses = all_level_losses.sum(dim=1)
        reconstruction_loss = position_losses.sum() / position_losses.shape[0]

        return reconstruction_loss, all_level_accuracies

    def training_step(
            self,
            batch: Tuple[Tuple[SequentialModelInputData, SequentialModuleLabelData]],
            batch_idx: int,
    ) -> torch.Tensor:
        """
        Override parent training_step to add reconstruction accuracy logging.
        
        Args:
            batch: Lightning-wrapped training batch data.
            batch_idx: Batch index.
            
        Returns:
            loss: Training loss.
        """
        import time
        start = time.time()

        # Lightning wraps batch in tuple during training, need to unpack
        batch = batch[0]
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]

        # Call model_step to get output and loss
        model_output, loss, rec_loss = self.model_step(
            model_input=model_input, label_data=label_data
        )

        # Update and log training sample count
        batch_size = model_input.mask.shape[0]
        self.train_samples(batch_size)
        self.log(
            "train/samples",
            self.train_samples,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Update and log training loss
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Update and log reconstruction loss
        self.rec_loss(rec_loss)
        self.log(
            "train/rec_loss",
            self.rec_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Call custom training loop function if provided
        if self.training_loop_function is not None:
            self.training_loop_function(self, loss)

        # Log training step time
        end = time.time()
        self.log("train/train_step_time", end - start, on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=batch_size,
                 )

        return loss

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

        fut_sids = None

        generated_ids, marginal_probs = self.generate(
            attention_mask=model_input.mask,
            fut_sids=fut_sids,
            **{
                self.feature_to_model_input_map[k]: v
                for k, v in model_input.transformed_sequences.items()
                if k in self.feature_to_model_input_map
            },
        )

        device = marginal_probs.device

        self.evaluator(
            marginal_probs=marginal_probs,
            generated_ids=generated_ids,
            labels=label_data.labels['sid_sequence_data'].to(device) if 'sid_sequence_data' in label_data.labels else
            label_data.labels['sequence_data'].to(device),
        )

        loss_to_aggregate(loss)

        # Log eval_step time (epoch-level average only)
        end = time.time()
        self.log("val/eval_step_time", end - start, on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=batch_size,
                 )

    def model_step(
            self,
            model_input: SequentialModelInputData,
            label_data: Optional[SequentialModuleLabelData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Override parent model_step to include reconstruction loss with weighting.

        Args:
            model_input: The input data to the model.
            label_data: The label data to the model. Its optional as it is not required for inference.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (model_output, combined_loss)
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

        fut_sids = label_data.labels['sid_sequence_data'].reshape(model_input.mask.size(0), -1)
        fut_ids = label_data.labels['sequence_data'].reshape(model_input.mask.size(0), -1)

        # Forward pass
        model_output = self.forward(
            attention_mask=model_input.mask,
            future_ids=fut_ids,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        if self.user_vec_mode == 1:
            batch_size = model_output.size(0)
            seq_lengths = model_input.mask.sum(dim=1)  # [B], sequence length for each sample

            if self.should_add_bos_token:
                last_indices = seq_lengths.long()  # [B], last valid index (offset by bos token)
            else:
                last_indices = (seq_lengths - 1).long()
            model_output = model_output[torch.arange(batch_size, device=model_output.device), last_indices].unsqueeze(
                1)  # [B, 1, C]

        if self.user_vec_mode == 2:
            batch_size = model_output.size(0)
            seq_lengths = model_input.mask.sum(dim=1)  # [B], sequence length for each sample

            last_indices = (seq_lengths).long()  # [B], last valid position (offset by bos token)
            second_last_indices = (seq_lengths - 1).long()  # [B], second last valid position

            last_hidden = model_output[torch.arange(batch_size, device=model_output.device), last_indices]  # [B, C]
            second_last_hidden = model_output[
                torch.arange(batch_size, device=model_output.device), second_last_indices]  # [B, C]
            model_output = torch.stack([second_last_hidden, last_hidden], dim=1)  # [B, 2, C]

            model_output = model_output.reshape(model_output.size(0), -1)
            model_output = self.user_vec_linear(model_output)

        if self.user_vec_mode == 3:
            model_output = model_output[:, -1:, :]

        model_output = model_output.squeeze(1)  # Squeeze only dim 2, keep [B, E]

        u_fut_sids = None
        if self.u_groudtruth:
            u_fut_sids = fut_sids
        # Transfer item vec to sid
        (output_emb,
         output_similarity,
         output_similarity_softmax,
         output_similarity_onehot,
         output_max_indices,
         output_max_score) = self.codebook_ran_module(
            model_output,
            self.emb_projection_flag,
            u_fut_sids,
            True,
            self.split_codebook_mlp,
            task_type="output"
        )

        ce_loss = 0.0
        ce_level_accuracies = []

        for level in range(self.num_hierarchies):
            level_similarity = output_similarity[level]  # [batch_size, codebook_size]
            level_target = fut_sids[:, level].long()  # [batch_size]
            level_loss = F.cross_entropy(level_similarity, level_target)
            ce_loss += level_loss

        self.ce_level_accuracies = ce_level_accuracies

        reconstruction_loss_weight = getattr(self, 'reconstruction_loss_weight', 0.3)

        combined_loss = ce_loss
        reconstruction_loss = None

        if hasattr(self, 'reconstruction_loss') and self.reconstruction_loss is not None:
            reconstruction_loss = self.reconstruction_loss

            should_use_rec_loss = True
            if self.rec_loss_step is not None and hasattr(self,
                                                          'global_step') and self.global_step > self.rec_loss_step:
                should_use_rec_loss = False

            if should_use_rec_loss:
                combined_loss = ce_loss + reconstruction_loss_weight * reconstruction_loss

            self.reconstruction_loss = None
        else:
            if hasattr(self, 'log'):
                self.log("loss/ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                         sync_dist=True)

        return model_output, combined_loss, reconstruction_loss

    def forward(
            self,
            attention_mask: torch.Tensor,
            input_ids: torch.Tensor,
            input_sids: Optional[torch.Tensor] = None,
            user_id: Optional[torch.Tensor] = None,
            future_ids: Optional[torch.Tensor] = None,
            future_attention_mask: Optional[torch.Tensor] = None,
            use_cache: bool = False,
            past_key_values: Optional[DynamicCache] = None,
            **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, DynamicCache]]:
        sequence_embeds, combined_attention_mask = self._process_input_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_id=user_id,
            input_sids=input_sids,
            add_bos_token=self.should_add_bos_token
        )

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

    def generate(
            self,
            attention_mask: torch.Tensor,
            input_ids: torch.Tensor,
            user_id: torch.Tensor = None,
            input_sids: torch.Tensor = None,
            fut_sids: Optional[torch.Tensor] = None,
            **kwargs
    ):

        past_key_values = None

        outputs = self.forward(
            input_ids=input_ids,
            input_sids=input_sids,
            attention_mask=attention_mask,
            use_cache=False,
            past_key_values=past_key_values
        )

        if isinstance(outputs, tuple):
            hidden_states, past_key_values = outputs
        else:
            hidden_states = outputs

        # Get last token hidden state based on attention_mask
        if self.user_vec_mode == 1:
            # Find last valid position for each sample based on attention_mask
            batch_size = hidden_states.size(0)
            seq_lengths = attention_mask.sum(dim=1)  # [B], sequence length per sample

            # Extract output at last valid position for each sample
            if self.should_add_bos_token:
                last_indices = seq_lengths.long()  # [B], last valid position index
            else:
                last_indices = (seq_lengths - 1).long()
            last_hidden = hidden_states[
                torch.arange(batch_size, device=hidden_states.device), last_indices]  # [B, 1, C]

        elif self.user_vec_mode == 2:
            # Find last two valid positions based on attention_mask
            batch_size = hidden_states.size(0)
            seq_lengths = attention_mask.sum(dim=1)  # [B], sequence length per sample

            # Extract outputs at last two valid positions
            last_indices = seq_lengths.long()  # [B], last valid position
            second_last_indices = (seq_lengths - 1).long()  # [B], second-last valid position

            last_hidden_token = hidden_states[
                torch.arange(batch_size, device=hidden_states.device), last_indices]  # [B, C]
            second_last_hidden_token = hidden_states[
                torch.arange(batch_size, device=hidden_states.device), second_last_indices]  # [B, C]

            last_hidden = torch.stack([second_last_hidden_token, last_hidden_token], dim=1)  # [B, 2, C]
            b, _, _ = last_hidden.shape
            last_hidden = last_hidden.reshape(b, 1, -1)
            last_hidden = self.user_vec_linear(last_hidden)
        elif self.user_vec_mode == 3:
            last_hidden = hidden_states[:, -1:, :]

        if self.beamsearch:
            last_hidden = last_hidden.squeeze(1)  # Squeeze only dim 2, keep [B, E]

            # Dimension check and protection
            if last_hidden.dim() == 1:
                last_hidden = last_hidden.unsqueeze(0)  # [B, E]

            # beam search for sid
            generated_ids = None
            marginal_log_prob = None

            for hierarchy in range(self.num_hierarchies):
                if generated_ids is not None:
                    # we generated something before
                    squeezed_generated_ids = generated_ids.reshape(-1, hierarchy).to(
                        input_ids.device
                    )  # shape: (batch_size * top_k, hierarchy)

                    repeated_last_hidden = last_hidden.repeat_interleave(
                        self.top_k_for_generation, dim=0
                    )
                else:
                    # we are generating the first token
                    squeezed_generated_ids = None
                    repeated_last_hidden = last_hidden

                # Call inference module
                (output_emb,
                 output_similarity, output_similarity_softmax, _,
                 output_max_indices,
                 output_max_score) = self.codebook_ran_module_infer(
                    repeated_last_hidden, squeezed_generated_ids, hierarchy
                )

                candidate_logits = output_similarity

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
                    sumlogp=True,
                )

            return generated_ids, marginal_log_prob

    def _process_input_sequence(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            user_id: Optional[torch.Tensor] = None,
            input_sids: Optional[torch.Tensor] = None,
            add_bos_token: bool = False,
    ) -> torch.Tensor:
        """
        Process the input sequence for the decoder.
        """
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze()
        batch_size, seq_len = input_ids.shape

        if self.encoder_input_emb:
            # input_ids shape: (batch_size, seq_length * embedding_dim)
            # reshape to (batch_size, seq_length, embedding_dim)
            # input_ids = input_ids.reshape(attention_mask.shape[0], attention_mask.shape[1], -1)

            # Get unique input_ids and corresponding index mapping
            # unique_input_ids, inverse_indices = torch.unique(input_ids[attention_mask.bool()], return_inverse=True)

            unique_input_ids, inverse_indices, unique_input_sids = unique_ids_and_sids(input_ids, attention_mask,
                                                                                       input_sids)
            unique_input_ids = unique_input_ids.squeeze(1)

            # Process input embeddings (itememb_input_mode = 1)
            unique_input_emb = None
            if self.pretrained_item_embeddings is not None:
                pre_input_emb = self.pretrained_item_embeddings.to(input_ids.device)[unique_input_ids]
                unique_input_emb = pre_input_emb

            if self.item_embedding is not None:
                item_embeddings = self.item_embedding(unique_input_ids)
                unique_input_emb = torch.cat([unique_input_emb, item_embeddings],
                                             dim=1) if unique_input_emb is not None else item_embeddings
            unique_input_emb = self.input_mlp(unique_input_emb)

            # Get new embeddings through codebook_ran_module
            i_input_sids = None
            if self.i_groudtruth:
                i_input_sids = unique_input_sids
            (unique_output_emb,
             output_similarity,
             output_similarity_softmax,
             output_similarity_onehot,
             output_max_indices,
             output_max_score) = self.codebook_ran_module(
                unique_input_emb,
                self.emb_projection_flag,
                i_input_sids,
                task_type="input"
            )

            # Compute reconstruction loss: compare predicted semantic IDs with ground truth
            if input_sids is not None:
                # Determine current stage: training, validation, or testing
                # TrainerFn.FITTING includes training + validation, need to distinguish by training state
                if hasattr(self, 'trainer') and hasattr(self.trainer, 'state'):
                    if self.trainer.state.fn == TrainerFn.FITTING:
                        # FITTING includes training and validation, distinguish by self.training
                        stage = "train" if self.training else "val"
                    elif self.trainer.state.fn == TrainerFn.TESTING:
                        stage = "test"
                    else:
                        # Other cases (e.g., VALIDATING, PREDICTING) default to val
                        stage = "val"
                else:
                    # Without trainer, determine by self.training
                    stage = "train" if self.training else "val"

                reconstruction_loss, rec_level_accuracies = self._compute_reconstruction_loss(
                    output_max_indices,
                    output_similarity,
                    input_sids,
                    unique_input_ids,
                    inverse_indices,
                    attention_mask,
                    stage=stage
                )
                # Save reconstruction loss and accuracies as model attributes for logging in training_step
                self.reconstruction_loss = reconstruction_loss
                self.rec_level_accuracies = rec_level_accuracies
                self.rec_stage = stage

            # Map unique_output_emb back to original input_ids shape using inverse_indices
            # Length of inverse_indices equals the number of True positions in attention_mask
            flat_input_emb = unique_output_emb[inverse_indices]

            # Create a tensor with the same shape as input_ids to store results
            inputs_embeds = torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                unique_output_emb.shape[-1],
                device=input_ids.device,
                dtype=unique_output_emb.dtype
            )
            # Fill processed embeddings into corresponding positions
            inputs_embeds[attention_mask.bool()] = flat_input_emb

        # we shift the IDs here to match the hierarchy structure
        # so that we can use a single embedding table to store the embeddigns for all hierarchies
        else:
            input_sids = input_sids.reshape(input_sids.size(0), -1)
            shifted_sids = self._add_repeating_offset_to_rows(
                input_sids=input_sids,
                codebook_size=self.num_embeddings_per_hierarchy,
                num_hierarchies=self.num_hierarchies,
                attention_mask=attention_mask,
            )
            inputs_embeds = self.get_embedding_table(table_name="encoder")(
                shifted_sids
            )

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
        if add_bos_token:
            if self.bos_token is not None:
                bos_tokens = self.bos_token.unsqueeze(0).expand(batch_size, 1, -1)
                inputs_embeds = torch.cat([bos_tokens, inputs_embeds], dim=1)
                bos_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([bos_mask, attention_mask], dim=1)
        return inputs_embeds, attention_mask

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
        import time
        start = time.time()
        batch_size = batch.mask.shape[0]
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
        end = time.time()
        # Only log epoch-level average, not every step, to match training log frequency
        self.log("test/eval_step_time", end - start, on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=batch_size,
                 )
        return model_output

    def codebook_ran_module_infer(self,
                                        item_emb: torch.Tensor,
                                        generated_ids: torch.Tensor,
                                        hierarchy: int):
        codebook = self.intrr_codebooks[hierarchy]

        codebook_linear_layers = self.codebook_linear_layers
        if self.split_codebook_mlp:
            codebook_linear_layers = self.split_codebook_mlp_layers

        # Get generated_ids for hierarchy level
        if generated_ids is not None:
            prev_hierarchy = hierarchy - 1
            prev_token_ids = generated_ids[:, prev_hierarchy].long()
            former_item = self.intrr_codebooks[prev_hierarchy][prev_token_ids]
            current_input = torch.cat([former_item, item_emb], dim=-1)
            linear_layer = codebook_linear_layers[hierarchy - 1]
            codebook_i_input = linear_layer(current_input)
        else:
            if self.emb_projection_flag:
                item_emb = self.codebook_project_mlp(item_emb)
            codebook_i_input = item_emb

        if len(codebook_i_input.shape) == 2:
            similarity = torch.matmul(codebook_i_input, codebook.t())
        else:
            similarity = torch.einsum('bse, ce -> bsc', codebook_i_input, codebook)

        if self.temperature == 0:
            temperature = torch.exp(self.temp_nn).clamp(min=0.01)
            similarity_softmax = F.softmax(similarity / temperature, dim=-1)
        else:
            similarity_softmax = F.softmax(similarity / self.temperature, dim=-1)

        max_indices = torch.argmax(similarity_softmax, dim=-1)  # [B, S]
        similarity_onehot = F.one_hot(max_indices, num_classes=similarity_softmax.size(-1)).float()
        max_softmax_values = torch.gather(similarity_softmax, -1, max_indices.unsqueeze(-1)).squeeze(-1)

        if len(codebook_i_input.shape) == 2:
            selected_code = torch.matmul(similarity_softmax, codebook)
        else:
            selected_code = torch.einsum('bsc, ce -> bse', similarity_softmax, codebook)

        return selected_code, similarity, similarity_softmax, similarity_onehot, max_indices, max_softmax_values

    def codebook_ran_module(
            self,
            item_emb: torch.Tensor,
            emb_projection_flag: bool = False,
            fut_ids: Optional[torch.Tensor] = None,
            is_training: bool = True,
            split_mlp_flag: bool = False,
            task_type: str = "input",  # Task type: "input" or "output"
    ) -> (torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]):
        """
        Compress/reconstruct item_emb embeddings through multi-layer codebooks.

        Args:
            item_emb (torch.Tensor): Input embedding vectors.
            emb_projection_flag (bool): Whether to project embedding vectors.
            fut_ids (Optional[torch.Tensor]): Ground truth IDs for teacher-forcing.
            is_training (bool): Training flag (kept for backward compatibility, actual stage is determined by trainer.state.fn).
            split_mlp_flag (bool): Whether to use split MLP.
            task_type (str): Task type
                - "input": Input side (ItemID->SID mapping, uses reconstruction loss)
                - "output": Output side (User->SID recommendation, uses CE loss)

        Returns:
            output_emb (torch.Tensor): Encoded embedding vectors.
            output_similarity (list): Similarity scores for each hierarchy level.
            output_similarity_softmax (list): Softmax-processed similarity scores.
            output_similarity_onehot (list): One-hot form of similarity scores.
            output_max_indices (list): Maximum index values.
        """

        if emb_projection_flag:
            item_emb = self.codebook_project_mlp(item_emb)

        codebook_linear_layers = self.codebook_linear_layers
        if split_mlp_flag:
            codebook_linear_layers = self.split_codebook_mlp_layers

        # Initialize output variables
        output_emb = torch.zeros_like(item_emb)
        output_emb_list = []
        output_similarity = []
        output_similarity_softmax = []
        output_similarity_onehot = []
        output_max_indices = []
        output_max_score = []

        # Initialize previous layer output
        former_codebook_output = []

        for i, codebook in enumerate(self.intrr_codebooks):
            if i == 0:
                codebook_i_input = item_emb
            else:
                # Codebook fusion: [prev_ground_emb, item_emb]
                prev_codebook = self.intrr_codebooks[i - 1]
                if fut_ids is not None:
                    ground_emb = prev_codebook[fut_ids[:, i - 1].long()]
                    current_input = torch.cat([ground_emb, item_emb], dim=-1)
                else:
                    current_input = torch.cat([former_codebook_output[i - 1], item_emb], dim=-1)
                linear_layer = codebook_linear_layers[i - 1]
                codebook_i_input = linear_layer(current_input)

            # Compute similarity
            if len(codebook_i_input.shape) == 2:
                similarity = torch.matmul(codebook_i_input, codebook.t())
            else:
                similarity = torch.einsum('bse, ce -> bsc', codebook_i_input, codebook)

            # Apply softmax to similarity
            if self.temperature == 0:
                temperature = torch.exp(self.temp_nn).clamp(min=0.01)
                similarity_softmax = F.softmax(similarity / temperature, dim=-1)
            else:
                similarity_softmax = F.softmax(similarity / self.temperature, dim=-1)
            max_indices = torch.argmax(similarity_softmax, dim=-1)  # [B, S]
            similarity_onehot = F.one_hot(max_indices, num_classes=similarity_softmax.size(-1)).float()
            max_softmax_values = torch.gather(similarity_softmax, -1, max_indices.unsqueeze(-1)).squeeze(-1)

            # STE: forward uses hard, backward uses soft
            # First compute soft selection
            if len(codebook_i_input.shape) == 2:
                soft_code = torch.matmul(similarity_softmax, codebook)
            else:
                soft_code = torch.einsum('bsc, ce -> bse', similarity_softmax, codebook)

            # Then compute hard code
            if len(codebook_i_input.shape) == 2:
                hard_code = codebook[max_indices]  # [B, E]
            else:
                hard_code = F.embedding(max_indices, codebook)  # [B, S, E]

            # STE formula: z_q = z_soft + (z_hard - z_soft).detach()
            selected_code = soft_code + (hard_code - soft_code).detach()

            # Update previous layer state
            former_codebook_output.append(selected_code)

            # Accumulate to output
            output_emb += selected_code
            output_emb_list.append(selected_code)
            output_similarity.append(similarity)
            output_similarity_softmax.append(similarity_softmax)
            output_similarity_onehot.append(similarity_onehot)
            output_max_indices.append(max_indices)
            output_max_score.append(max_softmax_values)
        return output_emb, output_similarity, output_similarity_softmax, output_similarity_onehot, output_max_indices, output_max_score

    def _check_valid_prefix(
            self, prefix: torch.Tensor, batch_size: int = 50000
    ) -> torch.Tensor:
        """
        Optimized version of prefix validation with tensor-based caching.
        Uses vectorized operations to avoid Python loops.

        Args:
            prefix: A tensor of shape [N, hierarchy_level].
            batch_size: The size of the batch to process.

        Returns:
            A boolean tensor of shape [N] indicating the validity of each prefix.
        """
        current_hierarchy = prefix.shape[1]

        # Build cache for current hierarchy if not exists or hierarchy changed
        if (self._prefix_valid_cache is None or
                self._prefix_cache_hierarchy != current_hierarchy):

            # Ensure codebooks are on the correct device
            if prefix.device != self.codebooks.device:
                self.codebooks = self.codebooks.to(prefix.device)

            # Cache the trimmed codebooks as a tensor for vectorized comparison
            trimmed_codebooks = self.codebooks[:, :current_hierarchy]
            self._prefix_valid_cache = trimmed_codebooks
            self._prefix_cache_hierarchy = current_hierarchy

        # Vectorized comparison using broadcasting
        # prefix shape: [N, H]
        # cache shape: [C, H]
        # We want to check if each prefix matches any row in cache

        num_prefixes = prefix.shape[0]

        # Process in batches to avoid memory issues
        results = []
        for i in range(0, num_prefixes, batch_size):
            batch_prefix = prefix[i:i + batch_size]  # [B, H]

            # Expand dimensions for broadcasting
            # batch_prefix: [B, 1, H]
            # cache: [1, C, H]
            # comparison: [B, C, H]
            comparison = batch_prefix.unsqueeze(1) == self._prefix_valid_cache.unsqueeze(0)

            # Check if all elements in hierarchy match for any cached prefix
            # all_match: [B, C]
            all_match = comparison.all(dim=2)

            # Check if any cached prefix matches
            # any_match: [B]
            any_match = all_match.any(dim=1)

            results.append(any_match)

        return torch.cat(results)
