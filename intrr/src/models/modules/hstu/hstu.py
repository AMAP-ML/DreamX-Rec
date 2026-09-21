import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules.hstu.hstuconfig import HSTUConfig


class PointwiseAggregatedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # TODO: add relative attention bias based on time
        self.rab_p = RelativeAttentionBias(num_heads, relative_attention_num_buckets=32,
                                           relative_attention_max_distance=128)

    def split_heads(self, x, batch_size):
        # x shape: [batch_size, seq_len, d_model]
        # seq_len = x.size(1)
        # x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        seq_len = x.size(1)
        d_model = x.size(2)
        assert d_model == self.num_heads * self.head_dim, \
            f"d_model ({d_model}) != num_heads ({self.num_heads}) * head_dim ({self.head_dim})"

        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]

    def forward(self, v, k, q, mask=None):
        batch_size = q.shape[0]
        seq_len = q.shape[1]

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Apply scaling factor for numerical stability
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add relative attention bias
        rab = self.rab_p(q.shape[2], k.shape[2], device=q.device)
        att_w_bias = attention_scores + rab

        # Create causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # Combine with padding mask if provided
        if mask is not None:
            # mask shape: [batch, seq_len], 1 for valid, 0 for padding
            # Expand to [batch, 1, 1, seq_len] for broadcasting
            padding_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            # Combine causal and padding masks
            key_mask = mask[:, None, None, :].bool()  # [B,1,1,S]
            query_mask = mask[:, None, :, None].bool()  # [B,1,S,1]
            combined_mask = causal_mask & key_mask & query_mask
        else:
            combined_mask = causal_mask

        # Apply softplus activation BEFORE masking (as per HSTU paper)
        att_weights = F.softplus(att_w_bias).masked_fill(~combined_mask, 0.)
        # Fix dimension mismatch: compute normalization factor correctly
        # For HSTU, we normalize over the key dimension (last dimension of att_weights)
        # att_weights shape: [batch_size, num_heads, seq_len, seq_len]
        # We sum over the last dimension (keys) for each query position
        # Result should have shape: [batch_size, num_heads, seq_len, 1]
        att_weights_sum = att_weights.sum(dim=-1, keepdim=True) + 1e-8  # [batch_size, num_heads, seq_len, 1]
        att_weights = att_weights / att_weights_sum
        av = att_weights @ v
        return av.transpose(1, 2).flatten(2)


class RelativeAttentionBias(nn.Module):
    def __init__(self, num_heads, relative_attention_num_buckets, relative_attention_max_distance=128):
        super().__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)

    def forward(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=False,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    # https://github.com/huggingface/transformers/blob/6cdbd73e01a9719bfaec07d91fd108e8d932bbbb/src/transformers/models/t5/modeling_t5.py#L384
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets


class HSTUBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)  # Pre-Norm: LayerNorm before transformation
        self.f1 = nn.Linear(d_model, d_model * 4)  # Transform and split
        self.pointwise_attn = PointwiseAggregatedAttention(d_model, num_heads)
        self.f2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)

        x_proj = self.f1(x)
        u, v, q, k = x_proj.chunk(4, dim=-1)
        u = F.silu(u)

        # Spatial Aggregation: [batch, seq_len, d_model]
        av = self.pointwise_attn(v, k, q, mask=mask)

        # Pointwise Transformation: [batch, seq_len, d_model]
        y = self.f2(av * u)
        y = self.dropout(y)

        # Residual connection
        out = residual + y
        return out


class HSTURec(nn.Module):
    def __init__(self, config=None, d_model=None, num_heads=None, num_layers=None, dropout=0.1):
        """
        HSTU Recommendation Model

        Args:
            config: HSTUConfig object (optional, takes precedence over individual args)
            d_model: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of HSTU blocks
            dropout: Dropout rate
            vocab_size: Vocabulary size (optional, for embedding layer)
        """
        super().__init__()

        # Support both config object and individual parameters
        if config is not None:
            self.d_model = config.d_model
            self.num_heads = config.num_heads
            self.num_layers = config.num_layers
            self.dropout = getattr(config, 'dropout', dropout)
        else:
            if d_model is None or num_heads is None or num_layers is None:
                raise ValueError("Either config or (d_model, num_heads, num_layers) must be provided")
            self.d_model = d_model
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.dropout = dropout

        # HSTU blocks
        self.layers = nn.ModuleList([HSTUBlock(self.d_model, self.num_heads, self.dropout) for _ in range(self.num_layers)])


    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor, shape [batch, seq_len] if embedding layer exists,
               or [batch, seq_len, d_model] if no embedding layer
            mask: Attention mask, shape [batch, seq_len]

        Returns:
            Output tensor, shape [batch, seq_len, vocab_size] if output_layer exists,
            or [batch, seq_len, d_model] otherwise
        """
        # Apply HSTU blocks (residual connection is inside each block)
        for layer in self.layers:
            x = layer(x, mask=mask)


        return x


if __name__ == "__main__":
    hstuconfig = HSTUConfig(d_model=32, num_heads=2, num_layers=3)
    model = HSTURec(config=hstuconfig)
    input_shape = (2, 5, 32)
    mask_shape = (2, 5)
    # 生成标准的mask 掩码，用于序列推荐的 1表示有效，0 表示无效，1都在0前面 例如 11100
    # 方法1：手动指定每个序列的有效长度
    seq_lengths = torch.tensor([3, 4])  # 第一个序列有3个有效位置，第二个序列有4个有效位置
    mask = torch.zeros(*mask_shape)
    for i, length in enumerate(seq_lengths):
        mask[i, :length] = 1

    print("Mask shape:", mask.shape)
    print("Mask:\n", mask)
    x = torch.rand(*input_shape)

    print(model(x, mask=mask).shape)
    print(model(x, mask))