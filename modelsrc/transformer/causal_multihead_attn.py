import torch
from torch import nn, FloatTensor
from typing import Callable, Optional
import math


class CausalMultiheadAttn(nn.Module):
    """
    Causal multi-head attention with (optional) Relative Positional Embeddings (RPE).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        attn_implementation: str,
        attn_functions: Optional[dict[str, Callable]],
        num_kv_heads: Optional[int] = None,
        pos_encoder: Optional[Callable] = None,
        bias: bool = False,
        dropout: float = 0.0,
        qk_norm_factory: Optional[Callable] = None,
    ):
        """
        args:
            d_model: Hidden dimension
            num_heads: QKV heads, if not num_kv_heads, else Q heads
            attn_functions: A map of attention interfaces, conforming to
                https://huggingface.co/docs/transformers/main/attention_interface
            attn_implementation: A name in "attention_interfaces"
            num_kv_heads: GQA Support
            pos_encoder: A relative positional encoder
            bias: Apply bias to Q, K, V, and O
            dropout: Attention dropout -- not compatible with flex-attention!
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads  # Default to MHA
        self.pos_encoder = pos_encoder

        self.attn_implementation = attn_implementation

        if attn_functions and attn_implementation in attn_functions:
            self.attn_fn = attn_functions[attn_implementation]
        else:
            raise ValueError(
                f"attn_implementation implementation {attn_implementation} is unsupported"
            )

        assert d_model % num_heads == 0, "d_model must be evenly divisible by num_heads"
        assert (
            num_heads % self.num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"

        self.d_head = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Query projections for all heads
        self.query_linear = nn.Linear(d_model, d_model, bias=bias)

        # Key/Value projections for KV heads (potentially fewer than query heads)
        self.key_linear = nn.Linear(d_model, self.num_kv_heads * self.d_head, bias=bias)
        self.value_linear = nn.Linear(
            d_model, self.num_kv_heads * self.d_head, bias=bias
        )

        # Output projection
        self.output_linear = nn.Linear(d_model, d_model, bias=bias)

        # Store dropout probability for SDPA function
        self.dropout_p = dropout

        # QK normalization (Qwen3-style: per-head normalization over d_head)
        # Applied after reshaping to [batch, seq, num_heads, d_head]
        if qk_norm_factory:
            self.q_norm = qk_norm_factory(normalized_shape=self.d_head)
            self.k_norm = qk_norm_factory(normalized_shape=self.d_head)
        else:
            self.q_norm = None
            self.k_norm = None

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"attn_implementation={self.attn_implementation}, "
            f"dropout_p={self.dropout_p}"
        )

    def forward(
        self,
        qkv: FloatTensor,
        layer_index: int,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["DynamicCache"] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> FloatTensor:
        batch_size, seq_len, d_model = qkv.shape

        # Project to Q, K, V
        query = self.query_linear(qkv)  # [batch, seq_len, d_model]
        key = self.key_linear(qkv)  # [batch, seq_len, num_kv_heads * d_head]
        value = self.value_linear(qkv)  # [batch, seq_len, num_kv_heads * d_head]

        # Reshape for multi-head attention (before applying normalization and position embeddings)
        # Query: [batch, seq_len, num_heads, d_head]
        query = query.view(batch_size, seq_len, self.num_heads, self.d_head)

        # Key/Value: [batch, seq_len, num_kv_heads, d_head]
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.d_head)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.d_head)

        # Apply QK normalization per-head (Qwen3-style)
        # Normalizes over the last dimension (d_head) for each head independently
        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        # Apply relative positional embeddings to query and key tensors
        if self.pos_encoder:
            query, key = self.pos_encoder(query, key, position_ids)

        # Transpose to [batch, heads, seq_len, d_head], for attention function
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply KV Cache
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, layer_index)
        attended_values, attn_weights = self.attn_fn(
            module=self,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            dropout=(self.dropout_p if self.training else 0.0),
            scaling=self.scale,
            **kwargs,
        )

        # Note: Attention function implicitly performs attended_values.transpose(1, 2)
        attended_values = attended_values.reshape(
            batch_size, seq_len, d_model
        ).contiguous()

        # Final output projection
        return self.output_linear(attended_values)
