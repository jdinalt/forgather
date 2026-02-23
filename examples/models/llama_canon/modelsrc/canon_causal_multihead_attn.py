"""
Canon Causal Multi-Head Attention.

Extends standard causal multi-head attention with Canon-B: a depthwise
causal convolution applied to concatenated Q, K, V projections before
attention computation.

From "Physics of Language Models: Part 4.1, Architecture Design and the
Magic of Canon Layers" (Allen-Zhu, 2025).
"""

import math
from typing import Any, Callable, Optional

import torch
from torch import FloatTensor, nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class _CanonLayer(nn.Module):
    """Depthwise causal 1D convolution with optional residual connection."""

    def __init__(self, dim: int, kernel_size: int = 4, residual: bool = True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.residual = residual
        self.conv = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=kernel_size,
            groups=dim, bias=False, padding=kernel_size - 1,
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x_conv = x.transpose(1, 2)
        out = self.conv(x_conv)
        out = out[..., : x.shape[1]]
        out = out.transpose(1, 2)
        if self.residual:
            return x + out
        return out


class CanonCausalMultiheadAttn(nn.Module):
    """Causal multi-head attention with Canon-B on QKV projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        attn_implementation: str,
        attn_functions: Optional[dict[str, Callable]] = None,
        config: Any = None,
        num_kv_heads: Optional[int] = None,
        pos_encoder: Optional[Callable] = None,
        bias: bool = False,
        dropout: float = 0.0,
        qk_norm_factory: Optional[Callable] = None,
        layer_idx: int,
        sliding_window: Optional[int] = None,
        canon_kernel: int = 4,
        canon_residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.pos_encoder = pos_encoder

        if attn_implementation == "flash_attention_2":
            self.is_causal = True
        self.config = config
        self.sliding_window = sliding_window
        self.layer_idx = layer_idx
        self.attn_implementation = attn_implementation

        if attn_functions and attn_implementation in attn_functions:
            self.attn_fn = attn_functions[attn_implementation]
        else:
            self.attn_fn = ALL_ATTENTION_FUNCTIONS[attn_implementation]

        assert d_model % num_heads == 0, "d_model must be evenly divisible by num_heads"
        assert (
            num_heads % self.num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"

        self.d_head = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.query_linear = nn.Linear(d_model, self.num_heads * self.d_head, bias=bias)
        setattr(self.query_linear, "init_prefix", "attn.query")

        self.key_linear = nn.Linear(d_model, self.num_kv_heads * self.d_head, bias=bias)
        setattr(self.key_linear, "init_prefix", "attn.key")

        self.value_linear = nn.Linear(d_model, self.num_kv_heads * self.d_head, bias=bias)
        setattr(self.value_linear, "init_prefix", "attn.value")

        self.output_linear = nn.Linear(self.num_heads * self.d_head, d_model, bias=bias)
        setattr(self.output_linear, "init_prefix", "attn.output")

        self.dropout_p = dropout

        if qk_norm_factory:
            self.q_norm = qk_norm_factory(normalized_shape=self.d_head)
            self.k_norm = qk_norm_factory(normalized_shape=self.d_head)
        else:
            self.q_norm = None
            self.k_norm = None

        # Canon-B: applied to concatenated Q, K, V
        total_qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.d_head
        self.canon_b = _CanonLayer(
            dim=total_qkv_dim, kernel_size=canon_kernel, residual=canon_residual,
        )

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
        hidden_states: FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> FloatTensor:
        batch_size, seq_len, d_model = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.d_head)

        query = self.query_linear(hidden_states)
        key = self.key_linear(hidden_states)
        value = self.value_linear(hidden_states)

        # Canon-B: convolve concatenated QKV
        q_dim = self.num_heads * self.d_head
        k_dim = self.num_kv_heads * self.d_head
        v_dim = self.num_kv_heads * self.d_head
        qkv = torch.cat([query, key, value], dim=-1)
        qkv = self.canon_b(qkv)
        query, key, value = qkv.split([q_dim, k_dim, v_dim], dim=-1)

        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        if self.q_norm:
            query = self.q_norm(query)
        if self.k_norm:
            key = self.k_norm(key)

        if self.pos_encoder:
            query, key = self.pos_encoder(query, key, position_ids)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key, value = past_key_values.update(
                key, value, self.layer_idx, cache_kwargs
            )

        attended_values, attn_weights = self.attn_fn(
            module=self,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            dropout=(self.dropout_p if self.training else 0.0),
            scaling=self.scale,
            config=self.config,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attended_values = attended_values.reshape(batch_size, seq_len, -1).contiguous()
        return self.output_linear(attended_values)
