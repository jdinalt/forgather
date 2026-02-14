import math
from typing import Any, Callable, Optional

import torch
from torch import FloatTensor, nn

"""
Attention layer with ALiBi relative positional encoding
TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION
https://arxiv.org/pdf/2108.12409.pdf
"""


class CausalAlibiAttn(nn.Module):
    """
    Causal multi-head attention with ALiBi relative positional encoding.

    ALiBi applies linear biases to attention scores based on relative positions,
    allowing models to extrapolate to longer sequences than seen during training.
    Supports Grouped Query Attention (GQA) for improved efficiency.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        attn_implementation: str,
        attn_functions: dict[str, Callable],
        config: Any = None,
        num_kv_heads: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
        trainable_alibi: bool = False,
        alt_alibi_init: bool = False,
        layer_idx: int,
        sliding_window: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads  # Default to MHA
        self.trainable_alibi = trainable_alibi
        self.alt_alibi_init = alt_alibi_init
        self.layer_idx = layer_idx
        self.config = config
        self.attn_implementation = attn_implementation
        self.sliding_window = sliding_window

        assert attn_functions is not None, "A dict of attention functions is required"
        if attn_implementation not in attn_functions:
            raise ValueError(
                "ALiBi attention only supports the following types: "
                f"{attn_functions.keys()}"
            )
        self.attn_fn = attn_functions[attn_implementation]

        assert d_model % num_heads == 0, "d_model must be evenly divisible by num_heads"
        assert (
            num_heads % self.num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"

        self.d_head = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Query projections for all heads
        self.query_linear = nn.Linear(d_model, d_model, bias=bias)
        setattr(self.query_linear, "init_prefix", "attn.query")

        # Key/Value projections for KV heads (potentially fewer than query heads)
        self.key_linear = nn.Linear(d_model, self.num_kv_heads * self.d_head, bias=bias)
        setattr(self.key_linear, "init_prefix", "attn.key")

        self.value_linear = nn.Linear(
            d_model, self.num_kv_heads * self.d_head, bias=bias
        )
        setattr(self.value_linear, "init_prefix", "attn.value")

        # Output projection
        self.output_linear = nn.Linear(d_model, d_model, bias=bias)
        setattr(self.output_linear, "init_prefix", "attn.output")

        # Store dropout probability for SDPA function
        self.dropout_p = dropout

        self.alt_alibi_init = alt_alibi_init
        self.alibi_slopes = nn.Parameter(
            torch.empty((self.num_heads,), dtype=torch.float32)
        )
        self.alibi_slopes.requires_grad = trainable_alibi
        self.reset_parameters()

    def reset_parameters(self):
        device = self.alibi_slopes.device
        dtype = self.alibi_slopes.dtype

        # Initialize ALiBi slopes - one slope per attention head
        if self.alt_alibi_init:
            # Alternative initialization: high half slopes shift towards 1.0+, low half approach zero
            # This can work better with trainable slopes in some cases
            alibi_slopes = 1.0 / torch.logspace(
                1, 8, self.num_heads, base=2, dtype=dtype, device=device
            )
            # Zero out the lower half of slopes (position-agnostic heads)
            alibi_slopes.masked_fill_(
                torch.where(
                    torch.arange(0, self.num_heads) >= (self.num_heads / 2), True, False
                ),
                0,
            )
        else:
            # Original ALiBi slope distribution from the paper
            # Creates exponentially decreasing slopes: 1/2^0, 1/2^1, ..., 1/2^7
            alibi_slopes = 1.0 / torch.logspace(
                0, 7, self.num_heads, base=2, dtype=dtype, device=device
            )
        with torch.no_grad():
            self.alibi_slopes.copy_(alibi_slopes)

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"trainable_alibi={self.trainable_alibi}, alt_alibi_init={self.alt_alibi_init}, "
            f"attn_implementation={self.attn_implementation}, sliding_window={self.sliding_window}"
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

        # Project to Q, K, V
        query = self.query_linear(hidden_states)  # [batch, seq_len, d_model]
        key = self.key_linear(hidden_states)  # [batch, seq_len, num_kv_heads * d_head]
        value = self.value_linear(
            hidden_states
        )  # [batch, seq_len, num_kv_heads * d_head]

        # Reshape for multi-head attention
        # Query: [batch, seq_len, num_heads, d_head] -> [batch, num_heads, seq_len, d_head]
        query = query.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            1, 2
        )

        # Key/Value: [batch, seq_len, num_kv_heads, d_head] -> [batch, num_kv_heads, seq_len, d_head]
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.d_head).transpose(
            1, 2
        )
        value = value.view(
            batch_size, seq_len, self.num_kv_heads, self.d_head
        ).transpose(1, 2)

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
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # Note: Attention function implicitly performs attended_values.transpose(1, 2)
        attended_values = attended_values.reshape(batch_size, seq_len, -1).contiguous()

        # Final output projection
        return self.output_linear(attended_values)
