import math
from typing import Any, Optional

import torch
from torch import FloatTensor, nn
from torch.nn.functional import scaled_dot_product_attention

# Attention layer with ALiBi relative positional encoding
# TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION
# https://arxiv.org/pdf/2108.12409.pdf


def alibi_biases(
    query_len: int, key_len: int, alibi_slopes: torch.Tensor, device, dtype
):
    """Generate ALiBi relative position biases.

    ALiBi applies linear biases to attention scores based on relative positions.
    This allows models to extrapolate to longer sequences than seen during training.

    Returns:
        Tensor of shape (query_len, key_len) with relative position differences
    """
    x = torch.arange(key_len, device=device, dtype=dtype)[None, :]
    y = torch.arange(query_len, device=device, dtype=dtype)[:, None]
    return alibi_slopes.view(-1, 1, 1) * (x - y)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    # Additional args
    scaling: float,
    dropout: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    **kwargs,
):
    num_key_value_groups = query.shape[1] // key.shape[1]
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if alibi_slopes is not None:
        q_len = query.shape[-2]
        kv_len = key.shape[-2]
        attn_bias = alibi_biases(
            q_len, kv_len, alibi_slopes, device=query.device, dtype=query.dtype
        )
        attn_weights = attn_weights + attn_bias

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


attn_functions = {
    "eager": eager_attention_forward,
}


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
        num_kv_heads: Optional[int] = None,  # GQA support
        # attn_functions: Optional[dict[str, Callable]],
        config: Any = None,
        bias: bool = True,
        dropout: float = 0.0,
        trainable_alibi: bool = False,
        alt_alibi_init: bool = False,
        layer_idx: int,
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

        self.attn_fn = attn_functions[attn_implementation]

        assert d_model % num_heads == 0, "d_model must be evenly divisible by num_heads"
        assert (
            num_heads % self.num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"

        self.d_head = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

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

        # Initialize ALiBi slopes - one slope per attention head
        if alt_alibi_init:
            # Alternative initialization: high half slopes shift towards 1.0+, low half approach zero
            # This can work better with trainable slopes in some cases
            alibi_slopes = 1.0 / torch.logspace(
                1,
                8,
                self.num_heads,
                base=2,
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
                0, 7, self.num_heads, base=2, dtype=torch.float
            )

        self.alibi_slopes = nn.Parameter(alibi_slopes)
        self.alibi_slopes.requires_grad = trainable_alibi

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"trainable_alibi={self.trainable_alibi}, alt_alibi_init={self.alt_alibi_init}"
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
            **kwargs,
        )

        # Note: Attention function implicitly performs attended_values.transpose(1, 2)
        attended_values = attended_values.reshape(batch_size, seq_len, -1).contiguous()

        # Final output projection
        return self.output_linear(attended_values)
