import math
from typing import Callable, Optional

import torch
from torch import FloatTensor, nn
from torch.nn.functional import scaled_dot_product_attention

# Attention layer with ALiBi relative positional encoding
# TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION
# https://arxiv.org/pdf/2108.12409.pdf


def alibi_biases(query_len: int, key_len: int, device, dtype):
    """Generate ALiBi relative position biases.

    ALiBi applies linear biases to attention scores based on relative positions.
    This allows models to extrapolate to longer sequences than seen during training.

    Returns:
        Tensor of shape (query_len, key_len) with relative position differences
    """
    x = torch.arange(key_len, device=device, dtype=dtype)[None, :]
    y = torch.arange(query_len, device=device, dtype=dtype)[:, None]
    return x - y


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
        num_kv_heads: Optional[int] = None,  # GQA support
        sdpa_function: Callable = scaled_dot_product_attention,
        bias: bool = True,
        dropout: float = 0.0,
        trainable_alibi: bool = False,
        alt_alibi_init: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads  # Default to MHA
        self.sdpa_function = sdpa_function
        self.trainable_alibi = trainable_alibi
        self.alt_alibi_init = alt_alibi_init

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

    def forward(self, qkv: FloatTensor, **kwargs) -> FloatTensor:
        batch_size, seq_len, d_model = qkv.shape

        # Project to Q, K, V
        query = self.query_linear(qkv)  # [batch, seq_len, d_model]
        key = self.key_linear(qkv)  # [batch, seq_len, num_kv_heads * d_head]
        value = self.value_linear(qkv)  # [batch, seq_len, num_kv_heads * d_head]

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

        # Create ALiBi attention mask
        # ALiBi applies linear biases based on relative positions between tokens
        attn_bias = alibi_biases(
            seq_len, seq_len, device=query.device, dtype=query.dtype
        )
        # Scale biases by learnable slopes (one per head)
        attn_bias = attn_bias * self.alibi_slopes.view(-1, 1, 1)

        # Apply causal mask to ALiBi biases
        # Only allow attention to current and previous positions
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device)
        )
        attn_bias.masked_fill_(causal_mask.logical_not(), float("-inf"))

        # Apply scaled dot product attention with ALiBi biases as attention mask
        # Use enable_gqa=True to let PyTorch handle GQA automatically
        # Set is_causal=False since we handle causal masking in the ALiBi bias
        attended_values = self.sdpa_function(
            query,
            key,
            value,
            attn_mask=attn_bias,
            dropout_p=(self.dropout_p if self.training else 0.0),
            is_causal=False,  # We handle causal masking in ALiBi bias
            scale=self.scale,
            enable_gqa=(self.num_kv_heads < self.num_heads),
        )

        # Reshape back to [batch, seq_len, d_model]
        attended_values = attended_values.transpose(1, 2).reshape(
            batch_size, seq_len, d_model
        )

        # Final output projection
        return self.output_linear(attended_values)
