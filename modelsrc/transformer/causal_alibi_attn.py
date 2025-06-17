import torch
from torch import nn, Tensor, FloatTensor
import math

# Attention layer with ALiBi relative positional encoding
# TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION
# https://arxiv.org/pdf/2108.12409.pdf


def alibi_biases(query_len: int, key_len: int, device, dtype):
    x = torch.arange(key_len, device=device, dtype=dtype)[None, :]
    y = torch.arange(query_len, device=device, dtype=dtype)[:, None]
    return x - y


class CausalAlibiAttn(nn.Module):
    """
    A simple causal Alibi multi-head-attention implementation
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        bias: bool = True,
        dropout: float = 0.0,
        trainable_alibi: bool = False,
        alt_alibi_init: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.trainable_alibi = trainable_alibi
        self.alt_alibi_init = alt_alibi_init

        assert d_model % num_heads == 0, "d_model must be evenly divisible by num_heads"

        # The dimension of each head.
        self.d_head = d_model // num_heads

        # We scale the attention scores by the inverse-square-root of the head dimension
        # this shifts the temerature of softmax.
        self.dot_product_scale = 1.0 / math.sqrt(self.d_head)

        # Input projection matricies: K, K, V
        self.query_linear = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.key_linear = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.value_linear = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.output_linear = nn.Linear(self.d_model, self.d_model, bias=bias)

        if dropout == 0.0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

        if alt_alibi_init:
            # Observations with trainable slopes suggest that the high half of the slopes shift
            # towards / past 1.0 and the low half approach zero or even go slightly negative.
            # This init skews the high half higher and zeros the low-half.
            alibi_slopes = 1.0 / torch.logspace(
                1, 8, self.num_heads, base=2, dtype=torch.float
            )
            alibi_slopes.masked_fill_(
                torch.where(
                    torch.arange(0, self.num_heads) >= (self.num_heads / 2), True, False
                ),
                0,
            )
        else:
            # This generates the original slope distribution from the paper.
            alibi_slopes = 1.0 / torch.logspace(
                0, 7, self.num_heads, base=2, dtype=torch.float
            )
        self.alibi_slopes = nn.Parameter(alibi_slopes)
        self.alibi_slopes.requires_grad = trainable_alibi

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, trainable_alibi={self.trainable_alibi}, "
            f"alt_alibi_init={self.alt_alibi_init}"
        )

    def forward(self, qkv: FloatTensor) -> FloatTensor:
        # qkv: (batch_size, seq_len, d_qkv)
        batch_size, seq_len, d_qkv = qkv.shape

        # Feed the inputs through the K, Q, V matrices.
        query, key, value = (
            self.query_linear(qkv),
            self.key_linear(qkv),
            self.value_linear(qkv),
        )

        # Split projections into multiple heads and swap position of sequence / heads dimension
        query = query.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            1, 2
        )
        key = key.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            1, 2
        )

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.dot_product_scale

        # Apply Alibi biases
        scores += alibi_biases(
            seq_len, seq_len, device=scores.device, dtype=scores.dtype
        ) * self.alibi_slopes.view(-1, 1, 1)

        # Mask future positions from the past
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), True, device=qkv.device), diagonal=1
        )
        scores.masked_fill_(causal_mask, float("-inf"))
        del causal_mask

        # Calculate the attention weights; avoid NANs that might emerge from zeros in softmax's denominator
        attention_weights = self.dropout(torch.softmax(scores, dim=-1).clamp(min=1e-10))
        del scores

        # Use the attention weights to get a weighted combination of value vectors
        attended_values = torch.matmul(attention_weights, value)
        del attention_weights

        # Concatenate attention heads and project to original embedding size using the output linear layer
        attended_values = (
            attended_values.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, d_qkv)
        )

        # Project the concatenated output through the output matrix.
        return self.output_linear(attended_values)
