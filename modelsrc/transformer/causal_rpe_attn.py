import torch
from torch import nn, Tensor, FloatTensor
from typing import Callable
import math

from .rotary_embeddings import apply_rotary_emb


class CausalRpeAttn(nn.Module):
    """
    Causal multi-head attention with Relative Positional Embeddings.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        apply_pos_emb: Callable,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model must be evenly divisible by num_heads"

        # The dimension of each head
        self.d_head = d_model // num_heads

        # We scale the attention scores by the inverse-square-root of the head dimension
        self.dot_product_scale = 1.0 / math.sqrt(self.d_head)

        # Input projection matrices: Q, K, V
        self.query_linear = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.key_linear = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.value_linear = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.output_linear = nn.Linear(self.d_model, self.d_model, bias=bias)

        if dropout == 0.0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)
        self.apply_pos_emb = apply_pos_emb

    def extra_repr(self):
        return f"d_model={self.d_model}, num_heads={self.num_heads}"

    def forward(self, qkv: FloatTensor, pos_emb, **kwargs) -> FloatTensor:
        # qkv: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = qkv.shape

        # Feed the inputs through the Q, K, V matrices
        query, key, value = (
            self.query_linear(qkv),
            self.key_linear(qkv),
            self.value_linear(qkv),
        )

        # Split projections into multiple heads and swap position of sequence / heads dimension
        # Shape: (batch_size, seq_len, num_heads, d_head)
        query = query.view(batch_size, seq_len, self.num_heads, self.d_head)
        key = key.view(batch_size, seq_len, self.num_heads, self.d_head)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_head)

        # Apply positional embeddings to query and key tensors
        query, key = self.apply_pos_emb(query, key, pos_emb)

        # Transpose to put heads dimension before sequence for attention computation
        # Shape: (batch_size, num_heads, seq_len, d_head)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.dot_product_scale

        # Mask future positions from the past
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), True, device=qkv.device), diagonal=1
        )
        scores.masked_fill_(causal_mask, float("-inf"))

        # Calculate the attention weights
        attention_weights = self.dropout(torch.softmax(scores, dim=-1))

        # Use the attention weights to get a weighted combination of value vectors
        attended_values = torch.matmul(attention_weights, value)

        # Concatenate attention heads and project to original embedding size using the output linear layer
        attended_values = (
            attended_values.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, d_model)
        )

        # Project the concatenated output through the output matrix
        return self.output_linear(attended_values)
