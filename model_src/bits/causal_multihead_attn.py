import torch
from torch import nn, Tensor, FloatTensor
import math


# A simple causal multi-head-attention implementation
class CausalMultiheadAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

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

    def extra_repr(self):
        return f"d_model={self.d_model}, num_heads={self.num_heads}"

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

        # Mask future positions from the past
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), True, device=qkv.device), diagonal=1
        )
        scores.masked_fill_(causal_mask, float("-inf"))

        # Calculate the attention weights; avoid NANs that might emerge from zeros in softmax's denominator
        attention_weights = self.dropout(torch.softmax(scores, dim=-1).clamp(min=1e-10))

        # Use the attention weights to get a weighted combination of value vectors
        attended_values = torch.matmul(attention_weights, value)

        # Concatenate attention heads and project to original embedding size using the output linear layer
        attended_values = (
            attended_values.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, d_qkv)
        )

        # Project the concatenated output through the output matrix.
        return self.output_linear(attended_values)
