import torch
from torch import nn, Tensor, FloatTensor
import math

from .rotary_embeddings import precompute_freqs_cis, apply_rotary_emb


class CausalMultiheadRoPEAttn(nn.Module):
    """
    Causal multi-head attention with Rotary Position Embeddings (RoPE).
    
    This implementation embeds RoPE computation directly within the attention layer
    to ensure pipeline parallelism compatibility by avoiding cross-stage dependencies.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        max_seq_len: int = 2048,
        bias: bool = True,
        dropout: float = 0.0,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

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

        # RoPE frequencies - precomputed and stored as persistent buffer
        # This ensures the frequencies move with the module during pipeline parallelism
        freqs_cis = precompute_freqs_cis(self.d_head, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=True)

    def extra_repr(self):
        return f"d_model={self.d_model}, num_heads={self.num_heads}, max_seq_len={self.max_seq_len}"

    def forward(self, qkv: FloatTensor, **kwargs) -> FloatTensor:
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

        # Apply RoPE to query and key tensors
        # Use shared freqs_cis if provided, otherwise use local buffer
        freqs_cis = kwargs.get('freqs_cis', None)
        if freqs_cis is not None:
            # Use shared frequencies (memory efficient)
            freqs_cis = freqs_cis[:seq_len]
        else:
            # Fallback to local buffer (backward compatibility)
            freqs_cis = self.freqs_cis[:seq_len]
        query, key = apply_rotary_emb(query, key, freqs_cis)

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