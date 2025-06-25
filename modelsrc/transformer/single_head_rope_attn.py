import torch
from torch import nn, Tensor, FloatTensor
import math

from .rotary_embeddings import precompute_freqs_cis, apply_rotary_emb


class SingleHeadRoPEAttn(nn.Module):
    """
    Single-head causal attention with Rotary Position Embeddings (RoPE).
    
    This implementation embeds RoPE computation directly within the attention layer
    to ensure pipeline parallelism compatibility by avoiding cross-stage dependencies.
    """

    def __init__(
        self,
        d_model: int,
        *,
        max_seq_len: int = 2048,
        bias: bool = True,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.bias = bias

        # We scale the attention scores by the inverse-square-root of the head dimension
        self.dot_product_scale = 1.0 / math.sqrt(self.d_model)

        # For single-head attention, we can use a more efficient formulation
        # See: https://transformer-circuits.pub/2021/framework/index.html#splitting-attention-head-terms-into-circuits
        self.query_linear = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.key_linear = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.value_linear = nn.Linear(self.d_model, self.d_model, bias=self.bias)

        # RoPE frequencies - precomputed and stored as persistent buffer
        freqs_cis = precompute_freqs_cis(self.d_model, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=True)

    def extra_repr(self):
        return f"d_model={self.d_model}, max_seq_len={self.max_seq_len}, bias={self.bias}"

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # Compute Q, K, V projections
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)

        # Reshape for RoPE application (add head dimension)
        # Shape: (batch_size, seq_len, 1, d_model)
        query = query.unsqueeze(2)
        key = key.unsqueeze(2)

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

        # Remove the head dimension and compute attention scores
        # Shape: (batch_size, seq_len, d_model)
        query = query.squeeze(2)
        key = key.squeeze(2)

        # Compute scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.dot_product_scale

        # Mask future positions from the past
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), True, device=x.device), diagonal=1
        )
        scores.masked_fill_(causal_mask, float("-inf"))

        # Calculate the attention weights; avoid NANs that might emerge from zeros in softmax's denominator
        attention_weights = torch.softmax(scores, dim=-1).clamp(min=1e-10)

        # Use the attention weights to get a weighted combination of value vectors
        attended_values = torch.matmul(attention_weights, value)
        return attended_values