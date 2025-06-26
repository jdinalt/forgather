import torch
from torch import nn, Tensor, FloatTensor
from torch.nn.functional import scaled_dot_product_attention
from typing import Callable, Optional
import math

from .rotary_embeddings import apply_rotary_emb


class CausalRpeAttn(nn.Module):
    """
    Causal multi-head attention with Relative Positional Embeddings (RPE).
    
    This implementation supports various relative positional encoding methods through
    an injectable function interface:
    - RoPE (Rotary Position Embedding): Encodes position by rotating Q/K vectors
    - T5-style learnable relative position biases: Learned embeddings for relative distances
    - Other custom relative position encoding schemes
    
    Uses an injectable SDPA function and supports Grouped Query Attention (GQA).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        num_kv_heads: Optional[int] = None,  # GQA support
        apply_pos_emb: Callable,
        sdpa_function: Callable = scaled_dot_product_attention,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads  # Default to MHA
        self.sdpa_function = sdpa_function
        
        assert d_model % num_heads == 0, "d_model must be evenly divisible by num_heads"
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_head = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Query projections for all heads
        self.query_linear = nn.Linear(d_model, d_model, bias=bias)
        
        # Key/Value projections for KV heads (potentially fewer than query heads)
        self.key_linear = nn.Linear(d_model, self.num_kv_heads * self.d_head, bias=bias)
        self.value_linear = nn.Linear(d_model, self.num_kv_heads * self.d_head, bias=bias)
        
        # Output projection
        self.output_linear = nn.Linear(d_model, d_model, bias=bias)

        # Store dropout probability for SDPA function
        self.dropout_p = dropout
        
        # Generic relative positional embedding application function
        self.apply_pos_emb = apply_pos_emb

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}"
        )

    def forward(self, qkv: FloatTensor, pos_emb, **kwargs) -> FloatTensor:
        batch_size, seq_len, d_model = qkv.shape
        
        # Project to Q, K, V
        query = self.query_linear(qkv)  # [batch, seq_len, d_model]
        key = self.key_linear(qkv)      # [batch, seq_len, num_kv_heads * d_head]
        value = self.value_linear(qkv)  # [batch, seq_len, num_kv_heads * d_head]
        
        # Reshape for multi-head attention (before applying relative position embeddings)
        # Query: [batch, seq_len, num_heads, d_head]
        query = query.view(batch_size, seq_len, self.num_heads, self.d_head)
        
        # Key/Value: [batch, seq_len, num_kv_heads, d_head]
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.d_head)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.d_head)
        
        # Apply relative positional embeddings to query and key tensors
        # This is a generic interface that supports different RPE methods:
        # - RoPE: Rotates Q/K vectors based on absolute positions
        # - T5-style: Could add learned relative position biases
        query_with_pos, key_with_pos = self.apply_pos_emb(query, key, pos_emb)
        
        # Transpose to [batch, heads, seq_len, d_head] for SDPA
        query = query_with_pos.transpose(1, 2)
        key = key_with_pos.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Apply scaled dot product attention with causal masking
        # Use enable_gqa=True to let PyTorch handle GQA automatically
        attended_values = self.sdpa_function(
            query, key, value,
            dropout_p=(self.dropout_p if self.training else 0.0),
            is_causal=True,
            scale=self.scale,
            enable_gqa=(self.num_kv_heads < self.num_heads)
        )
        
        # Reshape back to [batch, seq_len, d_model]
        attended_values = (
            attended_values.transpose(1, 2)
            .reshape(batch_size, seq_len, d_model)
        )
        
        # Final output projection
        return self.output_linear(attended_values)
