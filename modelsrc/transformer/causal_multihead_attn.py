"""
Reference implementation of causal multi-head attention.

This is a clean, educational implementation that demonstrates the core concepts
of multi-head attention for transformer models. It uses an injectable scaled
dot product attention function to handle the actual attention computation.

The implementation is designed to be readable and understandable, making it
ideal for learning how attention mechanisms work.
"""

import torch
from torch import nn, Tensor, FloatTensor
from torch.nn.functional import scaled_dot_product_attention
import math
from typing import Callable


class CausalMultiheadAttn(nn.Module):
    """
    Causal multi-head attention for transformer language models.

    This implementation follows the standard transformer attention mechanism:
    1. Project input to Query, Key, Value matrices
    2. Split into multiple attention heads
    3. Apply scaled dot product attention with causal masking
    4. Concatenate heads and project to output

    The actual attention computation is handled by an injectable SDPA function,
    allowing for different attention implementations while keeping the core
    structure simple and educational.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        sdpa_function: Callable = scaled_dot_product_attention,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize causal multi-head attention.

        Args:
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads
            sdpa_function: Scaled dot product attention function to use
            bias: Whether to use bias in linear projections
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.sdpa_function = sdpa_function

        # Ensure the model dimension can be evenly split across heads
        assert d_model % num_heads == 0, "d_model must be evenly divisible by num_heads"

        # Calculate the dimension of each attention head
        self.d_head = d_model // num_heads

        # Scaling factor for attention scores (from "Attention Is All You Need")
        # This prevents the dot products from growing too large and making softmax saturate
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Linear projections for Query, Key, and Value
        # These transform the input embeddings into query, key, and value representations
        self.query_linear = nn.Linear(d_model, d_model, bias=bias)
        self.key_linear = nn.Linear(d_model, d_model, bias=bias)
        self.value_linear = nn.Linear(d_model, d_model, bias=bias)

        # Output projection to combine the attention heads back to the original dimension
        self.output_linear = nn.Linear(d_model, d_model, bias=bias)

        # Store dropout probability for SDPA function
        self.dropout_p = dropout

    def extra_repr(self) -> str:
        """String representation for debugging and logging."""
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, d_head={self.d_head}"
        )

    def forward(self, qkv: FloatTensor, **kwargs) -> FloatTensor:
        """
        Apply causal multi-head attention to input sequence.

        Args:
            qkv: Input tensor of shape (batch_size, seq_len, d_model)
                 This serves as Query, Key, and Value (self-attention)
            **kwargs: Additional arguments passed through to SDPA function

        Returns:
            Attended output tensor of shape (batch_size, seq_len, d_model)

        Process:
            1. Project input to Q, K, V matrices
            2. Reshape for multi-head attention
            3. Apply scaled dot product attention with causal masking
            4. Concatenate heads and apply output projection
        """
        batch_size, seq_len, d_model = qkv.shape

        # Step 1: Project input through Query, Key, Value linear layers
        # In self-attention, the same input serves as source for all three
        query = self.query_linear(qkv)  # Shape: (batch_size, seq_len, d_model)
        key = self.key_linear(qkv)  # Shape: (batch_size, seq_len, d_model)
        value = self.value_linear(qkv)  # Shape: (batch_size, seq_len, d_model)

        # Step 2: Reshape projections for multi-head attention
        # Split the d_model dimension into num_heads * d_head, then move heads to dimension 1
        # This allows each head to attend independently to different representation subspaces
        query = query.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            1, 2
        )
        key = key.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            1, 2
        )
        # After reshape and transpose: (batch_size, num_heads, seq_len, d_head)

        # Step 3: Apply scaled dot product attention
        # Use the injectable SDPA function to compute attention
        # is_causal=True ensures positions can only attend to previous positions (causal masking)

        # Apply scaled dot product attention with causal masking
        # Ignore any provided attention mask and always use causal masking
        # Use conditional dropout based on training mode
        attended_values = self.sdpa_function(
            query,
            key,
            value,
            dropout_p=(self.dropout_p if self.training else 0.0),
            is_causal=True,  # Always use causal masking, ignore any provided mask
            scale=self.scale,
        )
        # Output shape: (batch_size, num_heads, seq_len, d_head)

        # Step 4: Concatenate attention heads and apply output projection
        # Move heads dimension back and reshape to concatenate all heads
        attended_values = attended_values.transpose(
            1, 2
        ).reshape(  # Move seq_len back to dimension 1
            batch_size, seq_len, d_model
        )  # Concatenate all heads

        # Apply final linear transformation to produce output
        output = self.output_linear(attended_values)

        return output
