"""
Reference implementation of scaled dot product attention.

This is a simple, readable implementation that follows the mathematical definition
of scaled dot product attention closely. It's intended as a reference for learning
and debugging, not for production performance.

The implementation follows the same signature as torch.nn.functional.scaled_dot_product_attention
for compatibility with injectable attention systems.
"""

import torch
from torch import Tensor
import math
from typing import Optional


def eager_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> Tensor:
    """
    Reference implementation of scaled dot product attention.

    This function computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    Compatible with torch.nn.functional.scaled_dot_product_attention signature.

    Args:
        query: Query tensor of shape (batch, num_heads_q, seq_len, head_dim)
        key: Key tensor of shape (batch, num_heads_kv, seq_len, head_dim)
        value: Value tensor of shape (batch, num_heads_kv, seq_len, head_dim)
        attn_mask: Optional attention mask. If boolean, True positions are kept.
                   If float, values are added to attention scores.
                   Cannot be used together with is_causal=True.
        dropout_p: Dropout probability applied to attention weights
        is_causal: If True, apply causal (lower triangular) mask.
                   Cannot be used together with attn_mask.
        scale: Scaling factor for attention scores. If None, uses 1/sqrt(head_dim)
        enable_gqa: If True, enables Grouped Query Attention (GQA) support.
                    Requires num_heads_q % num_heads_kv == 0.

    Returns:
        Attended values tensor of shape (batch, num_heads_q, seq_len, head_dim)

    Raises:
        ValueError: If both attn_mask and is_causal are provided
        ValueError: If GQA constraints are not met when enable_gqa=True

    Mathematical Description:
        1. Compute attention scores: scores = query @ key.transpose(-2, -1)
        2. Scale scores: scores = scores * scale
        3. Apply masks: scores = apply_masks(scores)
        4. Compute attention weights: weights = softmax(scores)
        5. Apply dropout: weights = dropout(weights)
        6. Compute output: output = weights @ value
    """
    # Validate mutual exclusivity of attn_mask and is_causal (matches PyTorch behavior)
    if attn_mask is not None and is_causal:
        raise ValueError("Cannot specify both attn_mask and is_causal=True")

    # Get dimensions for scaling and masking
    batch_size, num_heads_q, seq_len, head_dim = query.shape
    batch_size_k, num_heads_kv, seq_len_k, head_dim_k = key.shape
    batch_size_v, num_heads_kv_v, seq_len_v, head_dim_v = value.shape

    # Validate tensor shapes
    if seq_len != seq_len_k or seq_len != seq_len_v:
        raise ValueError(
            f"Sequence length mismatch: query={seq_len}, key={seq_len_k}, value={seq_len_v}"
        )
    if head_dim != head_dim_k or head_dim != head_dim_v:
        raise ValueError(
            f"Head dimension mismatch: query={head_dim}, key={head_dim_k}, value={head_dim_v}"
        )
    if num_heads_kv != num_heads_kv_v:
        raise ValueError(
            f"Key and value must have same number of heads: key={num_heads_kv}, value={num_heads_kv_v}"
        )

    # Validate GQA constraints
    if enable_gqa:
        if num_heads_q % num_heads_kv != 0:
            raise ValueError(
                f"For GQA, number of query heads ({num_heads_q}) must be divisible by number of key/value heads ({num_heads_kv})"
            )

    # Handle GQA by expanding key and value tensors to match query heads
    if num_heads_q != num_heads_kv:
        if not enable_gqa:
            raise ValueError(
                f"Number of heads mismatch: query={num_heads_q}, key/value={num_heads_kv}. Use enable_gqa=True for Grouped Query Attention."
            )

        # Expand key and value to match query heads
        expand_factor = num_heads_q // num_heads_kv
        key = key.repeat_interleave(expand_factor, dim=1)
        value = value.repeat_interleave(expand_factor, dim=1)

    # Step 1: Compute raw attention scores by multiplying queries and keys
    # This gives us how much each query position should attend to each key position
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 2: Scale the attention scores
    # Without scaling, the dot products can become very large, making softmax saturate
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    scores = scores * scale

    # Step 3a: Apply attention mask if provided
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # Boolean mask: True means "keep this position", False means "mask out"
            # We set masked positions to negative infinity so they become 0 after softmax
            scores.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            # Float mask: values are added to the attention scores
            # This allows for relative position biases, ALiBi, etc.
            scores = scores + attn_mask

    # Step 3b: Apply causal mask if requested
    if is_causal and seq_len > 1:
        # Causal mask prevents positions from attending to future positions
        # Create a lower triangular matrix: positions can only see themselves and the past
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device)
        )
        # Mask out the upper triangle (future positions)
        scores.masked_fill_(causal_mask.logical_not(), float("-inf"))

    # Step 4: Convert scores to attention weights using softmax
    # Softmax ensures all attention weights sum to 1 for each query position
    attention_weights = torch.softmax(scores, dim=-1)

    # Step 5: Apply dropout to attention weights
    # Always applies dropout according to dropout_p argument (matches PyTorch behavior)
    attention_weights = torch.nn.functional.dropout(attention_weights, p=dropout_p)

    # Step 6: Apply attention weights to values
    # This computes the weighted average of values based on attention weights
    attended_values = torch.matmul(attention_weights, value)

    return attended_values
