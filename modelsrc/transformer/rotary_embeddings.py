import math
from typing import Tuple

import torch
from torch import Tensor


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    
    This function calculates the rotary position embedding frequencies used in RoPE.
    
    Args:
        dim: Dimension of the embedding (typically d_head)
        end: Maximum sequence length
        theta: Base for the geometric progression (default: 10000.0)
        
    Returns:
        Tensor of shape (end, dim//2) containing complex exponentials
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
    """
    Reshape frequency tensor to be broadcastable with input tensor.
    
    Args:
        freqs_cis: Frequency tensor of shape (seq_len, dim//2)
        x: Input tensor of shape (batch_size, seq_len, num_heads, dim)
        
    Returns:
        Reshaped freqs_cis tensor for broadcasting
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    q: Tensor, k: Tensor, freqs_cis: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch_size, seq_len, num_heads, d_head)
        k: Key tensor of shape (batch_size, seq_len, num_heads, d_head)
        freqs_cis: Frequency tensor of shape (seq_len, d_head//2)
        
    Returns:
        Tuple of (rotated_q, rotated_k) tensors with same shapes as input
    """
    # Reshape q and k to complex representation
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    
    # Reshape freqs_cis for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, q_complex)
    
    # Apply rotation
    q_out = torch.view_as_real(q_complex * freqs_cis).flatten(3)
    k_out = torch.view_as_real(k_complex * freqs_cis).flatten(3)
    
    return q_out.type_as(q), k_out.type_as(k)