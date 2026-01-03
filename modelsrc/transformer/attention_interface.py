"""
A collection of attention functions, conforming to the HF Attention Interface,
which support ALiBi attention.

https://huggingface.co/docs/transformers/main/attention_interface

TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION
https://arxiv.org/pdf/2108.12409.pdf
"""

from typing import Any, Callable, Optional, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    FlexKernelOptions,
    flex_attention,
)
from torch.nn.functional import scaled_dot_product_attention

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


def _compiled_flex_attn(*args, **kwargs):
    return torch.compile(
        flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs"
    )(*args, **kwargs)


def alibi_biases(
    query_len: int, key_len: int, alibi_slopes: torch.Tensor, device, dtype
):
    """Generate ALiBi relative position biases.

    ALiBi applies linear biases to attention scores based on relative positions.
    This allows models to extrapolate to longer sequences than seen during training.

    Returns:
        Tensor of shape (query_len, key_len) with relative position differences
    """
    x = torch.arange(key_len, device=device, dtype=alibi_slopes.dtype)[None, :]
    y = torch.arange(query_len, device=device, dtype=alibi_slopes.dtype)[:, None]
    return (alibi_slopes.view(-1, 1, 1) * (x - y)).to(dtype=dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    # Additional args
    scaling: float,
    dropout: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    **kwargs,
):
    num_key_value_groups = query.shape[1] // key.shape[1]
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if alibi_slopes is not None:
        q_len = query.shape[-2]
        kv_len = key.shape[-2]
        attn_bias = alibi_biases(
            q_len, kv_len, alibi_slopes, device=query.device, dtype=query.dtype
        )
        attn_weights = attn_weights + attn_bias

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    # Additional args
    scaling: Optional[float] = None,
    kernel_options: Optional[FlexKernelOptions] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    compile_flex: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if kwargs.get("dropout", 0.0) != 0.0:
        raise ValueError("Flex attention does not support dropout")

    assert isinstance(
        attention_mask, BlockMask
    ), f"Expected BlockMask, found {type(attention_mask)}"

    if alibi_slopes is not None:
        assert score_mod is None, "Pass either alibi_slopes OR score_mod, but not both!"

        # See: https://pytorch.org/blog/flexattention/
        def alibi(score, b, h, q_idx, kv_idx):
            bias = alibi_slopes[h] * (kv_idx - q_idx)
            return score + bias

        score_mod = alibi

    num_key_value_groups = query.shape[1] // key.shape[1]
    flex_fn = _compiled_flex_attn if compile_flex else flex_attention
    attention_output = flex_fn(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=attention_mask,
        enable_gqa=(num_key_value_groups != 1),
        scale=scaling,
        kernel_options=kernel_options,
    )

    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, None


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    # Additional args
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    q_len = query.shape[-2]
    kv_len = key.shape[-2]

    if alibi_slopes is not None:
        # attn_bias.shape = (heads, q_len, kv_len)
        attn_bias = alibi_biases(
            q_len, kv_len, alibi_slopes, device=query.device, dtype=query.dtype
        )

        if attention_mask is None:
            if q_len > 1:
                causal_mask = torch.tril(
                    torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device)
                )
                attention_mask = torch.where(causal_mask, attn_bias, float("-inf"))
            else:
                attention_mask = attn_bias
        elif torch.is_floating_point(attention_mask):
            attention_mask = attention_mask + attn_bias
        else:
            attention_mask = torch.where(
                attention_mask, attn_bias.unsqueeze(0), float("-inf")
            )

    num_key_value_groups = query.shape[1] // key.shape[1]
    is_causal = q_len > 1 and attention_mask is None and is_causal

    attn_output = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        enable_gqa=(num_key_value_groups != 1),
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def flash_attn_2_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    # Additional args
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    config: Any = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    Basic Flash Attention 2 implementation with ALiBI support

    Limitations:
        - Does not support packed examples.
        - Does not support trainable ALiBI biases
        - Only supports bfloat16 for QKV
        - Only supports basic causal masking
    """
    assert (
        flash_attn_func is not None
    ), "Flash attn 2 is not installed. See: https://github.com/Dao-AILab/flash-attention/tree/main"

    window_size = getattr(config, "window_size", None)
    if window_size is None:
        window_size = (-1, -1)
    else:
        window_size = (window_size, 0)

    """
    Note that Flash Attn 2 expects:
        Q: (batch_size, seqlen, nheads, headdim)
        KV: (batch_size, seqlen, nheads_k, headdim)
        returns: (batch_size, seqlen, nheads, headdim)
    This does not match Attention Interface, thus we must transpose (1, 2) on input and output.

    ALiBI slopes are expected to be float32
    """

    attn_output = flash_attn_func(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
    )
    return attn_output.transpose(1, 2), None
