"""
A collection of attention functions, conforming to the HF Attention Interface
https://huggingface.co/docs/transformers/main/attention_interface
"""
from typing import Callable, Optional, Union

import torch
from torch import nn, Tensor, FloatTensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    FlexKernelOptions,
)

from torch.nn.functional import scaled_dot_product_attention

def _compiled_flex_attn(*args, **kwargs):
    return torch.compile(flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs")(*args, **kwargs)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
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
    **kwargs,
):
    num_key_value_groups = query.shape[1] // key.shape[1]
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
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
    score_mod: Optional[Callable] = None,
    compile_flex: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if kwargs.get("dropout", 0.0) != 0.0:
        raise ValueError(
            "Flex attention does not support dropout"
        )

    assert isinstance(attention_mask, BlockMask)

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
    is_causal: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    num_key_value_groups = query.shape[1] // key.shape[1]
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
    
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