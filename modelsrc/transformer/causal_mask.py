from typing import Optional

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)


def causal_mask(
    config: PretrainedConfig,
    dtype: torch.dtype,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    input_embeds: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.Tensor] = None,
):
    """
    Create an attention mask fn for the model

    This method is designed to be called externally (e.g., from pipeline parallel code)
    to pre-compute attention masks that can be passed as extra_kwargs.

    Args:
        config: The model's PretrainedConfig
        dtype: Model's default dtype
        input_ids: Input token IDs (batch_size, seq_length)
        attention_mask: Optional 2D padding mask (batch_size, seq_length)
        position_ids: Optional position indices (batch_size, seq_length)

    Returns:
        The attention mask in the format required by the model's attention implementation
        (e.g., 4D tensor for eager/sdpa, BlockMask for flex_attention)
    """
    assert config is not None

    window_size = getattr(config, "window_size", None)

    # When using SDPA, if just simple a simple causal attention mask
    # is required, bypass mask generation. SDPA will then use
    # the "is_causal" flag, which saves memory and is faster.
    if (
        config._attn_implementation == "sdpa"
        and not window_size
        and attention_mask is None
        and past_key_values is None
        and position_ids is None
    ):
        return None

    if input_embeds is None:
        # Create a dummy input_embeds tensor for shape inference
        # We only need batch_size and dtype, not actual embeddings
        assert input_ids is not None
        batch_size, seq_length = input_ids.shape

        input_embeds = torch.empty(
            batch_size,
            seq_length,
            config.hidden_size,
            device=torch.device("meta"),
            dtype=dtype,
        )

    # Convert to bool mask, if long
    if isinstance(attention_mask, torch.Tensor) and attention_mask.dtype == torch.long:
        attention_mask = attention_mask.to(dtype=torch.bool)

    if cache_position is None:
        if input_ids is None:
            device = input_embeds.device
            seq_length = input_embeds.shape[1]
        else:
            device = input_ids.device
            seq_length = input_ids.shape[1]

        cache_position = torch.arange(0, seq_length, device=device)

    # Use HuggingFace's create_causal_mask utility
    assert cache_position is not None
    mask_fn = create_sliding_window_causal_mask if window_size else create_causal_mask
    attention_mask = mask_fn(
        config=config,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    # debug
    # print(repr(attention_mask))
    # print(attention_mask.shape)

    return attention_mask
