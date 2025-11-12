from typing import Optional
from transformers.cache_utils import Cache
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers import PretrainedConfig

import torch


def causal_mask(
    config: PretrainedConfig,
    dtype: torch.dtype,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    input_embeds: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
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
    assert config
    assert hasattr(config, "sliding_window")

    if input_embeds is None:
        # Create a dummy input_embeds tensor for shape inference
        # We only need batch_size and dtype, not actual embeddings
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
        cache_position = torch.arange(0, input_ids.shape[1], device=input_ids.device)

    # Use HuggingFace's create_causal_mask utility
    attention_mask = create_sliding_window_causal_mask(
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
