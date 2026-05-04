from typing import Any, Dict, Optional

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)


def gemma_mask_fn(
    config: PretrainedConfig,
    dtype: torch.dtype,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    input_embeds: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Build per-layer-type attention masks for Gemma-3.

    Returns a dict keyed by the values that appear in ``config.layer_types``:

    - ``"full_attention"``: a standard causal mask built by
      :func:`transformers.masking_utils.create_causal_mask`.
    - ``"sliding_attention"``: a sliding-window causal mask built by
      :func:`transformers.masking_utils.create_sliding_window_causal_mask`,
      which reads the window size from ``config.sliding_window``.

    :class:`GemmaDecoderLayer` unpacks this dict in its forward pass and
    selects the entry matching its own layer type.
    """
    assert config is not None

    if input_embeds is None:
        assert input_ids is not None
        batch_size, seq_length = input_ids.shape
        input_embeds = torch.empty(
            batch_size,
            seq_length,
            config.hidden_size,
            device=input_ids.device,
            dtype=dtype,
        )

    if isinstance(attention_mask, torch.Tensor) and attention_mask.dtype == torch.long:
        attention_mask = attention_mask.to(dtype=torch.bool)

    if cache_position is None:
        if input_ids is not None:
            device = input_ids.device
            seq_length = input_ids.shape[1]
        else:
            device = input_embeds.device
            seq_length = input_embeds.shape[1]
        cache_position = torch.arange(0, seq_length, device=device)

    # Pass embed tensor positionally: kwarg name differs across transformers
    # versions (input_embeds in <=5.1, inputs_embeds in >=5.5 with a deprecation
    # shim on the old name). past_key_values is keyword-only in 5.5+.
    full_mask = create_causal_mask(
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    sliding_mask = create_sliding_window_causal_mask(
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    return {
        "full_attention": full_mask,
        "sliding_attention": sliding_mask,
    }
