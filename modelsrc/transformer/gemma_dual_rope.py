from typing import Any, Dict, Optional, Tuple

from torch import Tensor, nn

from .rotary_embeddings import RotaryEmbedding


class GemmaDualRotaryEmbedding(nn.Module):
    """
    Rotary position embedding module for Gemma-3, which uses two different
    RoPE frequency bases depending on layer type:

    - ``full_attention`` layers use a large base (e.g. ``rope_theta=1e6``)
    - ``sliding_attention`` layers use a small base (e.g. ``rope_theta=1e4``)

    HuggingFace stores both in ``config.rope_parameters`` as a dict keyed by
    layer type::

        {
            "full_attention":    {"rope_type": "default", "rope_theta": 1000000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta":   10000.0},
        }

    This module creates one :class:`RotaryEmbedding` per layer type and, in
    ``forward``, returns a dict of ``(cos, sin)`` tuples keyed the same way.
    The :class:`GemmaDecoderLayer` selects the appropriate entry based on its
    own layer type.

    Note on ``head_dim``: Gemma decouples ``head_dim`` from
    ``hidden_size // num_attention_heads``. Since :class:`RotaryEmbedding`
    computes ``d_head = hidden_size // num_attention_heads`` internally, we
    pass a synthetic ``hidden_size = head_dim * num_attention_heads`` so the
    frequency buffer comes out the right size.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_position_embeddings: int = 2048,
        head_dim: Optional[int] = None,
        rope_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        float32_output: bool = False,
        device=None,
    ):
        super().__init__()

        effective_hidden = (
            head_dim * num_attention_heads if head_dim is not None else hidden_size
        )

        rope_parameters = rope_parameters or {}
        global_params = rope_parameters.get(
            "full_attention", {"rope_theta": 1000000.0}
        )
        local_params = rope_parameters.get(
            "sliding_attention", {"rope_theta": 10000.0}
        )

        self.rotary_full = RotaryEmbedding(
            hidden_size=effective_hidden,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            rope_parameters=global_params,
            float32_output=float32_output,
            device=device,
        )
        self.rotary_sliding = RotaryEmbedding(
            hidden_size=effective_hidden,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            rope_parameters=local_params,
            float32_output=float32_output,
            device=device,
        )

    def forward(
        self, x: Tensor, position_ids: Tensor
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        return {
            "full_attention": self.rotary_full(x, position_ids),
            "sliding_attention": self.rotary_sliding(x, position_ids),
        }
