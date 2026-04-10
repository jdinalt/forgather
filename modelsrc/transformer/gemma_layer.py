from typing import Any, Callable, Optional

from torch import FloatTensor, nn


class GemmaDecoderLayer(nn.Module):
    """
    Gemma-3 decoder layer.

    Differs from the standard ``PreLNLayer`` in three ways:

    1. **Four layernorms per layer** instead of two:
       ``input_layernorm`` and ``post_attention_layernorm`` wrap the attention
       block, and ``pre_feedforward_layernorm`` / ``post_feedforward_layernorm``
       wrap the feedforward block. The names match HF Gemma3 so parameter
       remapping is identity.

    2. **Per-layer attention type**: each layer is either ``full_attention``
       or ``sliding_attention``, taken from ``config.layer_types[layer_idx]``.
       The attention module is constructed with ``sliding_window`` set to
       ``config.sliding_window`` for sliding layers and ``None`` for full
       layers.

    3. **Per-layer dispatch of position embeddings and attention masks**:
       :class:`GemmaDualRotaryEmbedding` and ``gemma_mask_fn`` produce dicts
       keyed by layer type. This layer unpacks the appropriate entry at
       forward time before invoking attention. Non-dict inputs are passed
       through unchanged so the module also works in single-RoPE setups.
    """

    def __init__(
        self,
        *,
        feedforward_factory: Callable,
        attention_factory: Callable,
        norm_factory: Callable,
        config: Any,
        layer_idx: int,
        dropout: Optional[float] = 0.0,
        residual_dropout: Optional[float] = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        layer_types = getattr(config, "layer_types", None) or []
        if 0 <= layer_idx < len(layer_types):
            self.layer_type = layer_types[layer_idx]
        else:
            self.layer_type = "full_attention"

        config_sliding = getattr(config, "sliding_window", None)
        if self.layer_type == "sliding_attention" and config_sliding:
            sliding_window = config_sliding
        else:
            sliding_window = None

        self.attention = attention_factory(
            layer_idx=layer_idx,
            sliding_window=sliding_window,
            **kwargs,
        )
        self.feedforward = feedforward_factory(**kwargs)

        self.input_layernorm = norm_factory()
        self.post_attention_layernorm = norm_factory()
        self.pre_feedforward_layernorm = norm_factory()
        self.post_feedforward_layernorm = norm_factory()

        self.dropout_p = dropout
        self.residual_dropout_p = residual_dropout
        self.dropout = nn.Identity() if not dropout else nn.Dropout(dropout)
        self.residual_dropout = (
            nn.Identity() if not residual_dropout else nn.Dropout(residual_dropout)
        )

    def extra_repr(self) -> str:
        return (
            f"layer_idx={self.layer_idx}, layer_type={self.layer_type!r}, "
            f"dropout={self.dropout_p}, residual_dropout={self.residual_dropout_p}"
        )

    def forward(
        self,
        x: FloatTensor,
        attention_mask=None,
        position_embeddings=None,
        **kwargs,
    ) -> FloatTensor:
        if isinstance(position_embeddings, dict):
            pos_emb = position_embeddings[self.layer_type]
        else:
            pos_emb = position_embeddings

        if isinstance(attention_mask, dict):
            attn_mask = attention_mask[self.layer_type]
        else:
            attn_mask = attention_mask

        residual = self.residual_dropout(x)
        x = self.input_layernorm(x)
        x = self.attention(
            x,
            attention_mask=attn_mask,
            position_embeddings=pos_emb,
            **kwargs,
        )
        x = self.post_attention_layernorm(x)
        x = residual + self.dropout(x)

        residual = self.residual_dropout(x)
        x = self.pre_feedforward_layernorm(x)
        x = self.feedforward(x, **kwargs)
        x = self.post_feedforward_layernorm(x)
        x = residual + self.dropout(x)

        return x
