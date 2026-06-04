from typing import Any, Callable, Optional

from torch import FloatTensor, nn


class SmolLM3DecoderLayer(nn.Module):
    """
    SmolLM3 decoder layer: a standard pre-LayerNorm Llama block with **NoPE**
    (No Positional Embedding) applied to a periodic subset of layers.

    SmolLM3 (https://huggingface.co/blog/smollm3) is architecturally a Llama
    model -- GQA, SiLU GLU MLP, RMSNorm, tied embeddings -- with one twist: it
    omits rotary position embeddings from every ``no_rope_layer_interval``-th
    layer. The intuition (Kazemnejad et al. 2023, "The Impact of Positional
    Encoding on Length Generalization", arXiv:2305.19466) is that interleaving
    position-free attention layers improves length generalization without a
    measurable cost on short-context quality.

    Which layers are NoPE is encoded in ``config.no_rope_layers``: a list of
    ``num_hidden_layers`` flags where ``1`` means "apply RoPE" and ``0`` means
    "NoPE" (matching HuggingFace's ``SmolLM3Config`` convention, generated as
    ``(layer_idx + 1) % no_rope_layer_interval != 0``).

    This layer differs from :class:`PreLNLayer` only at construction time: for a
    NoPE layer it builds the attention module with ``pos_encoder=None`` so that
    no rotation is applied. (``apply_rotary_pos_emb`` raises if handed ``None``
    position embeddings, so the encoder must be disabled at the module level
    rather than by passing empty embeddings.) ``position_embeddings`` are still
    computed once by :class:`CasualLM` and threaded through ``forward`` kwargs;
    a NoPE layer simply ignores them because its ``pos_encoder`` is ``None``.
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

        # 1 -> apply RoPE, 0 -> NoPE. Default to applying RoPE when the config
        # carries no schedule (e.g. a plain Llama-style config).
        no_rope_layers = getattr(config, "no_rope_layers", None)
        if no_rope_layers and 0 <= layer_idx < len(no_rope_layers):
            self.apply_rope = bool(no_rope_layers[layer_idx])
        else:
            self.apply_rope = True

        # For NoPE layers, override the bound pos_encoder (apply_rotary_pos_emb)
        # with None so the attention module skips the rotation entirely.
        attn_kwargs = {} if self.apply_rope else {"pos_encoder": None}
        self.attention = attention_factory(layer_idx=layer_idx, **attn_kwargs, **kwargs)
        self.feedforward = feedforward_factory(**kwargs)
        self.norm1 = norm_factory()
        self.norm2 = norm_factory()

        self.dropout_p = dropout
        self.residual_dropout_p = residual_dropout
        self.dropout = nn.Identity() if not dropout else nn.Dropout(dropout)
        self.residual_dropout = (
            nn.Identity() if not residual_dropout else nn.Dropout(residual_dropout)
        )

    def extra_repr(self) -> str:
        return (
            f"layer_idx={self.layer_idx}, apply_rope={self.apply_rope}, "
            f"dropout={self.dropout_p}, residual_dropout={self.residual_dropout_p}"
        )

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        residual = self.residual_dropout(x)
        x = self.norm1(x)
        x = self.attention(x, **kwargs)
        x = residual + self.dropout(x)
        residual = self.residual_dropout(x)
        x = self.norm2(x)
        x = self.feedforward(x, **kwargs)
        x = residual + self.dropout(x)
        return x
