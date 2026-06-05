from typing import Callable, Optional

from torch import FloatTensor, nn


class AttentionOnlyLayer(nn.Module):
    """
    A pre-LN transformer block with attention and **no feedforward (MLP)**.

    This matches the *attention-only* transformer studied in Elhage et al. 2021,
    "A Mathematical Framework for Transformer Circuits": the residual stream is
    updated solely by attention sub-blocks, with no MLP in between. It is the
    paper-faithful counterpart to the standard :class:`PreLNLayer` (attention +
    MLP), and the two are compared head-to-head in the ``attention_only`` Tiny
    Experiment.

    The forward pass is exactly :class:`PreLNLayer`'s attention half: a pre-norm
    residual update ``x = x + attention(norm(x))`` and nothing more.
    """

    def __init__(
        self,
        *,
        attention_factory: Callable,
        norm_factory: Callable,
        dropout: Optional[float] = 0.0,
        residual_dropout: Optional[float] = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.attention = attention_factory(**kwargs)
        self.norm1 = norm_factory()
        self.dropout_p = dropout
        self.residual_dropout_p = residual_dropout
        self.dropout = nn.Identity() if not dropout else nn.Dropout(dropout)
        self.residual_dropout = (
            nn.Identity() if not residual_dropout else nn.Dropout(residual_dropout)
        )

    def extra_repr(self):
        return f"dropout={self.dropout_p}, residual_dropout={self.residual_dropout_p}"

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        residual = self.residual_dropout(x)
        x = self.norm1(x)
        x = self.attention(x, **kwargs)
        x = residual + self.dropout(x)
        return x
