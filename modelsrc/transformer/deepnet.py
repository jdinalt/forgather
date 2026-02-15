from typing import Callable, Optional

from torch import FloatTensor, nn


class DeepnetLayer(nn.Module):
    """
    DeepNet: Scaling Transformers to 1,000 Layers
    https://arxiv.org/pdf/2203.00555

    This is a post-norm type transformer layer which implements
    "deepnorm" in forward().
    """

    def __init__(
        self,
        *,
        feedforward_factory: Callable,
        attention_factory: Callable,
        norm_factory: Callable,
        dropout: Optional[float] = 0.0,
        residual_dropout: Optional[float] = 0.0,
        alpha=1.0,  # See deepnet_alpha()
        **kwargs,
    ):
        super().__init__()
        self.feedforward = feedforward_factory(**kwargs)
        self.attention = attention_factory(**kwargs)
        self.norm1 = norm_factory()
        self.norm2 = norm_factory()
        if dropout == 0.0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)
        # Residual Dropout:A Simple Approach to Improve Transformer’s Data Efficiency
        # https://aclanthology.org/2024.sigul-1.35.pdf
        if residual_dropout == 0.0:
            self.residual_dropout = nn.Identity()
        else:
            self.residual_dropout = nn.Dropout(residual_dropout)
        self.alpha = alpha

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        residual = self.residual_dropout(x)
        x = self.attention(x, **kwargs)
        x = self.dropout(x)
        x = self.norm1(residual * self.alpha + x)
        residual = self.residual_dropout(x)
        x = self.feedforward(x, **kwargs)
        x = self.dropout(x)
        x = self.norm2(residual * self.alpha + x)
        return x


def deepnet_alpha(n_decoder_layers: int, n_encoder_layers: int, which: str = None):
    """
    Compute deeepnet "alpha", which is passed to DeepnetLayer.
    The residuals are scaled by this value.

    Note: Only tested with Decoder-Only config
    """
    # Alias for consistency with paper
    n = n_encoder_layers
    m = n_decoder_layers

    assert n >= 0 and m >= 0

    # Decoder only (e.g. GPT)
    if n == 0:
        return (2 * m) ** (1 / 4)
    # Enocoder Only (e.g. BERT)
    elif m == 0:
        return (2 * n) ** (1 / 4)
    # Encoder / Decoder
    else:
        match which:
            case "encoder":
                return 0.81 * (n**4 * m) ** (1 / 16)
            case "decoder":
                return (3 * m) ** (1 / 4)
            case _:
                raise Exception("Which argument must be either encoder or decoder")


def deepnet_beta(n_decoder_layers: int, n_encoder_layers: int, which: str = None):
    """
    Compute deeepnet "beta", which passed as "std" or "gain" when
    initialiizing feedforard, value_projection, and out_projection weights.

    Note: Only tested with Decoder-Only config
    """
    # Alias for consistency with paper
    n = n_encoder_layers
    m = n_decoder_layers

    assert n >= 0 and m >= 0

    # Decoder only (e.g. GPT)
    if n == 0:
        return (8 * m) ** (-1 / 4)
    # Enocoder Only (e.g. BERT)
    elif m == 0:
        return (8 * n) ** (-1 / 4)
    # Encoder / Decoder
    else:
        match which:
            case "encoder":
                return 0.87 * (n**4 * m) ** (-1 / 16)
            case "decoder":
                return (12 * m) ** (-1 / 4)
            case _:
                raise Exception("Which argument must be either encoder or decoder")
