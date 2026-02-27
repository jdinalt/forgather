from typing import Callable, Optional

from torch import FloatTensor, nn


# Attention Is All You Need: https://arxiv.org/pdf/1706.03762
# On Layer Normalization in the Transformer Architecture: https://arxiv.org/pdf/2002.04745
class PreLNLayer(nn.Module):
    def __init__(
        self,
        *,
        feedforward_factory: Callable,
        attention_factory: Callable,
        norm_factory: Callable,
        dropout: Optional[float] = 0.0,
        residual_dropout: Optional[float] = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.feedforward = feedforward_factory(**kwargs)
        self.attention = attention_factory(**kwargs)
        self.norm1 = norm_factory()
        self.norm2 = norm_factory()
        self.dropout_p = dropout
        self.residual_dropout_p = residual_dropout
        if not dropout:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)
        # Residual Dropout:A Simple Approach to Improve Transformer’s Data Efficiency
        # https://aclanthology.org/2024.sigul-1.35.pdf
        if not residual_dropout:
            self.residual_dropout = nn.Identity()
        else:
            self.residual_dropout = nn.Dropout(residual_dropout)

    def extra_repr(self):
        return (
            f"dropout={self.dropout_p}, "
            f"residual_dropout={self.residual_dropout_p}"
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
