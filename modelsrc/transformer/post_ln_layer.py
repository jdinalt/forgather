from typing import Callable, Optional
from torch import nn, Tensor, FloatTensor


# Attention Is All You Need: https://arxiv.org/pdf/1706.03762
# On Layer Normalization in the Transformer Architecture: https://arxiv.org/pdf/2002.04745
class PostLNLayer(nn.Module):
    def __init__(
        self,
        *,
        feedforward_factory: Callable,
        attention_factory: Callable,
        norm_factory: Callable,
        dropout: Optional[float] = 0.1,
        residual_dropout: Optional[float] = 0.0,
    ):
        super().__init__()
        self.feedforward = feedforward_factory()
        self.attention = attention_factory()
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

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        residual = self.residual_dropout(x)
        x = self.attention(x, **kwargs)
        x = self.dropout(x)
        x = self.norm1(residual + x)
        residual = self.residual_dropout(x)
        x = self.feedforward(x, **kwargs)
        x = self.dropout(x)
        x = self.norm2(residual + x)
        return x
