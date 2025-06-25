from typing import Optional, Callable
import torch
from torch import nn, FloatTensor, Tensor


# GLU Variants Improve Transformer
# https://arxiv.org/pdf/2002.05202v1.pdf
class GLUFeedforwardLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_feedforward: int,
        *,
        activation_factory: Optional[Callable] = lambda: nn.SiLU(),
        dropout: Optional[float] = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_feedforward = d_feedforward

        self.linear1 = nn.Linear(self.d_model, self.d_feedforward * 2, bias=False)
        self.linear2 = nn.Linear(self.d_feedforward, self.d_model, bias=False)
        self.activation = activation_factory()
        if dropout == 0.0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

    def extra_repr(self):
        return f"d_model={self.d_model}, d_feedforward={self.d_feedforward}"

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        x, gate = self.linear1(x).chunk(2, dim=-1)
        x = x * self.activation(gate)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
