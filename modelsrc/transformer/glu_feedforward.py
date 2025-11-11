from typing import Optional, Callable
from torch import nn, FloatTensor


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

        self.up_proj = nn.Linear(self.d_model, self.d_feedforward, bias=False)
        self.gate_proj = nn.Linear(self.d_model, self.d_feedforward, bias=False)
        self.down_proj = nn.Linear(self.d_feedforward, self.d_model, bias=False)
        self.activation = activation_factory()
        if dropout == 0.0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

    def extra_repr(self):
        return f"d_model={self.d_model}, d_feedforward={self.d_feedforward}"

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = up * self.activation(gate)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x
