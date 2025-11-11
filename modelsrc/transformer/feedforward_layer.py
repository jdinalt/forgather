from typing import Optional, Callable
from torch import nn, FloatTensor


# A basic feedforward layer
# https://arxiv.org/pdf/1706.03762
class FeedforwardLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_feedforward: int,
        *,
        activation_factory: Optional[Callable] = lambda: nn.ReLU(),
        dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_feedforward = d_feedforward

        self.linear1 = nn.Linear(self.d_model, self.d_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout != 0.0 else nn.Identity()
        self.activation = activation_factory()
        self.linear2 = nn.Linear(self.d_feedforward, self.d_model, bias=bias)

    def extra_repr(self):
        return f"d_model={self.d_model}, d_feedforward={self.d_feedforward}"

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
