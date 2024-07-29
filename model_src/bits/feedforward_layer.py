from torch import nn, Tensor, FloatTensor
from collections import OrderedDict


# A basic feedforward layer
# https://arxiv.org/pdf/1706.03762
class FeedforwardLayer(nn.Sequential):
    def __init__(
        self,
        d_model: int,
        d_feedforward: int,
        *,
        activation=nn.ReLU(),
        dropout: float = 0.0,
        bias: bool = True,
    ):
        self.d_model = d_model
        self.d_feedforward = d_feedforward

        super().__init__(
            OrderedDict(
                [
                    ("linear1", nn.Linear(self.d_model, self.d_feedforward, bias=bias)),
                    (
                        "dropout",
                        nn.Dropout(dropout) if dropout != 0.0 else nn.Identity(),
                    ),
                    ("activation", activation),
                    ("linear2", nn.Linear(self.d_feedforward, self.d_model, bias=bias)),
                ]
            )
        )

    def extra_repr(self):
        return f"d_model={self.d_model}, d_feedforward={self.d_feedforward}"
