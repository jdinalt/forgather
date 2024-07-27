from collections import OrderedDict

import torch
from torch import nn


# A basic feedforward layer, implemented as a nn.Sequential
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

        stack = OrderedDict(
            linear1=nn.Linear(self.d_model, self.d_feedforward, bias=bias),
            activation=activation,
            dropout=nn.Dropout(dropout),
            linear2=nn.Linear(self.d_feedforward, self.d_model, bias=bias),
        )
        # Remove the dropout, if zero
        if dropout == 0.0:
            del stack["dropout"]
        super().__init__(stack)
