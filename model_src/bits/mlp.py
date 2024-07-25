import torch
from torch import nn


class FeedforwardLayer(nn.Module):
    def __init__(self, d_model, d_feedforward):
        super().__init__()
        self.d_model = d_model
        self.d_feedforward = d_feedforward

        self.linear1 = nn.Linear(self.d_model, self.d_feedforward)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(self.d_feedforward, self.d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
