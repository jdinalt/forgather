import math

import torch
from torch import nn


# A vanilla positional encoder
class PositionalEncoder(nn.Module):
    def __init__(self, d_embed, max_seq):
        super().__init__()
        self.d_embed = d_embed
        self.max_seq = max_seq

        weight = torch.zeros(max_seq, d_embed)
        position = torch.arange(0, max_seq, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed)
        )
        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)
        weight = weight.unsqueeze(0)
        self.register_buffer("weight", weight)

    def forward(self, x):
        seq_len = x.size(-2)
        return x + self.weight[:, :seq_len]
