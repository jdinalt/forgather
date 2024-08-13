from typing import Optional
import math

import torch
from torch import nn, Tensor, FloatTensor, LongTensor


# An implementation of the original transformer sinusoidal positional encoder.
# https://arxiv.org/pdf/1706.03762
class SinusoidalPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_sequence_length: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

        weight = torch.zeros(max_sequence_length, d_model)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)
        weight = weight.unsqueeze(0)
        self.register_buffer("weight", weight, persistent=False)

    def extra_repr(self):
        return f"d_model={self.d_model}, max_sequence_length={self.max_sequence_length}"

    def forward(
        self, x: FloatTensor, *, position_ids: Optional[LongTensor] = None
    ) -> FloatTensor:
        seq_length = x.size(1)
        if position_ids is not None:
            return x + self.weight[position_ids]
        else:
            return x + self.weight[:, :seq_length]
