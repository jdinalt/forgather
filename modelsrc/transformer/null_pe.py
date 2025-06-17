from typing import Optional
import math

import torch
from torch import nn, Tensor, FloatTensor, LongTensor


class NullPE(nn.Module):
    """
    Null Positional Encoder; returns the input, unmodified.
    """

    def forward(
        self, x: FloatTensor, *, position_ids: Optional[LongTensor] = None
    ) -> Tensor:
        return x
