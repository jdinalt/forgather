import math
from typing import Iterable, Optional

import torch
from torch import LongTensor, Tensor, nn
from torch.nn import Buffer


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
        self.weight = Buffer(
            torch.zeros(self.max_sequence_length, self.d_model), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self):
        position = torch.arange(
            0, self.max_sequence_length, dtype=torch.float
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )
        self.weight[:, 0::2] = torch.sin(position * div_term)
        self.weight[:, 1::2] = torch.cos(position * div_term)

    def extra_repr(self):
        return f"d_model={self.d_model}, max_sequence_length={self.max_sequence_length}"

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> Iterable[str]:
        """Hack for vllm. We don't need to load these weights!
        TODO: Revisit
        """
        return [name for name, _ in weights]

    def forward(
        self, x: Tensor, *, position_ids: Optional[LongTensor] = None
    ) -> Tensor:
        seq_length = x.size(1)
        if position_ids is not None:
            return x + self.weight[position_ids]
        else:
            return x + self.weight[:seq_length].unsqueeze(0)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        self.max_sequence_length = new_num_position_embeddings
        self.reset_parameters()

    def get_position_embeddings(self):
        return self.weight
