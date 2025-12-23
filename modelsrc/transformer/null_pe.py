from typing import Optional

from torch import FloatTensor, LongTensor, Tensor, nn


class NullPE(nn.Module):
    """
    Null Positional Encoder; returns the input, unmodified.
    """

    def forward(
        self, x: FloatTensor, *, position_ids: Optional[LongTensor] = None
    ) -> Tensor:
        return x

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self):
        return None
