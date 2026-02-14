import math
from typing import Callable, Optional

from torch import FloatTensor, LongTensor, nn


class InputEncoder(nn.Module):
    """
    Converts input-ids to embeddings and, optionally, adds positional encodings

    Also performs embedding dropout by default, as per the Attention is All You Need.

    d_model: Model hidden dimesnion; used to compute sqrt(d_model), if scale_sqrt_d_model
    dropout: Embedding dropout probability, as per original Transformer paper.
    positional_encoder: And absolute positional encoder or None
    embedding: A torch nn.Embedding or equivalent
    scale: Set embedding scale. scale_sqrt_d_model == True overrides this value.
    scale_sqrt_d_model: Multiply  sqrt(d_model), as per Attention is All you Need.
        Note: When used, embedding std should be: 1/sqrt(d_model)
    """

    def __init__(
        self,
        d_model: int,
        embedding: nn.Module,
        *,
        dropout: float = 0.0,
        positional_encoder: Optional[nn.Module] = None,
        scale: float = 1.0,
        scale_sqrt_d_model: bool = False,
    ):
        super().__init__()
        self.scale = math.sqrt(d_model) if scale_sqrt_d_model else scale
        self.dropout = nn.Identity() if dropout == 0.0 else nn.Dropout(dropout)
        self.embedding = embedding
        setattr(self.embedding, "init_prefix", "embedding")

        if positional_encoder is not None:
            self.positional_encoder = positional_encoder
        else:
            self.positional_encoder = None

    def extra_repr(self):
        return f"scale={self.scale}"

    def forward(
        self, input_ids: LongTensor, position_ids: Optional[LongTensor] = None
    ) -> FloatTensor:
        x = self.embedding(input_ids)
        if self.scale != 1.0:
            x = x * self.scale
        if self.positional_encoder is not None:
            x = self.positional_encoder(x, position_ids=position_ids)
        return self.dropout(x)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.embedding = new_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        if self.positional_encoder:
            self.positional_encoder.resize_position_embeddings(
                new_num_position_embeddings
            )

    def get_position_embeddings(self):
        if self.positional_encoder:
            return self.positional_encoder.get_position_embeddings()
        else:
            return None
