from typing import Callable, Optional
from torch import nn, LongTensor, FloatTensor
import math


class InputEncoder(nn.Module):
    """
    Converts input-ids to embeddings and, optionally, adds positional encodings

    Also performs embedding dropout by default, as per the Attention is All You Need.

    d_model: Model hidden dimesnion
    vocab_size: Number of tokens in vocabulary
    dropout: Embedding dropout probability, as per original Transformer paper.
    positional_encoder_factory: Constructs a positional encoder; default is None
    embedding_factory: Constructs an embedding implementaiton. Default is nn.Embedding(vocab_size, d_model)
    scale: Set embedding scale. scale_sqrt_d_model == True overrides this value.
    scale_sqrt_d_model: Multiply  sqrt(d_model), as per Attention is All you Need.
        Note: When used, embedding std should be: 1/sqrt(d_model)
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        *,
        dropout: Optional[float] = 0.1,
        positional_encoder: Optional[Callable] = None,
        embedding: Optional[Callable] = None,
        scale: float = 1.0,
        scale_sqrt_d_model: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        if scale_sqrt_d_model:
            self.scale = math.sqrt(d_model)
        else:
            self.scale = scale

        if dropout == 0.0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, d_model)

        if positional_encoder is not None:
            self.positional_encoder = positional_encoder
        else:
            self.positional_encoder = None

    def extra_repr(self):
        return f"d_model={self.d_model}, vocab_size={self.vocab_size}"

    def forward(
        self, input_ids: LongTensor, position_ids: LongTensor = None
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
