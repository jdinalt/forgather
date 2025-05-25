from typing import Callable, Optional
from torch import nn, Tensor, LongTensor, FloatTensor


class InputEncoder(nn.Module):
    """
    Converts input-ids to embeddings and, optionally, adds positional encodings

    Also performs embedding dropout by default, as per the Attention is All You Need.

    d_model: Model hidden dimesnion
    vocab_size: Number of tokens in vocabulary
    dropout: Embedding dropout probability, as per original Transformer paper.
    positional_encoder_factory: Constructs a positional encoder; default is None
    embedding_factory: Constructs an embedding implementaiton. Default is nn.Embedding(vocab_size, d_model)
    embeddint_scale: Multiplier for embedding; defaults to sqrt(d_model), as per Attention is All you Need.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        *,
        dropout: Optional[float] = 0.1,
        positional_encoder: Optional[Callable] = None,
        embedding: Optional[Callable] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

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
        if self.positional_encoder is not None:
            x = self.positional_encoder(x, position_ids=position_ids)
        return self.dropout(x)
