from torch import nn, Tensor, LongTensor, FloatTensor


class InputEncoder(nn.Module):
    """
    Converts input-ids to embeddings and, optionally, adds positional encodings

    Also performs embedding dropout by default, as per the Attention is All You Need.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        *,
        dropout: float = 0.1,
        positional_encoder: nn.Module = None,
        # Defaults to d_model ** 0.5
        embedding_scale: float = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        if embedding_scale is None:
            self.embedding_scale = d_model ** 0.5
        else:
            self.embedding_scale = embedding_scale

        if dropout == 0.0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoder = positional_encoder

    def extra_repr(self):
        return f"d_model={self.d_model}, vocab_size={self.vocab_size}, embedding_scale={self.embedding_scale}"

    def forward(
        self, input_ids: LongTensor, position_ids: LongTensor = None
    ) -> FloatTensor:
        x = self.embedding(input_ids) * self.embedding_scale
        if self.positional_encoder is not None:
            x = self.positional_encoder(x, position_ids=position_ids)
        return self.dropout(x)
