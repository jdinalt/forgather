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
        dropout: float=0.1,
        positional_encoder: nn.Module=None,
    ):
        super().__init__()
        self.sqrt_d_model = d_model ** 0.5
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.positional_encoder = positional_encoder

    def forward(self, input_ids: LongTensor, position_ids: LongTensor=None) -> FloatTensor:
        x = self.embedding(input_ids) * self.sqrt_d_model
        if self.positional_encoder is not None:
            x = x + self.positional_encoder(x.size(1), position_ids=position_ids)
        return self.embedding_dropout(x)