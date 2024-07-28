from torch import nn, Tensor


# https://arxiv.org/pdf/2002.04745
# https://arxiv.org/pdf/2002.04745
class PreLNLayer(nn.Module):
    def __init__(
        self,
        *,
        feedforward: nn.Module,
        attention: nn.Module,
        norm1: nn.Module,
        norm2: nn.Module,
        dropout: float = 0.1,
        residual_dropout=0.0,
    ):
        super().__init__()
        self.feedforward = feedforward
        self.attention = attention
        self.norm1 = norm1
        self.norm2 = norm2
        self.dropout = nn.Dropout(dropout)
        # https://aclanthology.org/2024.sigul-1.35.pdf
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: Tensor):
        residual = self.residual_dropout(x)
        x = self.norm1(x)
        x = self.attention(x)
        x = residual + self.dropout(x)
        residual = self.residual_dropout(x)
        x = self.norm2(x)
        x = self.feedforward(x)
        x = residual + self.dropout(x)
        return x
