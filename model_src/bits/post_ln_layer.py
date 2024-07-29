from torch import nn, Tensor


# https://arxiv.org/pdf/1706.03762
class PostLNLayer(nn.Module):
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
        if dropout == 0.0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)
        # https://aclanthology.org/2024.sigul-1.35.pdf
        if residual_dropout == 0.0:
            self.residual_dropout = nn.Identity()
        else:
            self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: Tensor):
        residual = self.residual_dropout(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = self.norm1(residual + x)
        residual = self.residual_dropout(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.norm2(residual + x)
        return x
