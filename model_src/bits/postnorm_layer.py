from torch import nn


# Standard transformer layer, from original paper.
class PostnormLayer(nn.Module):
    def __init__(
        self,
        d_model,
        attention,
    ):
        super().__init__()
        self.d_model = d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        # Keep input as residual
        residual = x

        # Compute attention
        x = self.attention(x)

        # Add attention with residual and normalize.
        x = self.norm1(residual + x)

        return x
