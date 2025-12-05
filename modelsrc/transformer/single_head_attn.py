import torch
from torch import nn, FloatTensor
import math


class SingleHeadAttn(nn.Module):
    """
    A simple single-head causal attention layer.
    """

    def __init__(
        self,
        d_model: int,
        *,
        bias: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.bias = bias

        # We scale the attention scores by the inverse-square-root of the head dimension
        # this shifts the temerature of softmax.
        self.dot_product_scale = 1.0 / math.sqrt(self.d_model)

        # tl;dr When an attention layer has only one head, the origianl four matrices can
        # be represented with only two matrices, as the math works out the same.
        # See: https://transformer-circuits.pub/2021/framework/index.html#splitting-attention-head-terms-into-circuits
        self.query_key_linear = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.value_linear = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def extra_repr(self):
        return f"d_model={self.d_model}, bias={self.bias}"

    def forward(self, x: FloatTensor, **kwargs) -> FloatTensor:
        # x: (batch_size, seq_len, d_hidden)
        _, seq_len, _ = x.shape

        # Compute scores.
        scores = (
            x
            @ self.query_key_linear.weight
            @ x.transpose(-2, -1)
            * self.dot_product_scale
        )

        # Mask future positions from the past
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), True, device=x.device), diagonal=1
        )
        scores.masked_fill_(causal_mask, float("-inf"))

        # Calculate the attention weights; avoid NANs that might emerge from zeros in softmax's denominator
        attention_weights = torch.softmax(scores, dim=-1).clamp(min=1e-10)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Use the attention weights to get a weighted combination of value vectors
        attended_values = torch.matmul(attention_weights, self.value_linear(x))
        return attended_values
