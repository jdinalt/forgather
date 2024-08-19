import torch
from torch import nn, Tensor, FloatTensor
import math

# Attention layer with ALiBi relative positional encoding
# TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION
# https://arxiv.org/pdf/2108.12409.pdf


def alibi_biases(query_len: int, key_len: int, device, dtype):
    x = torch.arange(key_len, device=device, dtype=dtype)[None, :]
    y = torch.arange(query_len, device=device, dtype=dtype)[:, None]
    return x - y


# A simple causal multi-head-attention implementation
class SingleHeadAlibiAttn(nn.Module):
    """
    A simple single-head causal attention layer with ALiBI positional biases.
    """

    def __init__(
        self,
        d_model: int,
        bias: bool = True,
        slope_init: float = 0.5,
        trainable_slope: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.slope_init = slope_init
        self.trainable_slope = trainable_slope
        self.bias = bias

        # We scale the attention scores by the inverse-square-root of the head dimension
        # this shifts the temerature of softmax.
        self.dot_product_scale = 1.0 / math.sqrt(self.d_model)

        # tl;dr When an attention layer has only one head, the origianl four matrices can
        # be represented with only two matrices, as the math works out the same.
        # See: https://transformer-circuits.pub/2021/framework/index.html#splitting-attention-head-terms-into-circuits
        self.query_key_linear = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.value_linear = nn.Linear(self.d_model, self.d_model, bias=self.bias)

        # Unlike the original design, the slope is a learnable parameter.
        if self.trainable_slope:
            self.alibi_slope = nn.Parameter(torch.tensor(self.slope_init))
        else:
            self.alibi_slope = self.slope_init

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, slope_init={self.slope_init}, trainable_slope={self.trainable_slope} "
            f"bias={self.bias}"
        )

    def forward(self, x: FloatTensor) -> FloatTensor:
        # x: (batch_size, seq_len, d_hidden)
        _, seq_len, _ = x.shape

        # Compute scores.
        scores = (
            x
            @ self.query_key_linear.weight
            @ x.transpose(-2, -1)
            * self.dot_product_scale
        )

        # Apply Alibi relative positional biases.
        scores += (
            alibi_biases(seq_len, seq_len, device=x.device, dtype=x.dtype)
            * self.alibi_slope
        )

        # Mask future positions from the past
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), True, device=x.device), diagonal=1
        )
        scores.masked_fill_(causal_mask, float("-inf"))
        del causal_mask

        # Calculate the attention weights; avoid NANs that might emerge from zeros in softmax's denominator
        attention_weights = torch.softmax(scores, dim=-1)
        del scores

        # Use the attention weights to get a weighted combination of value vectors
        attended_values = torch.matmul(attention_weights, self.value_linear(x))

        return attended_values
