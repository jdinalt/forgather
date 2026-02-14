import math

import torch
from torch import FloatTensor, nn

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
        *,
        bias: bool = False,
        dropout: float = 0.0,
        slope_init: float = 0.5,
        trainable_alibi: bool = True,
        layer_idx: int,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.slope_init = slope_init
        self.trainable_alibi = trainable_alibi
        self.bias = bias
        self.layer_idx = layer_idx

        # We scale the attention scores by the inverse-square-root of the head dimension
        # this shifts the temerature of softmax.
        self.dot_product_scale = 1.0 / math.sqrt(self.d_model)

        # When an attention layer has only one head, the original four matrices can
        # be represented with only two matrices, as the math works out the same.
        # See: https://transformer-circuits.pub/2021/framework/index.html#splitting-attention-head-terms-into-circuits
        self.query_key_linear = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.value_linear = nn.Linear(self.d_model, self.d_model, bias=self.bias)

        # Unlike the original design, the slope is a learnable parameter.

        if self.trainable_alibi:
            self.alibi_slope = nn.Parameter(torch.empty((1,), dtype=torch.float32))
        else:
            self.alibi_slope = self.slope_init
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        if self.trainable_alibi:
            with torch.no_grad():
                self.alibi_slope.fill_(self.slope_init)

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, slope_init={self.slope_init}, trainable_alibi={self.trainable_alibi} "
            f"bias={self.bias}"
        )

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

        # Apply Alibi relative positional biases.
        scores += (
            alibi_biases(seq_len, seq_len, device=x.device, dtype=x.dtype)
            * self.alibi_slope
        )

        if seq_len > 1:
            # Mask future positions from the past
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            )
            scores = torch.where(causal_mask, scores, float("-inf"))

        # Calculate the attention weights; avoid NANs that might emerge from zeros in softmax's denominator
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Use the attention weights to get a weighted combination of value vectors
        attended_values = torch.matmul(attention_weights, self.value_linear(x))

        return attended_values
