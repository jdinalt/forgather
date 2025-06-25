import torch
from torch import nn, Tensor, FloatTensor
import math

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)

from torch.nn import functional as F

# Workaround for https://github.com/huggingface/transformers/issues/28459
if is_flash_attn_2_available():
    try:
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    except:
        print("Could not import flash2")


class CausalMultiheadFlashAttn(nn.Module):
    """
    Multihead Attention with support for Flash-Attention-2, Torch flash-attention, and native attention.

    This module is a little more complex than the baseline version, CausalMultiheadAttn, but can be much faster.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        bias: bool = True,
        attn_type: str = "native",
        dropout=0.1,
    ):
        super().__init__()

        self.attn_type = attn_type

        self.use_torch = True

        match attn_type:
            # Use local impementation; slowest option; good for debugging; useful when experimenting with non-standard stuff.
            case "native":
                self.use_torch = False
                pass
            # Use Flash-Attention2 implementation; fastest; limited to int16 and bfloat16 types; least memory usage.
            case "flash2":
                pass
                # Use torch.nn.functional.scaled_dot_product_attention() with any available backend
                # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            case "torch":
                self.torch_backend = [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
                # Use a specific torch backend
                # https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend
            case "torch_math":
                self.torch_backend = SDPBackend.MATH
            case "torch_flash":
                self.torch_backend = SDPBackend.FLASH_ATTENTION
            case "torch_efficient":
                self.torch_backend = SDPBackend.EFFICIENT_ATTENTION
            case "torch_cudnn":
                self.torch_backend = SDPBackend.CUDNN_ATTENTION
            case _:
                raise Exception(f"Unknown attention type {attn_type}")

        assert (
            attn_type != "flash2" or is_flash_attn_2_available()
        ), "Flash Attention 2 is not available. Missing package? Unsupported Hardware?"

        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_type = attn_type

        assert d_model % num_heads == 0, "d_model must be evenly divisible by num_heads"

        # The dimension of each head.
        self.d_head = d_model // num_heads

        # We scale the attention scores by the inverse-square-root of the head dimension
        # this shifts the temerature of softmax.
        self.dot_product_scale = 1.0 / math.sqrt(self.d_head)

        # Input projection matricies: K, K, V
        self.query_linear = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.key_linear = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.value_linear = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.output_linear = nn.Linear(self.d_model, self.d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, attn_type={self.attn_type}"

    # Project QKV input through input matrices, reshape to (batch_size, n_heads, seq_len, d_model), and apply cache.
    def _project_input(self, qkv):
        batch_size, seq_len, d_embed = qkv.shape
        query, key, value = (
            self.query_linear(qkv),
            self.key_linear(qkv),
            self.value_linear(qkv),
        )

        # Split projections into multiple heads and swap position of sequence / heads dimension
        query = query.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            1, 2
        )
        key = key.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            1, 2
        )

        return query, key, value

    def forward(self, qkv: FloatTensor, **kwargs) -> FloatTensor:
        if self.attn_type == "flash2":
            return self._flash2_forward(qkv)

        # qkv: (batch_size, seq_len, d_embed)
        batch_size, seq_len, d_embed = qkv.shape

        # Feed the inputs through the K, Q, V matrices.
        query, key, value = self._project_input(qkv)
        kv_seq_len = key.shape[-2]

        # https://github.com/pytorch/pytorch/issues/112577

        if self.use_torch:
            with sdpa_kernel(self.torch_backend):
                attended_values = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=(seq_len > 1),
                    scale=self.dot_product_scale,
                )
        # "native" scaled-dot-product attention implementation.
        else:
            # Compute attention scores
            scores = torch.matmul(query, key.transpose(-2, -1)) * self.dot_product_scale

            # Mask future positions from the past
            if seq_len > 1:
                scores.masked_fill_(
                    torch.tril(
                        torch.ones(
                            seq_len, kv_seq_len, dtype=torch.bool, device=qkv.device
                        ),
                        diagonal=0,
                    ).logical_not(),
                    float("-inf"),
                )

            # Calculate the attention weights; avoid NANs that might emerge from zeros in softmax's denominator
            attentions = self.dropout(torch.softmax(scores, dim=-1).clamp(min=1e-10))

            # Use the attention weights to get a weighted combination of value vectors
            attended_values = torch.matmul(attentions, value)

        # Concatenate attention heads and project to original embedding size using the output linear layer
        attended_values = (
            attended_values.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, d_embed)
        )

        # Project the concatenated output through the output matrix.
        return self.output_linear(attended_values)

    def _flash2_forward(
        self,
        qkv,
    ):
        batch_size, seq_len, d_embed = qkv.shape

        # Feed the inputs through the K, Q, V matrices.
        query, key, value = self._project_input(qkv)

        # Expected inputs to flash2:
        # q: (batch_size, seqlen, nheads, headdim)
        # k: (batch_size, seqlen, nheads_k, headdim)
        # v: (batch_size, seqlen, nheads_k, headdim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attended_values = flash_attn_func(
            q=query,
            k=key,
            v=value,
            dropout_p=self.dropout.p if self.training else 0.0,
            softmax_scale=self.dot_product_scale,
            causal=True,
        )
        # attended_values: (batch_size, seqlen, nheads, headdim)

        # Concatentate heads back into d_embed
        attended_values = attended_values.view(batch_size, seq_len, d_embed)

        # Project the concatenated output through the output matrix.
        return self.output_linear(attended_values)
