from typing import Optional, Tuple, Union
import math

import torch
from torch import nn, Tensor
import torch.nn.init as init
from torch.nn import functional as F

from transformers.modeling_outputs import CausalLMOutput
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def causal_loss(logits, labels):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        # labels with this value are ignored when computing loss
        ignore_index=-100,
        reduction="mean",
    )

    return loss.nan_to_num().unsqueeze(0)


def causal_alpha(n_layers):
    return (2.0 * n_layers) ** 0.25


def causal_beta(n_layers):
    return (8.0 * n_layers) ** -0.25


class FeedforwardLayer(nn.Module):
    def __init__(self, d_model, d_feedforward, n_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.beta = causal_beta(n_layers)

        self.linear1 = nn.Linear(self.d_model, self.d_feedforward)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(self.d_feedforward, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def reset_parameters(self):
        # Deepnet initialization
        # https://arxiv.org/pdf/2203.00555.pdf
        init.xavier_uniform_(self.linear1.weight, gain=self.beta)
        init.constant_(self.linear1.bias, 0.0)
        init.xavier_uniform_(self.linear2.weight, gain=self.beta)
        init.constant_(self.linear2.bias, 0.0)


def alibi_biases(query_len, key_len, device="cpu"):
    x = torch.arange(key_len, device=device)[None, :]
    y = torch.arange(query_len, device=device)[:, None]
    return x - y


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        n_layers,
        dropout=0.1,
        # Set to False to disable Flash-Attention-2
        flash_attention=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.beta = causal_beta(n_layers)
        self.flash_attention = flash_attention

        assert d_model % num_heads == 0, "d_model must be evenly divisible by num_heads"

        # The dimension of each head.
        self.d_head = d_model // num_heads

        # We scale the attention scores by the inverse-square-root of the head dimension
        # this shifts the temerature of softmax.
        self.dot_product_scale = 1.0 / math.sqrt(self.d_head)

        self.in_proj = nn.Parameter(torch.zeros(3 * self.d_model, self.d_model))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * self.d_model))
        self.output_linear = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)

        # Use ALiBi relative positional encoding
        # https://arxiv.org/pdf/2108.12409.pdf
        # This is the original ALiBi distribution.
        alibi_slopes = 1.0 / torch.logspace(
            1, 8, self.num_heads, base=2, dtype=torch.float
        )
        self.alibi_slopes = nn.Parameter(alibi_slopes)
        # self.register_buffer('alibi_slopes', alibi_slopes)
        self.reset_parameters()

    def project_input(self, qkv):
        proj = F.linear(qkv, self.in_proj, self.in_proj_bias)
        return proj.chunk(chunks=3, dim=-1)

    def forward(self, qkv):
        if self.flash_attention:
            return self.flash_forward(qkv)
        # qkv: (batch_size, seq_len, d_qkv)
        batch_size, seq_len, d_qkv = qkv.shape

        # Feed the inputs through the K, Q, V matrices.
        query, key, value = self.project_input(qkv)

        # Split projections into multiple heads and swap position of sequence / heads dimension
        query = query.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            1, 2
        )
        key = key.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            1, 2
        )

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.dot_product_scale

        # Apply Alibi relative positional weights.
        scores += alibi_biases(
            scores.shape[-2], scores.shape[-1], device=scores.device
        ) * self.alibi_slopes.view(-1, 1, 1)

        # Mask future positions from the past
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), True, device=qkv.device), diagonal=1
        )
        scores.masked_fill_(causal_mask, float("-inf"))

        # Calculate the attention weights; avoid NANs that might emerge from zeros in softmax's denominator
        attention_weights = self.dropout(torch.softmax(scores, dim=-1).clamp(min=1e-10))

        # Use the attention weights to get a weighted combination of value vectors
        attended_values = torch.matmul(attention_weights, value)

        # Concatenate attention heads and project to original embedding size using the output linear layer
        attended_values = (
            attended_values.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, d_qkv)
        )

        # Project the concatenated output through the output matrix.
        output = self.output_linear(attended_values)

        return output

    def flash_forward(self, qkv):
        batch_size, seq_len, d_embed = qkv.shape

        # Feed the inputs through the K, Q, V matrices.
        # query : (batch_size, seq_len, d_model)
        # qkv : (batch_size, seq_len, 3, num_heads, d_kq)
        qkv = F.linear(qkv, self.in_proj, self.in_proj_bias).unflatten(
            -1, (3, self.num_heads, self.d_head)
        )

        attended_values = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.dropout.p,
            softmax_scale=self.dot_product_scale,
            causal=True,
            alibi_slopes=self.alibi_slopes.float(),
        )
        del qkv
        # attended_values: (batch_size, seqlen, nheads, headdim)

        # Concatentate heads back into d_embed
        attended_values = attended_values.view(batch_size, seq_len, d_embed)

        # Munge the concatenated values
        output = self.output_linear(attended_values)

        return output

    def reset_parameters(self):
        # Deepnet initialization
        # https://arxiv.org/pdf/2203.00555.pdf

        q, k, v = self.in_proj.chunk(3)
        init.xavier_uniform_(q, gain=1.0)
        init.xavier_uniform_(k, gain=1.0)
        init.xavier_uniform_(v, gain=self.beta)
        init.constant_(self.in_proj_bias, 0.0)
        init.xavier_uniform_(self.output_linear.weight, gain=self.beta)
        init.constant_(self.output_linear.bias, 0.0)


class ScaleAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        n_layers,
        dropout=0.1,
        # Set to False to disable Flash-Attention-2
        flash_attention=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.beta = causal_beta(n_layers)
        self.flash_attention = flash_attention

        assert d_model % num_heads == 0, "d_model must be evenly divisible by num_heads"

        # The dimension of each head.
        self.d_head = d_model // num_heads

        # We scale the attention scores by the inverse-square-root of the head dimension
        # this shifts the temerature of softmax.
        self.dot_product_scale = 1.0 / math.sqrt(self.d_head)

        self.query = nn.Parameter(torch.empty(d_model))
        # self.query_linear = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Parameter(torch.empty(d_model))
        self.value = nn.Parameter(torch.empty(d_model))
        self.output = nn.Parameter(torch.empty(d_model))
        # self.output_linear = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)

        # Use ALiBi relative positional encoding
        # https://arxiv.org/pdf/2108.12409.pdf
        # This is the original ALiBi distribution.
        alibi_slopes = 1.0 / torch.logspace(
            1, 8, self.num_heads, base=2, dtype=torch.float
        )
        self.alibi_slopes = nn.Parameter(alibi_slopes)
        # self.register_buffer('alibi_slopes', alibi_slopes)
        self.reset_parameters()

    def project_input(self, qkv):
        query = qkv * self.query
        # query = self.query_linear(qkv)
        key = (qkv * self.key).roll(shifts=self.d_head // 2, dims=-1)
        value = qkv * self.value

        return query, key, value

    def forward(self, qkv):
        if self.flash_attention:
            return self.flash_forward(qkv)
        # qkv: (batch_size, seq_len, d_qkv)
        batch_size, seq_len, d_qkv = qkv.shape

        # Feed the inputs through the K, Q, V matrices.
        query, key, value = self.project_input(qkv)

        # Split projections into multiple heads and swap position of sequence / heads dimension
        query = query.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            1, 2
        )
        key = key.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            1, 2
        )

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.dot_product_scale

        # Apply Alibi relative positional weights.
        scores += alibi_biases(
            scores.shape[-2], scores.shape[-1], device=scores.device
        ) * self.alibi_slopes.view(-1, 1, 1)

        # Mask future positions from the past
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), True, device=qkv.device), diagonal=1
        )
        scores.masked_fill_(causal_mask, float("-inf"))

        # Calculate the attention weights; avoid NANs that might emerge from zeros in softmax's denominator
        attention_weights = self.dropout(torch.softmax(scores, dim=-1).clamp(min=1e-10))

        # Use the attention weights to get a weighted combination of value vectors
        attended_values = torch.matmul(attention_weights, value)

        # Concatenate attention heads and project to original embedding size using the output linear layer
        attended_values = (
            attended_values.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, d_qkv)
        )

        # Project the concatenated output through the output matrix.
        # output = self.output_linear(attended_values)
        output = attended_values * self.output

        return output

    def flash_forward(self, qkv):
        batch_size, seq_len, d_embed = qkv.shape

        query, key, value = self.project_input(qkv)
        query = query.view(batch_size, seq_len, self.num_heads, self.d_head)
        key = key.view(batch_size, seq_len, self.num_heads, self.d_head)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_head)

        attended_values = flash_attn_func(
            q=query,
            k=key,
            v=value,
            dropout_p=self.dropout.p,
            softmax_scale=self.dot_product_scale,
            causal=True,
            alibi_slopes=self.alibi_slopes.float(),
        )
        del qkv
        # attended_values: (batch_size, seqlen, nheads, headdim)

        # Concatentate heads back into d_embed
        attended_values = attended_values.view(batch_size, seq_len, d_embed)

        # Munge the concatenated values
        output = self.output_linear(attended_values)

        return output

    def reset_parameters(self):
        init.normal_(self.query)
        # init.xavier_uniform_(self.query_linear.weight, gain=self.beta)
        # init.constant_(self.query_linear.bias, 0.)

        init.normal_(self.key)
        init.normal_(self.value)
        init.normal_(self.output)

        # init.xavier_uniform_(self.output_linear.weight, gain=self.beta)
        # init.constant_(self.output_linear.bias, 0.)


# Deepnet transformer layer
class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        attention,
        feedforward,
        n_layers,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.attention = attention
        self.feedforward = feedforward
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)
        # Deepnet alpha https://arxiv.org/pdf/2203.00555.pdf
        self.alpha = (n_layers * 2.0) ** 0.25

    def forward(self, x):
        # Keep input as residual
        residual = x * self.alpha

        # Compute attention
        x = self.attention(x)

        # Add attention with residual and normalize.
        x = self.norm1(residual + self.dropout(x))

        # Keep output as next residual.
        residual = x * self.alpha

        # Pass through feedforward network.
        x = self.feedforward(x)

        # Combine residual and ff output, then normalize again.
        x = self.norm2(residual + self.dropout(x))

        return x


# Huggingface model type string
model_type = "dinalt-causal-transformer"


# Huggingface config class.
# Huggingface 'PreTrainedModel' objects are passed a derivative of this class
# when constructed. This is required, if your model will derive from PreTrainedModel.
class CausalTransformerConfig(PretrainedConfig):
    model_type = model_type

    def __init__(
        # All of these MUST have defaults, even if unused.
        self,
        vocab_size=2000,
        hidden_size=256,
        max_sequence_length=2048,
        dim_feedforward=512,
        num_attention_heads=1,
        num_hidden_layers=4,
        **kwargs,
    ):
        # These are the canonical names used by Huggingface
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.dim_feedforward = dim_feedforward
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

        super().__init__(**kwargs)


# The formward method of this model is designed to be compatible with the HuggingFace Trainer and Tokenizer classes.
# This is essentially a wrapper for a Pytorch transformer model, which implements the HF API.
class CausalTransformer(PreTrainedModel):
    config_class = CausalTransformerConfig
    model_type = "Transformer"

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.d_model = config.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=config.hidden_size,
                    attention=MultiheadAttention(
                        d_model=config.hidden_size,
                        num_heads=config.num_attention_heads,
                        n_layers=config.num_hidden_layers,
                        flash_attention=config.flash_attention,
                    ),
                    feedforward=FeedforwardLayer(
                        d_model=config.hidden_size,
                        d_feedforward=config.dim_feedforward,
                        n_layers=config.num_hidden_layers,
                    ),
                    n_layers=config.num_hidden_layers,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        self.reset_parameters()
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> (Tensor, dict[str, Tensor]):

        # Convert input_ids to embeddings and add positional information.
        x = self.embedding(input_ids) * self.d_model**0.5

        # Pass the input through each of the layers.
        for layer in self.layers:
            x = layer(x)

        # Convert embeddings to log-probabilities of next token-id
        logits = self.output_projection(x)

        # Compute loss.
        if labels is not None:
            loss = causal_loss(logits, labels)
        else:
            loss = None

        if return_dict:
            return CausalLMOutput(loss=loss, logits=logits)
        elif loss is not None:
            return (loss, logits)
        else:
            return logits

    def reset_parameters(self):
        # Init the embedding weights as per original design.
        init.normal_(self.embedding.weight, std=self.d_model**-0.5)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return model_inputs


AutoConfig.register(model_type, CausalTransformerConfig)
AutoModelForCausalLM.register(CausalTransformerConfig, CausalTransformer)
