# See: https://huggingface.co/docs/transformers/custom_models
from typing import Optional, Tuple, Callable
from collections import OrderedDict
import math

from torch import nn, Tensor, LongTensor, FloatTensor
from torch.nn.functional import cross_entropy
import torch
from transformers.modeling_outputs import CausalLMOutput
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)

model_type = "forgather-causal-transformer"


# A basic feedforward layer
# https://arxiv.org/pdf/1706.03762
class FeedforwardLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_feedforward: int,
        *,
        activation=nn.ReLU(),
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.linear1 = nn.Linear(self.d_model, self.d_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout != 0.0 else nn.Identity()
        self.activation = activation
        self.linear2 = nn.Linear(self.d_feedforward, self.d_model, bias=bias)

    def forward(self, hidden_states: FloatTensor) -> FloatTensor:
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states

    def extra_repr(self):
        return f"d_model={self.d_model}, d_feedforward={self.d_feedforward}"


class InitWeights:
    def __init__(self, std: float):
        self.std = std

    def __call__(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def __repr__(self):
        return f"{type(self).__name__}(std={self.std})"


class LayerStack(nn.Module):
    def __init__(
        self,
        layer_factory: Callable,
        num_hidden_layers,
        *,
        post_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_factory() for _ in range(num_hidden_layers)])
        if post_norm is not None:
            self.layers.append(post_norm)

    def forward(
        self,
        hidden_states: FloatTensor,
    ) -> FloatTensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


# A simple causal multi-head-attention implementation
class CausalMultiheadAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

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

        # The purpose of the output layer is to expand the dimension of the low-rank
        # attention heads back to d_model and sum their output. With only a single head,
        # this only adds dead-weights.
        if self.num_heads > 1:
            self.output_linear = nn.Linear(self.d_model, self.d_model, bias=bias)

        if dropout == 0.0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

    def extra_repr(self):
        return f"d_model={self.d_model}, num_heads={self.num_heads}"

    def forward(self, qkv: Tensor) -> Tensor:
        # qkv: (batch_size, seq_len, d_qkv)
        batch_size, seq_len, d_qkv = qkv.shape

        # Feed the inputs through the K, Q, V matrices.
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

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.dot_product_scale

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
        if self.num_heads == 1:
            return attended_values
        else:
            return self.output_linear(attended_values)


class CausalLoss:
    def __repr__(self):
        return f"{type(self).__name__}()"

    @staticmethod
    def __call__(logits: Tensor, labels: Tensor) -> Tensor:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            # labels with this value are ignored when computing loss
            ignore_index=-100,
            reduction="mean",
        )

        return loss.nan_to_num()


# An implementation of the original transformer sinusoidal positional encoder.
# https://arxiv.org/pdf/1706.03762
class SinusoidalPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_sequence_length: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

        weight = torch.zeros(max_sequence_length, d_model)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)
        weight = weight.unsqueeze(0)
        self.register_buffer("weight", weight, persistent=False)

    def extra_repr(self):
        return f"d_model={self.d_model}, max_sequence_length={self.max_sequence_length}"

    def forward(
        self, x: FloatTensor, *, position_ids: Optional[LongTensor] = None
    ) -> Tensor:
        seq_length = x.size(1)
        if position_ids is not None:
            return x + self.weight[position_ids]
        else:
            return x + self.weight[:, :seq_length]


# Attention Is All You Need: https://arxiv.org/pdf/1706.03762
# On Layer Normalization in the Transformer Architecture: https://arxiv.org/pdf/2002.04745
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
        # Residual Dropout:A Simple Approach to Improve Transformer’s Data Efficiency
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
            self.embedding_scale = d_model**0.5
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


class CausalTransformerConfig(PretrainedConfig):
    model_type = model_type

    def __init__(self, **kwargs):
        self.hidden_size = kwargs.pop("hidden_size", 512)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 8)
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", 6)
        self.max_sequence_length = kwargs.pop("max_sequence_length", 2048)

        self.dim_feedforward = kwargs.pop("dim_feedforward", 2048)
        self.vocab_size = kwargs.pop("vocab_size", 2000)

        self.residual_dropout = kwargs.pop("residual_dropout", 0.0)
        self.layer_dropout = kwargs.pop("layer_dropout", 0.10)
        self.embedding_dropout = kwargs.pop("embedding_dropout", 0.10)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        self.initializer_range = kwargs.pop("initializer_range", 0.02)
        super().__init__(**kwargs)


class CausalTransformer(PreTrainedModel):
    config_class = CausalTransformerConfig
    model_type = model_type

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        layer_norm_factory = lambda: nn.LayerNorm(
            normalized_shape=config.hidden_size,
        )

        self.loss_fn = CausalLoss()

        self.input_encoder = InputEncoder(
            d_model=config.hidden_size,
            vocab_size=config.vocab_size,
            dropout=config.embedding_dropout,
            positional_encoder=SinusoidalPE(
                d_model=config.hidden_size,
                max_sequence_length=config.max_sequence_length,
            ),
        )

        self.output_decoder = nn.Linear(
            config.hidden_size,
            config.vocab_size,
        )

        self.layer_stack = LayerStack(
            layer_factory=lambda: PostLNLayer(
                feedforward=FeedforwardLayer(
                    d_model=config.hidden_size,
                    d_feedforward=config.dim_feedforward,
                    dropout=config.activation_dropout,
                ),
                attention=CausalMultiheadAttn(
                    d_model=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    dropout=config.attention_dropout,
                ),
                norm1=layer_norm_factory(),
                norm2=layer_norm_factory(),
                dropout=config.layer_dropout,
                residual_dropout=config.residual_dropout,
            ),
            num_hidden_layers=config.num_hidden_layers,
        )

        init_weights = InitWeights(
            std=config.initializer_range,
        )

        self.apply(init_weights)

    def forward(
        self,
        input_ids: LongTensor,
        labels: Optional[LongTensor] = None,
        position_ids: Optional[LongTensor] = None,
        attention_mask: Optional[FloatTensor] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> CausalLMOutput | Tuple[FloatTensor, dict[str, FloatTensor]] | FloatTensor:

        # Convert input_ids to embeddings and add positional information.
        hidden_states = self.input_encoder(input_ids, position_ids)

        # Pass the input through each of the layers.
        hidden_states = self.layer_stack(hidden_states)

        # Convert embeddings to log-probabilities of next token-id
        logits = self.output_decoder(hidden_states)

        # Compute loss.
        loss = self.loss_fn(logits, labels) if labels is not None else None

        # Return type depends on arguments.
        if return_dict:
            return CausalLMOutput(loss=logits, logits=logits)
        elif labels is not None:
            return (loss, logits)
        else:
            return logits

    # Bare-minimum for HF text generation interface to work.
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return model_inputs


AutoConfig.register(model_type, CausalTransformerConfig)
AutoModelForCausalLM.register(CausalTransformerConfig, CausalTransformer)
