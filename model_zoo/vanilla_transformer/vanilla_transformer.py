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


# We will abstract out the causal loss function for reuse.
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

    return loss.nan_to_num()


class FeedforwardLayer(nn.Module):
    def __init__(self, d_model, d_feedforward):
        super().__init__()
        self.d_model = d_model
        self.d_feedforward = d_feedforward

        self.linear1 = nn.Linear(self.d_model, self.d_feedforward)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(self.d_feedforward, self.d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
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
        self.query_linear = nn.Linear(self.d_model, self.d_model)
        self.key_linear = nn.Linear(self.d_model, self.d_model)
        self.value_linear = nn.Linear(self.d_model, self.d_model)

        # Output projection matrix:
        # The input and output matrices only make sense with multi-head
        # Don't bother with the output matrix, with a single head.
        if self.num_heads != 1:
            self.output_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, qkv):
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
        attention_weights = torch.softmax(scores, dim=-1).clamp(min=1e-10)

        # Use the attention weights to get a weighted combination of value vectors
        attended_values = torch.matmul(attention_weights, value)

        # Concatenate attention heads and project to original embedding size using the output linear layer
        attended_values = (
            attended_values.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, d_qkv)
        )

        # Project the concatenated output through the output matrix.
        if self.num_heads != 1:
            output = self.output_linear(attended_values)
        else:
            output = attended_values

        return output


# Standard transformer layer, from original paper.
class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        attention,
        feedforward,
    ):
        super().__init__()
        self.d_model = d_model
        self.attention = attention
        self.feedforward = feedforward
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        # Keep input as residual
        residual = x

        # Compute attention
        x = self.attention(x)

        # Add attention with residual and normalize.
        x = self.norm1(residual + x)

        # Keep output as next residual.
        residual = x

        # Pass through feedforward network.
        x = self.feedforward(x)

        # Combine residual and ff output, then normalize again.
        x = self.norm2(residual + x)

        return x


# A vanilla positional encoder
class PositionalEncoder(nn.Module):
    def __init__(self, d_embed, max_seq):
        super().__init__()
        self.d_embed = d_embed
        self.max_seq = max_seq

        weight = torch.zeros(max_seq, d_embed)
        position = torch.arange(0, max_seq, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed)
        )
        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)
        weight = weight.unsqueeze(0)
        self.register_buffer("weight", weight)

    def forward(self, x):
        seq_len = x.size(-2)
        return x + self.weight[:, :seq_len]


# Huggingface model type string
model_type = "simple-causal-transformer"


# Huggingface config class.
# Huggingface 'PreTrainedModel' objects are passed a derivative of this class
# when constructed. This is required, if your model will derive from PreTrainedModel.
class VanillaTransformerConfig(PretrainedConfig):
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
class VanillaTransformer(PreTrainedModel):
    config_class = VanillaTransformerConfig
    model_type = "Transformer"

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.d_model = config.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoder = PositionalEncoder(
            d_embed=config.hidden_size, max_seq=config.max_sequence_length
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=config.hidden_size,
                    attention=MultiheadAttention(
                        d_model=config.hidden_size,
                        num_heads=config.num_attention_heads,
                    ),
                    feedforward=FeedforwardLayer(
                        d_model=config.hidden_size,
                        d_feedforward=config.dim_feedforward,
                    ),
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
        x = self.positional_encoder(self.embedding(input_ids) * self.d_model**0.5)

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


AutoConfig.register(model_type, VanillaTransformerConfig)
AutoModelForCausalLM.register(VanillaTransformerConfig, VanillaTransformer)
