# See: https://huggingface.co/docs/transformers/custom_models
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn.utils import skip_init
from transformers.modeling_outputs import CausalLMOutput
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)

from .causal_loss import CausalLoss
from .sinusoidal_pe import SinusoidalPE
from .pre_ln_layer import PreLNLayer
from .causal_multihead_attn import CausalMultiheadAttn
from .feedforward_layer import FeedforwardLayer

model_type = "causal-transformer"


# https://huggingface.co/docs/transformers/main_classes/configuration
class CausalTransformerConfig(PretrainedConfig):
    model_type = model_type

    def __init__(self, **kwargs):
        default_args = dict(
            # Canonical transformer attributes
            vocab_size=2000,
            hidden_size=512,
            num_attention_heads=8,
            num_hidden_layers=6,
            # Model specific
            max_sequence_length=2048,
            dim_feedforward=2048,
            initializer_range=0.02,
            # Dropout used in Attenion Is All You Need https://arxiv.org/pdf/1706.03762
            embedding_dropout=0.1,
            layer_dropout=0.1,
            # Dropouts not present in original Transformer
            residual_dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
        )
        kwargs = default_args | kwargs
        super().__init__(**kwargs)


class CausalTransformer(PreTrainedModel):
    """
    A simple causal transformer model

    This tries to conform as closely as possible to the original archetecture,
    Attention is All You Need (https://arxiv.org/pdf/1706.03762), as closely as possible
    -- obvously excepting being causal, rather than an Encoder-Decoder model.
    """

    config_class = CausalTransformerConfig
    model_type = model_type

    def __init__(self, config):
        super().__init__(config)
        self.d_model = config.hidden_size
        self.sqrt_d_model = self.d_model**0.5
        self.loss_fn = CausalLoss()
        self.embedding = nn.Embedding(config.vocab_size, self.d_model)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.decoder = nn.Linear(self.d_model, config.vocab_size)

        self.positional_encoder = SinusoidalPE(
            d_model=self.d_model, max_sequence_length=config.max_sequence_length
        )

        self.layers = nn.Sequential(
            *(
                PreLNLayer(
                    feedforward=FeedforwardLayer(
                        d_model=self.d_model,
                        d_feedforward=self.config.dim_feedforward,
                        dropout=config.activation_dropout,
                    ),
                    attention=CausalMultiheadAttn(
                        d_model=self.d_model,
                        num_heads=config.num_attention_heads,
                        dropout=config.attention_dropout,
                    ),
                    norm1=nn.LayerNorm(self.d_model),
                    norm2=nn.LayerNorm(self.d_model),
                    dropout=config.layer_dropout,
                    residual_dropout=config.residual_dropout,
                )
                for _ in range(config.num_hidden_layers)
            )
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = False,
        **kwargs,
    ) -> CausalLMOutput | Tuple[Tensor, dict[str, Tensor]] | Tensor:
        # Convert input_ids to embeddings and add positional information.
        x = self.embedding(input_ids) * self.sqrt_d_model
        x = x + self.positional_encoder(x.size(1), position_ids=position_ids)
        x = self.embedding_dropout(x)

        # Pass the input through each of the layers.
        x = self.layers(x)

        # Convert embeddings to log-probabilities of next token-id
        logits = self.decoder(x)

        # Compute loss.
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        else:
            loss = None

        # Return type depends on arguments.
        if return_dict:
            return CausalLMOutput(loss=loss, logits=logits)
        elif loss is not None:
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

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


AutoConfig.register(model_type, CausalTransformerConfig)
AutoModelForCausalLM.register(CausalTransformerConfig, CausalTransformer)
