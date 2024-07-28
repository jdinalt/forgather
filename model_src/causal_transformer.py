# See: https://huggingface.co/docs/transformers/custom_models
from typing import Optional, Tuple
import importlib
import sys

import torch
from torch import nn, Tensor
from torch.nn.utils import skip_init
from transformers.modeling_outputs import CausalLMOutput
from transformers import (
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)

from .causal_lm import CasualLM

model_type = "causal-transformer"

def dynamic_import(name):
    module_name, symbol_name = name.split(":")
    package = sys.modules[__name__].__package__
    mod = importlib.import_module(module_name, package=package)
    for symbol in symbol_name.split("."):
        mod = getattr(mod, symbol)
    return mod

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

            # Dynamic bindings
            loss_fn_cls='.causal_loss:CausalLoss',
            input_encoder_cls='.input_encoder:InputEncoder',
            output_decoder_cls='torch.nn:Linear',
            positional_encoder_cls='.sinusoidal_pe:SinusoidalPE',
            layer_stack_cls='.causal_layer_stack:CausalLayerStack',
            layer_cls='.post_ln_layer:PostLNLayer',
            layer_norm_cls='torch.nn:LayerNorm',
            attention_cls='.causal_multihead_attn:CausalMultiheadAttn',
            feedforward_cls='.feedforward_layer:FeedforwardLayer',
        )
        kwargs = default_args | kwargs
        super().__init__(**kwargs)

class CausalTransformer(CasualLM):
    """
    A simple causal transformer model

    This tries to conform as closely as possible to the original archetecture,
    Attention is All You Need (https://arxiv.org/pdf/1706.03762), as closely as possible
    -- obvously excepting being causal, rather than an Encoder-Decoder model.
    """
    config_class = CausalTransformerConfig
    model_type = model_type

    def loss_fn_factory(self):
        return dynamic_import(self.config.loss_fn_cls)()

    def input_encoder_factory(self):
        return dynamic_import(self.config.input_encoder_cls)(
            self.config.hidden_size,
            self.config.vocab_size,
            dropout=self.config.embedding_dropout,
            positional_encoder=self.positional_encoder_factory(),
        )

    def output_decoder_factory(self):
        return dynamic_import(self.config.output_decoder_cls)(
            self.config.hidden_size,
            self.config.vocab_size,
        )

    def positional_encoder_factory(self):
        return dynamic_import(self.config.positional_encoder_cls)(
            self.config.hidden_size,
            max_sequence_length=self.config.max_sequence_length,
        )

    def layer_stack_factory(self):
        return dynamic_import(self.config.layer_stack_cls)(
            [ self.layer_factory(i) for i in range(self.config.num_hidden_layers) ],
        )

    def layer_factory(self, layer):
        return dynamic_import(self.config.layer_cls)(
            feedforward=self.feedforward_factory(),
            attention=self.attention_factory(),
            norm1=self.layer_norm_factory(),
            norm2=self.layer_norm_factory(),
            dropout=self.config.layer_dropout,
            residual_dropout=self.config.residual_dropout,
        )

    def layer_norm_factory(self):
        return dynamic_import(self.config.layer_norm_cls)(
            self.config.hidden_size,
        )

    def feedforward_factory(self):
        return dynamic_import(self.config.feedforward_cls)(
            self.config.hidden_size,
            d_feedforward=self.config.dim_feedforward,
            dropout=self.config.activation_dropout,
        )

    def attention_factory(self):
        return dynamic_import(self.config.attention_cls)(
            self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.attention_dropout,
        )

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