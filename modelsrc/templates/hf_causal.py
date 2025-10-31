# See: https://huggingface.co/docs/transformers/custom_models
# This is a template model, with the details filled-in by the code-generator.
from typing import Optional, Tuple

from functools import partial
from torch import nn, Tensor, LongTensor, FloatTensor
import torch
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
)
from transformers.cache_utils import Cache

-- for module, name in imports:
from {{ module }} import {{ name }}
-- endfor

model_type = "{{ model_type }}"


class DynamicCausalLMConfig(PretrainedConfig):
    model_type = model_type

# PreTrainedModel: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
class DynamicCasualLM(PreTrainedModel, GenerationMixin):
    config_class = DynamicCausalLMConfig
    model_type = model_type
    main_input_name = "{{ main_input_name | default('input_ids') }}"
    model_tags = {{ model_tags | default(None) }}
    _no_split_modules = {{ no_split_modules | default(None) }}
    _skip_keys_device_placement = {{ skip_keys_device_placement | default(None) }}
    _keep_in_fp32_modules = {{ keep_in_fp32_modules | default(None) }}
    _keep_in_fp32_modules_strict = {{ keep_in_fp32_modules_strict | default(None) }}
    _keys_to_ignore_on_load_missing = {{ keys_to_ignore_on_load_missing | default(None) }}
    _keys_to_ignore_on_load_unexpected = {{ keys_to_ignore_on_load_unexpected | default(None) }}
    _keys_to_ignore_on_save = {{ keys_to_ignore_on_save | default(None) }}
    _tied_weights_keys = {{ tied_weights_keys | default(None) }}
    is_parallelizable = {{ is_parallelizable | default(False) }}
    supports_gradient_checkpointing = {{ supports_gradient_checkpointing | default(False) }}
    _is_stateful = {{ is_stateful | default(False) }}
    _supports_flash_attn = {{ supports_flash_attn | default(False) }}
    _supports_sdpa = {{ supports_sdpa | default(False) }}
    _supports_flex_attn = {{ supports_flex_attn | default(False) }}
    _can_compile_fullgraph = {{ can_compile_fullgraph | default(False) }}
    _tp_plan = {{ tp_plan | default(None) }}
    _tp_size = {{ tp_size | default(None) }}
    _pp_plan = {{ pp_plan | default(None) }}
    _supports_attention_backend = {{ supports_attention_backend | default(False) }}
    _can_record_outputs = {{ can_record_outputs | default(None) }}

    def __init__(self, config: PretrainedConfig, attn_implementation: str=None):
        if attn_implementation:
            config._attn_implementation = attn_implementation
        super().__init__(config)
        self.causal_lm = self.construct_model(attn_implementation=config._attn_implementation, **config.to_dict())
        self.causal_lm.config = config

    @staticmethod
    def construct_model(
    ## Expands code for constructing model here.
    -- for var, has_default, default in variables:
        {{ var }}{% if has_default %}={{ repr(default) }}{% endif %},
    -- endfor
        **kwargs
    ):
        {{ definitions|indent(8) }}
        
        return {{ main_body|indent(8) }}

    def forward(
        self,
        *args,
        **kwargs,
    ):

        return self.causal_lm(
            *args,
            **kwargs,
        )

    def get_attn_mask_fn(self, *args, **kwargs):
        """
        Returns a function to create an attention mask for pipeline parallel or other external use.

        This delegates to the underlying CasualLM's create_attention_mask method,
        which uses HuggingFace's masking utilities to create the appropriate mask
        format (4D tensor for eager/sdpa, BlockMask for flex_attention).
        """
        return self.causal_lm.get_attn_mask_fn(*args, **kwargs)

AutoConfig.register(model_type, DynamicCausalLMConfig)
AutoModelForCausalLM.register(DynamicCausalLMConfig, DynamicCasualLM)
