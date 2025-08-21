# See: https://huggingface.co/docs/transformers/custom_models
# This is a template model, with the details filled-in by the code-generator.
from typing import Optional, Tuple

from functools import partial
from torch import nn, Tensor, LongTensor, FloatTensor
import torch
from transformers.modeling_outputs import CausalLMOutput
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
)

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
    _supports_sdpa = {{ supports_sdpa | default(True) }}
    _supports_flex_attn = {{ supports_flex_attn | default(False) }}
    _can_compile_fullgraph = {{ can_compile_fullgraph | default(False) }}
    _tp_plan = {{ tp_plan | default(None) }}
    _tp_size = {{ tp_size | default(None) }}
    _pp_plan = {{ pp_plan | default(None) }}
    _supports_attention_backend = {{ supports_attention_backend | default(False) }}
    _can_record_outputs = {{ can_record_outputs | default(None) }}

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.causal_lm = self.construct_model(**config.to_dict())
        if "torch_dtype" in config:
            self.to(config.torch_dtype)

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
        input_ids: LongTensor,
        labels: Optional[LongTensor] = None,
        position_ids: Optional[LongTensor] = None,
        attention_mask: Optional[FloatTensor] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> CausalLMOutput | Tuple[FloatTensor, dict[str, FloatTensor]] | FloatTensor:

        outputs = self.causal_lm(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Return type depends on arguments.
        if return_dict:
            return CausalLMOutput(**outputs)
        elif labels is not None:
            return (outputs["loss"], outputs["logits"])
        else:
            return outputs["logits"]

    # Bare-minimum for HF text generation interface to work.
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return model_inputs


AutoConfig.register(model_type, DynamicCausalLMConfig)
AutoModelForCausalLM.register(DynamicCausalLMConfig, DynamicCasualLM)
