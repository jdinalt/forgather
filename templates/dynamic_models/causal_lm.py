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


class DynamicCasualLM(PreTrainedModel, GenerationMixin):
    config_class = DynamicCausalLMConfig
    model_type = model_type

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
