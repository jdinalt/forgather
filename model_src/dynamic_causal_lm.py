# See: https://huggingface.co/docs/transformers/custom_models
from typing import Optional, Tuple, Callable
from abc import abstractmethod

from torch import nn, Tensor, LongTensor, FloatTensor
from transformers.modeling_outputs import CausalLMOutput
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)

from .materialize import materialize_config

model_type = "dynamic-causal-lm"


class DynamicCausalLMConfig(PretrainedConfig):
    model_type = model_type


class DynamicCasualLM(PreTrainedModel):
    config_class = DynamicCausalLMConfig
    model_type = model_type

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.causal_lm = materialize_config(config.model_definition, **config.to_dict())

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
