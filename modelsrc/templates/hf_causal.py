# See: https://huggingface.co/docs/transformers/custom_models
# This is a template model, with the details filled-in by the code-generator.
from typing import Optional, Tuple, Union

from functools import partial
from torch import nn, Tensor, LongTensor, FloatTensor
import torch
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    GenerationMixin,
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

-- for module, name in imports:
from {{ module }} import {{ name }}
-- endfor

model_type = "{{ model_type }}"


class DynamicCausalLMConfig(PretrainedConfig):
    model_type = model_type
    keys_to_ignore_at_inference = ["past_key_values"]
    use_cache = {{ use_cache | default(True) }}
    base_model_tp_plan = {{ base_model_tp_plan | default(None) }}
    base_model_pp_plan = {{ base_model_pp_plan | default(None) }}

# PreTrainedModel: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
class DynamicCasualLM(GenerationMixin, PreTrainedModel):
    config_class = DynamicCausalLMConfig
    base_model_prefix = "causal_lm"
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
    _supports_attention_backend = {{ supports_attention_backend | default(False) }}
    _can_record_outputs = {{ can_record_outputs | default(None) }}
    can_return_hidden_states = True
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        model_dict = self.construct_model(
            config=config,
            attn_implementation=config._attn_implementation,
            **config.to_dict()
        )

        # Detect if being loaded by vLLM's AutoModel (for base model without lm_head)
        # or AutoModelForCausalLM (for full model with lm_head)
        import traceback
        stack = traceback.extract_stack()
        self.is_base_model = any('AutoModel.from_config' in str(frame.line) for frame in stack)

        self.causal_lm = model_dict['causal_model']()
        if not self.is_base_model:
            # vLLM expects the output-decoder to be named "lm_head"
            self.lm_head = model_dict['lm_head']()
            self.loss_function = model_dict['loss_fn']
        else:
            self.lm_head = None
        
        self.post_init()

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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_dict: bool = None,
        return_hidden_states: bool = None,
        **kwargs,
    ):
        """Most arguments are the same as HF Llama

        TODO Details

            Args:
                return_dict: Return CausalLMOutputWithPast -- see 'Returns'
                return_hidden_states: Return hidden states. Takes precedence over return_dict

            Returns:
                This is complex, as we need to support a number of use-cases.
                - With just input_ids, it returns logits
                - We can also return CausalLMOutputWithPast, if return_dict
                - If labels are provided, we return (loss, logits), if not return_dict, else CausalLMOutputWithPast
                - If constructed with AutoModel.*, as is the case with vLLM, we return BaseModelOutputWithPast
                - If our head has been deleted, as can be the case for pipeline parallel, we return hidden_states
                - If we have been explicitly asked for hidden states, as is the case with fusing loss with
                    logits, we return hidden states. While we would like this to be the same as the vLLM case,
                    vLLm requires a tuple and Torch PP barfs if a tuple is returned.

        """

        # TODO: Allow the default to be True, then uncomment
        # if return_dict is None:
        #    return_dict = getattr(self.config, "return_dict", None)

        outputs: BaseModelOutputWithPast = self.causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        # Base model output (for wrapped model with external head)
        if self.is_base_model:
            return outputs
        # Normal, CausalLM outputs (logits, loss)
        elif not return_hidden_states and self.lm_head:
            hidden_states = outputs.last_hidden_state
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            if return_dict:
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            elif loss is not None:
                return loss, logits
            else:
                return logits
        # PP or fused logits and loss
        else:
            return outputs[0]
    
    def initialize_weights(self):
        self.causal_lm.initialize_weights()
    
    def get_attn_mask_fn(self):
        return self.causal_lm.get_attn_mask_fn()
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.causal_lm.get_input_embeddings()
    
    def set_input_embeddings(self, value: nn.Embedding):
        self.causal_lm.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        if not self.lm_head or isinstance(self.lm_head, nn.Linear):
            return self.lm_head
        else:
            return self.lm_head.get_output_embeddings()

    def set_output_embeddings(self, new_embedding: nn.Module):
        if not self.lm_head or isinstance(self.lm_head, nn.Linear):
            self.lm_head = new_embedding
        else:
            self.lm_head.set_output_embeddings(new_embedding)
    
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        self.causal_lm.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Union[nn.Embedding, tuple[nn.Embedding]]:
        return self.causal_lm.get_position_embeddings()

AutoConfig.register(model_type, DynamicCausalLMConfig)
AutoModel.register(DynamicCausalLMConfig, DynamicCasualLM)
AutoModelForCausalLM.register(DynamicCausalLMConfig, DynamicCasualLM)
