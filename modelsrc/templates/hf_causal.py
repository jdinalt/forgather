# See: https://huggingface.co/docs/transformers/custom_models
# This is a template model, with the details filled-in by the code-generator.
# If you are looking at the post-processed template, it is best not to edit it
# directly, but to regenerate it with the Forgather code generator.
from functools import partial
from typing import Any, Optional, Tuple, Union, override

import torch
from torch import FloatTensor, LongTensor, Tensor, nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

# Begin auto-generated imports
-- for module, name in imports:
from {{module}} import {{ name }}

-- endfor
# End auto-generated imports

# Common definition. The same value must be set in the "model_type" field in the config class and
# in AutoConfig.register().
model_type = "{{ model_type }}"

class DynamicCausalLMConfig(PretrainedConfig):
    model_type = model_type
    keys_to_ignore_at_inference = ["past_key_values"]
    use_cache = {{ use_cache | default(True) }}
    base_model_tp_plan = {{ base_model_tp_plan | default(None) }}
    base_model_pp_plan = {{ base_model_pp_plan | default(None) }}

class ProxyModuleList(nn.ModuleList):
    """
    Proxy class for making a nn.ModuleDict look like a nn.ModuleList

    This is needed for vLLM pipeline parallel support. See full comments in DynamicCasualLM
    """
    def __init__(self, module_dict: nn.ModuleDict):
        super().__init__([layer for layer in module_dict.values()])
        self.__dict__["_module_dict"] = module_dict
    
    def __setitem__(self, index, value):
        super().__setitem__(index, value)

        # vLLM "deletes" layers which don't belong to a rank by replacing them with a 
        # sub-class of nn.Identity. Our nn.ModuleDict expects these layers to be deleted.
        # We update the proxy object and perform the deletion on the real object.
        self._module_dict[str(index)] = value

    def __delitem__(self, index):
        raise NotImplementedError("ProxyModuleList does not support deletion")

    def insert(self, index, value):
        raise NotImplementedError("ProxyModuleList does not support insertion")

    
# PreTrainedModel: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
class DynamicCasualLM(GenerationMixin, PreTrainedModel):
    # Most of these values are documented in modeling_utils.py, linked above.
    # These variables are defined by the "model_code_generator" section.
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

    """
    This is a non-standard API. It signifies that the model accepts a `return_hidden_states`` argument to
    the `forward()` method. When set, the model bypasses the language head and returns the hidden-states.
    This is exists for lack of a standard API to do this and we need this functionality to implement
    fusing the computation of the logits with the loss, using an external loss wrapper.
    
    This functionality is very important when working with newer models, which often have a HUGE vocabulary.
    The result is an explosion in peak memory when computing softmax over the logits.

    If there is ever a standard interface for this, we would preferentially switch to it.
    """
    can_return_hidden_states = True

    # Note: PreTrainedModel.post_init() will search all modules for a _tp_plan attribute and
    # merge these dictionaries with the base_model_tp_plan, defined in the configuration. It's most possible
    # to set these values in the 
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: PretrainedConfig):
        # Virtual layers -- see comments below
        v_layers = {}
        self.__dict__["v_layers"] = v_layers
        super().__init__(config)

        model_dict = self.construct_model(
            config=config,
            attn_implementation=config._attn_implementation,
            **config.to_dict()
        )

        """
        vLLM assumes that all HF causal LMs consist of a base model, without a language head, and a 
        wrapper model, which adds a causal language head. The former is mapped to AutoModel, while the latter is
        mapped to AutoModelForCausalLM.

        We try to meet these expectations, although we just have a single class. If you ask for an AutoModel, you get
        the same model as AutoModelForCausalLM, sans language-head.

        This is most definitely a hack, but the assumptions are ill-advised to begin with. Hopeful the situation will 
        improve, but we have to work with what we have.
        """
        import traceback
        stack = traceback.extract_stack()
        self.is_base_model = any('AutoModel.from_config' in str(frame.line) for frame in stack)

        self.causal_lm = model_dict['causal_model']()
        if not self.is_base_model:
            # vLLM expects the output-decoder to be named "lm_head"
            self.lm_head = model_dict['lm_head']()
            setattr(self.lm_head, "init_prefix", "lm_head")
            self.loss_function = model_dict['loss_fn']
        else:
            self.lm_head = None
        
        self.post_init()
        """
        The 'pp_plan', which is the HF/vLLM API for specifying how to convert a model to Pipeline Parallel,
        consists of a Dict[str, Tuple[List[str], str]]
        e.g.
            pp_plan = {
                # Module name
                "causal_lm.layer_stack.layers": (
                    # Inputs
                    ["hidden_states", "attention_mask"],
                    # Outputs
                    ["hidden_states"],
                ),
                ...
            }
        
        Unfortunately, it assumes that module name is an attribute of the top-level-module, and not
        a FQN. When the actual layers are in sub-modules, this complicates things. Further, it assumes that
        there is exactly one instance of a nn.ModuleList, which contains the layers. As we use a
        nn.ModuleDict, needed for Torch PP support, this adds another complication.

        For each of the modules which don't belong to a rank, vLLM replaces that module with
        an instance of PPMissingLayer, which is a subclass of nn.Identity. It acts like 
        Identity, but has logic to return only the first element from returned tuples or the first value
        from a returned dictionary.

        As to support this hot-mess of an "API," we create a dictionary of virtual layers, v_layers, where the
        keys are the full FQNs of the target modules. When an attribute lookup "miss" occurs, it is dispatched to
        self.__getattr__(), where we check if that key exists. If it does not, we hand the call off to our parent, 
        if not, we perform special handling of the access.

        For modules which are not a nn.ModuleDict, we just proxy the get and set calls to the correct parent modules.

        In the case of nn.ModuleDict, we create a proxy object, which is a sub-class of nn.ModuleList; this is what
        vLLM is looking for. Modifications to the proxy nn.ModuleList are mapped to modification to the nn.ModuleDict.

        This appears to be sufficient for our models to work with vLLM in pipeline-parallel mode, but this interface
        is horrifically brittle. When I have a few spare cycles, I'll see if I can work with the vLLM maintainers to
        support a cleaner interface.
        """
        for fqn in self._pp_plan.keys():
            split_fqn = fqn.split('.')
            module_name = split_fqn[-1]
            parent_fqn = ".".join(split_fqn[:-1])

            sub_mod = self.get_submodule(fqn)
            proxy_mod = None
            if isinstance(sub_mod, nn.ModuleDict):
                proxy_mod = ProxyModuleList(sub_mod)
            v_layers[fqn] = dict(
                parent_fqn = parent_fqn,
                module_name = module_name,
                proxy_mod = proxy_mod,
            )

    def __getattr__(self, name):
        v_layers = self.__dict__['v_layers']
        v_layer = v_layers.get(name, None)
        if v_layer is None:
            return super().__getattr__(name)
        
        # Return a proxy, if we have one
        proxy_mod = v_layer["proxy_mod"]
        if proxy_mod is not None:
            return proxy_mod
        # Else, return the mapped module
        mod = self.get_submodule(name)
        return mod

    def __setattr__(self, name, value):
        v_layers = self.__dict__['v_layers']
        v_layer = v_layers.get(name, None)
        if v_layer is None:
            return super().__setattr__(name, value)
        parent_mod = self.get_submodule(v_layer["parent_fqn"])
        setattr(parent_mod, v_layer["module_name"], value)

    @staticmethod
    def construct_model(
    ## Expands code for constructing model here.
    -- for var, has_default, default in variables:
        {{ var }}{% if has_default %}={{ repr(default) }}{% endif %},
    -- endfor
        **kwargs
    ) -> dict[str, Any]:
        """
        Model constructor

        Both the arguments and the code in this function are generated by 
        Forgather's code generator.

        Args:
            The arguments are expanded from the config, as a dictionary, and the code is generated
            from the Forgather model configuration.

        Returns:
            A dictionary of constructed model parts.
                causal_model: A callable, which constructs an instance of the main wrapped model
                lm_head: A callable, which constructs and instance of the language head module
                loss_fn: The default loss function for the model
        """
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

        The model is expected to accept exactly one of input_ids or inputs_embeds.

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
    
    @override
    def _init_weights(self, module: torch.nn.Module):
        """ Initialize the weights of the specified module; delegated to causal_lm """
        self.causal_lm._init_weights_fn(module)
    
    @override
    def get_attn_mask_fn(self):
        return self.causal_lm.get_attn_mask_fn()
    
    @override
    def get_input_embeddings(self) -> nn.Embedding:
        return self.causal_lm.get_input_embeddings()
    
    @override
    def set_input_embeddings(self, value: nn.Embedding):
        self.causal_lm.set_input_embeddings(value)

    @override
    def get_output_embeddings(self) -> nn.Module:
        if not self.lm_head or isinstance(self.lm_head, nn.Linear):
            return self.lm_head
        else:
            return self.lm_head.get_output_embeddings()

    @override
    def set_output_embeddings(self, new_embedding: nn.Module):
        if not self.lm_head or isinstance(self.lm_head, nn.Linear):
            self.lm_head = new_embedding
        else:
            self.lm_head.set_output_embeddings(new_embedding)
    
    @override
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        self.causal_lm.resize_position_embeddings(new_num_position_embeddings)

    @override
    def get_position_embeddings(self) -> Union[nn.Embedding, tuple[nn.Embedding]]:
        return self.causal_lm.get_position_embeddings()

AutoConfig.register(model_type, DynamicCausalLMConfig)
AutoModel.register(DynamicCausalLMConfig, DynamicCasualLM)
AutoModelForCausalLM.register(DynamicCausalLMConfig, DynamicCasualLM)
