from typing import Optional, Callable, Union
from functools import partial

import torch
from torch import nn, FloatTensor

from transformers.cache_utils import DynamicCache, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


class CasualLM(nn.Module):
    """
    A causal language model with a HF compatible "forward" method and KV cache support
    """

    def __init__(
        self,
        loss_fn: Callable,
        input_encoder: Callable,
        output_decoder: Callable,
        layer_stack: Callable,
        init_weights: Callable,
        attn_mask_fn: Callable,
        config=None,
    ):
        super().__init__()
        self.config = config
        self.loss_fn = loss_fn
        self.input_encoder = input_encoder
        self.output_decoder = output_decoder
        self.layer_stack = layer_stack
        self.use_internal_mask = True
        self.attn_mask_fn = partial(
            attn_mask_fn,
            config=self.config,
            dtype=torch.get_default_dtype(),
        )
        self.init_weights = init_weights
        self.output_logits = True

    def initialize_weights(self):
        self.init_weights(self)

    def extra_repr(self):
        return f"loss_fn={self.loss_fn}"

    def get_attn_mask_fn(self):
        self.use_internal_mask = False
        return self.attn_mask_fn

    def get_input_embeddings(self) -> nn.Embedding:
        return self.input_encoder.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding):
        self.input_encoder.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        if isinstance(self.output_decoder, nn.Linear):
            return self.output_decoder
        else:
            return self.output_decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embedding: nn.Module):
        if isinstance(self.output_decoder, nn.Linear):
            self.output_decoder = new_embedding
        else:
            self.output_decoder.set_output_embeddings(new_embedding)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        self.model.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Union[nn.Embedding, tuple[nn.Embedding]]:
        return self.model.get_position_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> (
        CausalLMOutputWithPast
        | tuple[FloatTensor, dict[str, FloatTensor]]
        | FloatTensor
    ):
        """
        args:
            See https://huggingface.co/docs/transformers/main/model_doc/llama
        """
        if self.input_encoder:
            if use_cache:
                # Init cache?
                if past_key_values is None:
                    past_key_values = DynamicCache(config=self.config)
                past_seen_tokens = past_key_values.get_seq_length()
                seq_length = input_ids.shape[1]
                device = input_ids.device
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + seq_length, device=device
                )
                if position_ids is None:
                    position_ids = cache_position.unsqueeze(0)

            # Convert input_ids to embeddings and add positional information.
            hidden_states = self.input_encoder(input_ids, position_ids)

            # Only create attention_mask internally if not provided externally (for pipeline parallel)
            if self.use_internal_mask:  # and not torch.compiler.is_exporting():
                attention_mask = self.attn_mask_fn(
                    input_ids=input_ids,
                    input_embeds=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                )
        else:
            # Intermediate pipeline stage: input_ids is actually hidden_states
            hidden_states = input_ids

        if self.layer_stack:
            # Pass the input through each of the layers.
            hidden_states = self.layer_stack(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        if self.output_decoder and self.output_logits:
            # Convert embeddings to log-probabilities of next token-id
            logits = self.output_decoder(hidden_states)

            # Compute loss.
            loss = self.loss_fn(logits, labels) if labels is not None else None
            # Return type depends on arguments.
            if return_dict:
                return CausalLMOutputWithPast(
                    loss=loss, logits=logits, past_key_values=past_key_values
                )
            elif labels is not None:
                return (
                    loss,
                    logits,
                )
            else:
                return logits
        else:
            # Intermediate pipeline stage: return only hidden_states
            # The attention_mask is passed externally via extra_kwargs, not forwarded
            return hidden_states
