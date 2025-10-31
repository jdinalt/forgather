from typing import Optional, Callable

import torch
from torch import nn, Tensor, LongTensor, FloatTensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from transformers.cache_utils import DynamicCache, Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutput


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
        config=None,
    ):
        super().__init__()
        self.config = config
        self.loss_fn = loss_fn
        self.input_encoder = input_encoder
        self.output_decoder = output_decoder
        self.layer_stack = layer_stack
        self.default_dtype = torch.get_default_dtype()
        self.use_internal_mask = True
        init_weights(self)

    def extra_repr(self):
        return f"loss_fn={self.loss_fn}"

    def get_attn_mask_fn(self):
        # Don't call internal mask implementation
        self.use_internal_mask = False
        return self.create_attention_mask

    def create_attention_mask(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        """
        Create an attention mask for the model using the HuggingFace masking utilities.

        This method is designed to be called externally (e.g., from pipeline parallel code)
        to pre-compute attention masks that can be passed as extra_kwargs.

        Args:
            input_ids: Input token IDs (batch_size, seq_length)
            attention_mask: Optional 2D padding mask (batch_size, seq_length)
            position_ids: Optional position indices (batch_size, seq_length)

        Returns:
            The attention mask in the format required by the model's attention implementation
            (e.g., 4D tensor for eager/sdpa, BlockMask for flex_attention)
        """
        return self._create_attention_mask(input_ids, attention_mask, position_ids)
    
    def _create_attention_mask(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache ] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        assert self.config

        # When using SDPA, if just simple a simple causal attention mask
        # is required, bypass mask generation. SDPA will then use
        # the "is_causal" flag, which saves memory and is faster.
        if (self.config._attn_implementation == "sdpa"
            and attention_mask is None
            and past_key_values is None
        ):
            return None

        if input_embeds is None:
            # Create a dummy input_embeds tensor for shape inference
            # We only need batch_size and dtype, not actual embeddings
            batch_size, seq_length = input_ids.shape

            input_embeds = torch.empty(
                batch_size, seq_length, self.config.hidden_size,
                device=torch.device("meta"), dtype=self.default_dtype
            )

        # Convert to bool mask, if long
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dtype == torch.long:
            attention_mask = attention_mask.to(dtype=torch.bool)

        if cache_position is None:
            cache_position = torch.arange(
                0, input_ids.shape[1], device=input_ids.device
            )

        # Use HuggingFace's create_causal_mask utility
        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # debug
        #print(repr(attention_mask))
        #print(attention_mask)
        
        return attention_mask

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
    ) -> CausalLMOutput | tuple[FloatTensor, dict[str, FloatTensor]] | FloatTensor:
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
            if self.use_internal_mask and not torch.compiler.is_exporting:
                attention_mask = self._create_attention_mask(
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

        if self.output_decoder:
            # Convert embeddings to log-probabilities of next token-id
            logits = self.output_decoder(hidden_states)

            # Compute loss.
            loss = self.loss_fn(logits, labels) if labels is not None else None
            # Return type depends on arguments.
            if return_dict:
                return CausalLMOutput(loss=loss, logits=logits)
            elif labels is not None:
                return (loss, logits,)
            else:
                return logits
        else:
            # Intermediate pipeline stage: return only hidden_states
            # The attention_mask is passed externally via extra_kwargs, not forwarded
            return hidden_states
