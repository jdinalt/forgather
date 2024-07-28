# See: https://huggingface.co/docs/transformers/custom_models
from typing import Optional, Tuple, Callable
from abc import abstractmethod

from torch import nn, Tensor, LongTensor, FloatTensor
from transformers.modeling_outputs import CausalLMOutput
from transformers import PreTrainedModel, PretrainedConfig

class CasualLM(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.loss_fn = self.loss_fn_factory()
        self.input_encoder = self.input_encoder_factory()
        self.output_decoder = self.output_decoder_factory()
        self.layer_stack = self.layer_stack_factory()
        self.post_init()

    @abstractmethod
    def loss_fn_factory(self) -> Callable:
        """
        loss_fn(logits: FloatTensor, labels: LongTensor) -> FloatTensor
        """
        ...

    @abstractmethod
    def input_encoder_factory(self) -> nn.Module:
        """
        input_encoder(input_ids: LongTensor, position_ids: LongTensor) -> FloatTensor
        """
        ...

    @abstractmethod
    def output_decoder_factory(self) -> nn.Module:
        """
        output_decoder(hidden_states: FloatTensor) -> FloatTensor
        """
        ...

    @abstractmethod
    def layer_stack_factory(self) -> nn.Module:
        """
        layer_stack(hidden_states: FloatTensor, attention_mask: Optional[FloatTensor], **kwargs) -> FloatTensor
        """
        ...

    def forward(
        self,
        input_ids: LongTensor,
        labels: Optional[LongTensor] = None,
        position_ids: Optional[LongTensor] = None,
        attention_mask: Optional[FloatTensor] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> CausalLMOutput | Tuple[FloatTensor, dict[str, FloatTensor]] | FloatTensor:
        # Convert input_ids to embeddings and add positional information.
        hidden_states = self.input_encoder(input_ids, position_ids)

        # Pass the input through each of the layers.
        hidden_states = self.layer_stack(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Convert embeddings to log-probabilities of next token-id
        logits = self.output_decoder(hidden_states)

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
