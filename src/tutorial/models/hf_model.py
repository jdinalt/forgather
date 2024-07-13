from typing import Optional, Tuple, Union
import math
import torch
from torch import nn, Tensor
import torch.nn.init as init
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutput
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)


# We will abstract out the causal loss function for reuse.
def causal_loss(logits, labels):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        # labels with this value are ignored when computing loss
        ignore_index=-100,
        reduction="mean",
    )

    return loss.nan_to_num()


# A MLP consists of two or more linear layers, each seperated by
# a non-linear activation function. Here, we use ReLU, which changes values
# less than zero to zero and passes values greater than zero unchanged.
#
# This is a 2-layer MLP, common to Transformer models. The rows in the first layer activate
# the corresponding columns in the second layer, where the input is matched by dot-product
# similarity to the input -- this prduces a normalized value between -1 and 1, with one being
# a perfect match and -1 being an exact opposite. This value is then offset by the corresponding bias
# parameter, with the ReLU layer blocking all inputs less-than or equal to zero.
#
# In the case where the resulting value is non-zero, the corresponding column is added to the output, in proportion
# to the magnitude of the signal.
class FeedforwardNet(nn.Module):
    def __init__(self, d_model, d_feedforward):
        super().__init__()
        self.d_model = d_model
        self.d_feedforward = d_feedforward

        self.linear1 = nn.Linear(self.d_model, self.d_feedforward)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(self.d_feedforward, self.d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


# Huggingface model type string
# This is a unique identifier for a model type, which allows the API to find the
# implementation for the type.
model_type = "simple-causal2"


# Huggingface config class.
#
# Huggingface 'PreTrainedModel' objects are passed a derivative of this class
# when constructed. This is required, if your model will derive from PreTrainedModel.
class CausalLM2Config(PretrainedConfig):
    model_type = model_type

    def __init__(
        # All of these MUST have defaults, even if unused.
        self,
        vocab_size=8000,
        hidden_size=256,
        max_sequence_length=2048,
        dim_feedforward=512,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.dim_feedforward = dim_feedforward

        super().__init__(**kwargs)


# The formward method of this model is designed to be compatible with the HuggingFace Trainer and Tokenizer classes.
# This is essentially a wrapper for a Pytorch transformer model, which implements the HF API.
class CausalLM2(PreTrainedModel):
    config_class = CausalLM2Config
    model_type = "Transformer"

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.d_model = config.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.feedforward = FeedforwardNet(self.d_model, config.dim_feedforward)
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> (Tensor, dict[str, Tensor]):

        # Convert input_ids to embeddings.
        x = self.embedding(input_ids)

        # Pass the input through the feedforward network.
        x = self.feedforward(x)

        # Convert embeddings to log-probabilities of next token-id
        logits = self.output_projection(x)

        # Compute loss.
        if labels is not None:
            loss = causal_loss(logits, labels)
        else:
            loss = None

        if return_dict:
            return CausalLMOutput(loss=loss, logits=logits)
        elif loss is not None:
            return (loss, logits)
        else:
            return logits

    # This is needed for the Huggingface text generation APIs.
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return model_inputs


AutoConfig.register(model_type, CausalLM2Config)
AutoModelForCausalLM.register(CausalLM2Config, CausalLM2)
