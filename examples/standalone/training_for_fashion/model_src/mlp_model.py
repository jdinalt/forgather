from typing import Callable, Optional
import torch
from torch import nn, FloatTensor, LongTensor


# https://en.wikipedia.org/wiki/Multilayer_perceptron
class MultilayerPerceptron(torch.nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        activation_factory: Callable,
        loss_fn: Callable,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.loss_fn = loss_fn
        self.layers = nn.Sequential(
            # Converts multi-dimensional input to single dimension
            nn.Flatten(),
            nn.Linear(d_input, d_model),
            activation_factory(),
            nn.Linear(d_model, d_model),
            activation_factory(),
            nn.Linear(d_model, d_output),
        )

    def extra_repr(self):
        return (
            f"d_model={self.d_model}, d_input={self.d_input}, d_output={self.d_output}"
        )

    def forward(
        self, inputs: FloatTensor, labels: Optional[LongTensor] = None
    ) -> FloatTensor | tuple[FloatTensor, FloatTensor]:
        # Pass inputs through layers
        logits = self.layers(inputs)

        # If labels, compute and return loss with logits, else logits
        if labels is not None:
            return (self.loss_fn(logits, labels), logits)
        else:
            return logits
