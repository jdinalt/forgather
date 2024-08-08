from typing import Optional, Callable

from torch import nn, Tensor, FloatTensor


class CausalLayerStack(nn.Module):
    def __init__(
        self,
        layer_factory: Callable,
        num_hidden_layers,
        *,
        post_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_factory() for _ in range(num_hidden_layers)])
        if post_norm is not None:
            self.layers.append(post_norm)

    def forward(
        self,
        hidden_states: FloatTensor,
        *,
        attention_mask: Optional[FloatTensor] = None,
        **kwargs,
    ) -> FloatTensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
