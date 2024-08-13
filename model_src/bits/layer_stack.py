from typing import Optional, Callable

from torch import nn, Tensor, FloatTensor


class LayerStack(nn.Module):
    def __init__(
        self,
        layer_factory: Callable,
        num_hidden_layers,
        *,
        post_norm_factory: Optional[Callable] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_factory() for _ in range(num_hidden_layers)])
        if post_norm_factory is not None:
            self.layers.append(post_norm_factory())

    def forward(
        self,
        hidden_states: FloatTensor,
        **kwargs,
    ) -> FloatTensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
