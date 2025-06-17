from typing import Optional, Callable

from torch import nn, Tensor, FloatTensor


class ExplicitLayerStack(nn.Module):
    """
    Like LayerStack, but the factory for each layer is specified explicilty in a list

    The version is a bit more flexible for creating "unusual" model configurations.
    """

    def __init__(
        self,
        factory_list: list[Callable],
    ):
        super().__init__()
        self.layers = nn.ModuleList([factory() for factory in factory_list])

    def forward(
        self,
        hidden_states: FloatTensor,
        **kwargs,
    ) -> FloatTensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
