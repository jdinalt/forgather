from typing import Optional, Callable

from torch import nn, Tensor, FloatTensor


class LayerStack(nn.Module):
    """
    A sequential stack of L idential layers

    layer_factory: Callable, which returns a new layers instance.
    num_hidden_layers: The number of layers to construct.
    post_norm_factory [optional]: When using a model with pre-layer norm, it
        can be helpful to add an attional normalization layer before the prediction head.
        If not None, the provided callable will be used to construct such a norm layer.
    """

    def __init__(
        self,
        layer_factory: Callable,
        num_hidden_layers: int,
        *,
        post_norm_factory: Optional[Callable] = None,
    ):
        super().__init__()

        # Use module dict, rather than list, as this results in
        # the module path names remaining static when sharding the model.
        self.layers = nn.ModuleList(
            [layer_factory() for layer_idx in range(num_hidden_layers)]
        )
        self.layer_norm = None
        if post_norm_factory is not None:
            self.layer_norm = post_norm_factory()

    def forward(
        self,
        hidden_states: FloatTensor,
        **kwargs,
    ) -> FloatTensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        if self.layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        return hidden_states
