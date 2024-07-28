from typing import Optional, List

from torch import nn, Tensor, FloatTensor

class CausalLayerStack(nn.Module):
    def __init__(self, layers: List[nn.Module], *, post_norm: Optional[nn.Module]=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
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