from typing import Optional, Callable

from torch import nn, Tensor, FloatTensor
import torch.utils.checkpoint as checkpoint

class CheckpointLayerStack(nn.Module):
    """
        This module is the same as LayerStack, except with activaiton checkpointing support.

        enable_checkpoint: Enables checkpointing
        checkpoint_stride: The number of layers between checkpoints. e.g.
            1 = checkpoint all layers
            2 = checkpoint every other layer
            3 = checkpoint every 3rd layer
            etc..
        checkpoint_kwargs: Additional kwargs to checkpoint funciton. See:
            https://docs.pytorch.org/docs/stable/checkpoint.html
    """
    def __init__(
        self,
        layer_factory: Callable,
        num_hidden_layers: int,
        *,
        enable_checkpoint: bool = True,
        checkpoint_stride: int = 1,
        checkpoint_kwargs: dict = None,
        post_norm_factory: Optional[Callable] = None,
    ):
        super().__init__()

        # Default checkpoint kwargs
        self.checkpoint_kwargs = dict(
            use_reentrant=False,
        )

        # Merge provided args, if any.
        if checkpoint_kwargs:
            self.checkpoint_kwargs |= checkpoint_kwargs
        
        self.enable_checkpoint = enable_checkpoint
        self.checkpoint_stride = checkpoint_stride
        self.layers = nn.ModuleList([
            layer_factory() for layer_idx in range(num_hidden_layers)
        ])

        self.layer_norm = None
        if post_norm_factory is not None:
            self.layer_norm = post_norm_factory()

    def forward(
        self,
        hidden_states: FloatTensor,
        **kwargs,
    ) -> FloatTensor:
        if self.enable_checkpoint:
            for i, layer in enumerate(self.layers):
                if i % self.checkpoint_stride == 0:
                    hidden_states = checkpoint.checkpoint(layer, hidden_states, **self.checkpoint_kwargs)
                else:
                    hidden_states = layer(hidden_states)
        else:
            for layer in self.layers:
                hidden_states = layer(hidden_states)
        
        if self.layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        return hidden_states
