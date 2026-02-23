from typing import Any, Callable, Optional

import torch.utils.checkpoint as checkpoint
from torch import FloatTensor, nn


class LayerStack(nn.Module):
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
        checkpoint_kwargs: Optional[dict[str, Any]] = None,
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

        # HR PretrainedModel will set gradient_checkpointing to True,
        # if gradient_checkpointing_enable() is called and an attribute
        # named gradient_checkpointing exists.
        #
        # It will also set self._gradient_checkpointing_func, which is a partial
        # function wrapping the checkpoint args passed into the config.
        # We ignore this and use our own, as we wassume this module knows the ground-truth
        # about which args should actually be used.
        self.gradient_checkpointing = enable_checkpoint
        self.checkpoint_stride = checkpoint_stride

        self.layers = nn.ModuleDict()
        for layer_idx in range(num_hidden_layers):
            self.layers[str(layer_idx)] = layer_factory(layer_idx=layer_idx)

        self.layer_norm = None
        if post_norm_factory is not None:
            self.layer_norm = post_norm_factory()

    def extra_repr(self):
        return f"gradient_checkpointing={self.gradient_checkpointing}, checkpoint_stride={self.checkpoint_stride}"

    def forward(
        self,
        hidden_states: FloatTensor,
        **kwargs,
    ) -> FloatTensor:
        if self.gradient_checkpointing:
            for i, layer in self.layers.items():
                i = int(i)
                if i % self.checkpoint_stride == 0:
                    hidden_states = checkpoint.checkpoint(
                        layer,
                        hidden_states,
                        **kwargs,  # type: ignore[arg-type]
                        **self.checkpoint_kwargs,
                    )
                else:
                    hidden_states = layer(hidden_states, **kwargs)
        else:
            for i, layer in self.layers.items():
                i = int(i)
                hidden_states = layer(hidden_states, **kwargs)

        if self.layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        return hidden_states
