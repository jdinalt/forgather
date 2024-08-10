from torch import nn, Tensor
import torch


class InitWeights:
    """
    Conventional transformer weight initialization.
    """

    def __init__(self, std: float):
        self.std = std

    @torch.no_grad()
    def __call__(self, module: nn.Module) -> None:
        """
        Called with the top-level module with weights as an argument.
        """
        module.apply(self.init_weights)

    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def __repr__(self):
        return f"{type(self).__name__}(std={self.std})"
