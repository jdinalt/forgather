import math
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

class NormGrad(Optimizer):
    """
    RMS Gradient Normalization
    """
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        eps: float = 1e-9,
        clip_threshold: float = 1.0,
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                #update = grad / ((grad.square().mean().sqrt() + group["eps"]) / group["clip_threshold"]).clamp_(min=1.0)
                update = grad / (grad.square().mean().sqrt() + group["eps"])
                p.add_(update, alpha=(-group["lr"]))

        return loss

