import math
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer


class SGD(Optimizer):
    """
    Template Optimizer: Minimal SGD Implementation
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
    ):
        defaults = dict(
            lr=lr,
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

                p.add_(grad, alpha=-group["lr"])

        return loss
