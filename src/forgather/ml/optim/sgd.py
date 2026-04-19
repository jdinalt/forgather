import math
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer


class SGD(Optimizer):
    """Minimal vanilla SGD optimizer.

    Applies the plain stochastic gradient descent update rule:

        ``p = p - lr * grad``

    No momentum, weight decay, or gradient clipping.  Intended as a minimal
    reference implementation and starting point for custom optimizers.  For
    production training, prefer `AdamW` or `Adafactor`.

    Parameters
    ----------
    params : iterable of Parameter
        Model parameters to optimize.
    lr : float, optional
        Learning rate.  Default is 1e-3.
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
