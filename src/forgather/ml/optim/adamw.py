import math
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    Adam
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        torch_compile: bool = False,
    ):
        self.compile = torch_compile
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def _init_state(self, state, group, p, grad):
        state["step"] = torch.tensor(0.0, dtype=torch.float32)
        state["m"] = torch.zeros_like(grad)
        state["v"] = torch.zeros_like(grad)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        with torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]

                    # Init state
                    if "step" not in state:
                        self._init_state(state, group, p, grad)

                    state["step"] += 1
                    betas = group["betas"]

                    args = [
                        p,
                        grad,
                        state["step"],
                        state["m"],
                        state["v"],
                        group["lr"],
                        betas[0],
                        betas[1],
                        group["eps"],
                        group["weight_decay"],
                    ]
                    if self.compile:
                        torch.compile(_adam, fullgraph=True, dynamic=False)(*args)
                    else:
                        _adam(*args)

        return loss


def _adam(
    p: Tensor,
    grad: Tensor,
    step: Tensor,
    m: Tensor,
    v: Tensor,
    alpha: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
):
    """
    DECOUPLED WEIGHT DECAY REGULARIZATION
    https://arxiv.org/pdf/1711.05101
    """
    if weight_decay > 0.0:
        p.add_(p, alpha=(-alpha * weight_decay))

    """
    https://arxiv.org/pdf/1412.6980
    """
    m.lerp_(grad, 1.0 - beta1)
    v.lerp_(grad.square(), 1.0 - beta2)
    update = m / (v.sqrt() + eps)

    # Bias correction
    alpha = alpha * torch.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)

    if p.dtype == update.dtype:
        p.add_(update, alpha=-alpha)
    else:
        # p -= (alpha * update).to(dtype=p.dtype)
        p.copy_(p.float() - alpha * update)
