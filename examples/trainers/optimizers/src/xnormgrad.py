import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from .subspace_proj import SubspaceProjector

class XNormGrad1(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        p: float = 0.2,
        i: float = 0.8,
        beta: float = 0.995,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        defaults = {"lr": lr, "p": p, "i": i, "beta": beta, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if "proj_args" not in group:
                    self.update_norm(group, p)
                else:
                    self.update_projection(group, p)

        return loss

    def update_norm(self, group, p):
        grad = p.grad
        state = self.state[p]
        
        if "step" not in state:
            state["step"] = 0
        state["step"] += 1

        proportional, i, beta = group["p"], group["i"], group["beta"]
        update = torch.zeros_like(grad)

        if proportional != 0:
            update += grad * proportional
        
        if i != 0:
            if "integral" not in state:
                state["integral"] = torch.zeros_like(grad)
            integral = state["integral"]
            integral.mul_(beta).add_(grad, alpha=(1.0 - beta))
            update += integral * i

        update /= (grad.square().mean().sqrt() + group["eps"])
        p.add_(update, alpha=(-group["lr"]))

        if group["weight_decay"] > 0.0:
            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

    def update_projection(self, group, p):
        grad = p.grad
        state = self.state[p]
        
        if "step" not in state:
            state["step"] = 0
        state["step"] += 1

        # Projection
        if "projector" not in state:
            projector = state["projector"] = SubspaceProjector(
                grad,
                **group["proj_args"],
            )
        else:
            projector = state["projector"]
            projector.update(grad)

        grad_proj = projector.down(grad)
        
        proportional, i, beta = group["p"], group["i"], group["beta"]
        update = torch.zeros_like(grad_proj)

        if proportional != 0:
            update += grad_proj * proportional
        
        if i != 0:
            if "integral" not in state:
                state["integral"] = torch.zeros_like(grad_proj)
            integral = state["integral"]
            integral.mul_(beta).add_(grad_proj, alpha=(1.0 - beta))
            update += integral * i

        # Project up
        update = projector.up(update)

        # Add residual to update
        update += grad - projector.up(grad_proj)

        # Normalize
        update /= (grad.square().mean().sqrt() + group["eps"])
        p.add_(update, alpha=(-group["lr"]))

        if group["weight_decay"] > 0.0:
            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))