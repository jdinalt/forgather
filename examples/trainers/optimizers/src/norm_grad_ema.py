import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from .subspace_proj import SubspaceProjector

class NormGradEMA(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
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
        if grad.is_sparse:
            raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

        state = self.state[p]
        
        if "step" not in state:
            state["step"] = 0

        state["step"] += 1
        norm_grad = grad / (grad.square().mean().sqrt() + group["eps"])
        p.add_(norm_grad, alpha=(-group["lr"]))

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        # Add weight decay at the end (fixed version)
        if group["weight_decay"] > 0.0:
            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

    def update_projection(self, group, p):
        grad = p.grad
        state = self.state[p]
        
        if "step" not in state:
            state["step"] = 0

        beta1, beta2 = group["betas"]
        
        # Projection
        if "projector" not in state:
            projector = state["projector"] = SubspaceProjector(
                grad,
                **group["proj_args"],
            )
            exp_avg_sq = state["exp_avg_sq"] = torch.zeros(group["proj_args"]["rank"], device=grad.device, dtype=grad.dtype)
        else:
            projector = state["projector"]
            projector.update(grad)
            exp_avg_sq = state["exp_avg_sq"]
        
        grad = projector.down(grad)

        # Rows
        S = grad.square().mean(dim=1)

        state["step"] += 1
        beta2t = min(1.0 - math.pow(state["step"], -0.8), 0.999)
        exp_avg_sq.mul_(beta2t).add_(S, alpha=(1.0 - beta2t))

        norm_grad = grad / (exp_avg_sq.sqrt().view(-1, 1) + group["eps"])
        
        # Project up
        norm_grad = projector.up(norm_grad)

        norm_grad.div_((norm_grad.square().mean().sqrt() /1.0).clamp_(min=1.0))
        p.add_(norm_grad, alpha=(-group["lr"]))

        if group["weight_decay"] > 0.0:
            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))


