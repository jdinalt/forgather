import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from .subspace_proj import SubspaceProjector

class XNormGrad2(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "bias": "auto", }
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
        if "bias" not in state and group["bias"] == "auto":
            if p.dim() <= 1:
                state["bias"] = "none"
            elif p.shape[0] > p.shape[1]:
                state["bias"] = "row"
            else:
                state["bias"] = "column"
        
        match group["bias"]:
            case "none":
                norm_grad = grad / (grad.square().mean().sqrt() + group["eps"])
            case "row":
                try:
                    norm_grad = grad / (grad.square().mean(dim=1).sqrt().view(-1, 1) + group["eps"])
                except:
                    print(grad.shape, p.shape)
                    raise
            case "column":
                norm_grad = grad / (grad.square().mean(dim=0).sqrt() + group["eps"])
            case _:
                raise Exception(f"Undefined bias type {group['bias']}")
        
        
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
        else:
            projector = state["projector"]
            projector.update(grad)
        
        grad = projector.down(grad)
        match projector.proj_type:
            case "right":
                S = grad.square().mean(dim=1).sqrt().view(-1, 1)
            case "left":
                S = grad.square().mean(dim=0).sqrt()
            case _:
                raise Exception("Unknown projection type")
        
        state["step"] += 1

        norm_grad = grad / (S + group["eps"])
        
        # Project up
        norm_grad = projector.up(norm_grad)
        
        p.add_(norm_grad, alpha=(-group["lr"]))

        if group["weight_decay"] > 0.0:
            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))