import math
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer


class Apollo(Optimizer):
    """
    APOLLO: SGD-like Memory, AdamW-level Performance
    https://arxiv.org/pdf/2412.05270
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        rank: int = 1,
        scale: float = 1.0,
        scale_front: bool = False,
        update_steps: int = 10,
        mini: bool = False,
        projector_factory: Callable = None,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            scale=scale,
            scale_front=scale_front,
            update_steps=update_steps,
            mini=mini,
            projector_factory=projector_factory,
        )
        super().__init__(params, defaults)

    def _init_state(self, state, group, p, grad):
        rank = group["rank"]

        if grad.shape[0] < grad.shape[1]:
            dim = grad.shape[0]
            proj_shape = (rank, grad.shape[1])
            proj_type = "left"
        else:
            dim = grad.shape[1]
            proj_shape = (rank, grad.shape[0])
            proj_type = "right"

        state["projector"] = group["projector_factory"](
            rank=rank,
            dim=dim,
            proj_type=proj_type,
        )

        state["step"] = torch.tensor(0.0, dtype=torch.float32)
        state["m"] = torch.zeros(*proj_shape, device=grad.device, dtype=grad.dtype)
        state["v"] = torch.zeros(*proj_shape, device=grad.device, dtype=grad.dtype)

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

                # Init state
                if "step" not in state:
                    self._init_state(state, group, p, grad)

                projector = state["projector"]
                state["step"] += 1
                step = state["step"]
                beta1, beta2 = group["betas"]
                M, V = state["m"], state["v"]
                lr = group["lr"]
                alpha = group["scale"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                scale = group["scale"]
                scale_front = group["scale_front"]

                # Weight decay?
                if weight_decay > 0.0:
                    p.add_(p, alpha=(-alpha * weight_decay))

                # Apply bias correction to lr
                lr = lr * torch.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)

                # Update projector
                projector.step(grad)

                # Project gradient into low-rank sub-space
                R = projector.down(grad) * projector.scale

                # Update EMA of 1st and 2nd moments
                M.lerp_(R, 1.0 - beta1)
                V.lerp_(R.square(), 1.0 - beta2)
                R_tilde = M / (V.sqrt() + eps)

                if group["mini"]:
                    S = torch.linalg.norm(R_tilde) / (torch.linalg.norm(R) + 1e-8)
                else:
                    S = torch.linalg.norm(R_tilde, dim=0) / (
                        torch.linalg.norm(R, dim=0) + 1e-8
                    )
                    if grad.shape[0] >= grad.shape[1]:
                        S = S.view(-1, 1)

                update = grad * S

                if scale_front and scale != 1.0:
                    update *= math.sqrt(scale)

                # Apply Norm-Growth Limiter in Fira (https://arxiv.org/abs/2410.01623) to avoid destructive gradient updates.
                if "scaled_grad" in state:
                    scaled_grad_norm = torch.linalg.norm(update)
                    limiter = (
                        max(
                            scaled_grad_norm / (state["scaled_grad"] + 1e-8),
                            1.01,
                        )
                        / 1.01
                    )
                    update = update / limiter
                    state["scaled_grad"] = scaled_grad_norm / limiter
                else:
                    state["scaled_grad"] = torch.norm(update)

                if not scale_front and scale != 1.0:
                    update *= math.sqrt(scale)

                p.add_(update, alpha=-lr)
                # print(f"{projector.proj_type=}, {projector.A.shape=}, {R_tilde.shape=}, {S.shape=}, {grad.shape=}")

        return loss
