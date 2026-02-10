import math
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from .rounding_utils import fp32_to_bf16_stochastic_round


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
        weight_decay: float = 0.01,
        torch_compile: bool = False,
        bf16_stochastic_round: bool = False,
    ):
        self.compile = torch_compile
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bf16_stochastic_round=bf16_stochastic_round,
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
                        group["bf16_stochastic_round"],
                    ]
                    if self.compile:
                        torch.compile(_adam, fullgraph=True, dynamic=False)(*args)
                    else:
                        _adam(*args)

        return loss

    def state_dict(self):
        """Return optimizer state with structure validation."""
        state_dict = super().state_dict()

        # Validate state structure for debugging
        for param_id, param_state in state_dict["state"].items():
            expected_keys = {"step", "m", "v"}
            if not expected_keys.issubset(param_state.keys()):
                missing = expected_keys - param_state.keys()
                raise ValueError(
                    f"AdamW state missing keys for param {param_id}: {missing}"
                )

        return state_dict

    def load_state_dict(self, state_dict):
        """Load optimizer state with validation."""
        # Validate before loading
        for param_id, param_state in state_dict["state"].items():
            expected_keys = {"step", "m", "v"}
            if not expected_keys.issubset(param_state.keys()):
                missing = expected_keys - param_state.keys()
                raise ValueError(
                    f"Cannot load AdamW: missing keys for param {param_id}: {missing}"
                )

        super().load_state_dict(state_dict)


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
    bf16_stochastic_round: bool,
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
        update = p.float() - alpha * update
        if bf16_stochastic_round:
            update = fp32_to_bf16_stochastic_round(update)
        p.copy_(update)
