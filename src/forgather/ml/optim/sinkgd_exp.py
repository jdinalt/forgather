import math
from typing import Callable, Iterable, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .rounding_utils import fp32_to_bf16_stochastic_round


class SinkGD(Optimizer):
    """
    SinkGD (Experimental): Sinkhorn Gradient Descent with Adafactor-style factored normalization.

    A stateless optimizer that pre-processes gradients via alternating row/column
    RMS normalization (the factored Adafactor approach to SR-Sinkhorn). Requires
    no optimizer state beyond an optional momentum buffer, giving it memory
    footprint between SGD (no state) and Adam (two states).

    From: "Gradient Multi-Normalization for Stateless and Scalable LLM Training"
    Scetbon, Ma, Gong, Meeds -- arXiv:2502.06742

    Key differences from sinkgd.py (the paper-faithful implementation):
    - Uses Adafactor-style factored row/column RMS normalization instead of
      explicit row/column L2 normalization. Mathematically equivalent at
      convergence but more efficient per iteration.
    - Supports torch.compile for fused kernel generation.
    - Supports stochastic rounding for bf16 training.
    - Supports momentum (EMA on normalized update) with optional Nesterov.
    - Supports per-group mode: "full", "col_only", "sparse".

    Parameter handling by dimension:
    - >=2D: Reshape to (-1, last_dim), apply Sinkhorn normalization, reshape back.
    - 1D (biases, norms): Configurable via vector_mode ("l2_norm", "sgd", "sign").
    - 0D (scalars): Gradient used directly (no normalization).

    Args:
        params: Parameters or parameter groups.
        lr: Learning rate. Roughly matches Adam when normalize_output=True.
        eps: Denominator clamp for numerical stability.
        num_iters: Number of Sinkhorn normalization iterations. Paper uses 5;
            1 is cheaper with negligible quality loss.
        weight_decay: Decoupled weight decay (AdamW-style).
        normalize_output: Divide output by sqrt(m*n) so lr is Adam-comparable.
        momentum: EMA coefficient on normalized update (0 = off/stateless).
        nesterov: Use Nesterov-style momentum look-ahead.
        mode: Normalization mode for 2D+ params:
            "full" - standard row+column normalization (default)
            "col_only" - column normalization only, preserves row structure
            "sparse" - normalize only active rows (nonzero energy)
        vector_mode: How to handle 1D parameters:
            "l2_norm" - normalize by L2 norm (default, matches sinkgd.py)
            "sgd" - pass gradient through unchanged
            "sign" - use sign of gradient (like LION)
        torch_compile: Enable torch.compile for the inner loop.
        bf16_stochastic_round: Use stochastic rounding for bf16 params.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        eps: float = 1e-15,
        num_iters: int = 5,
        weight_decay: float = 0.0,
        normalize_output: bool = True,
        momentum: float = 0.0,
        nesterov: bool = False,
        mode: str = "full",
        vector_mode: str = "l2_norm",
        torch_compile: bool = True,
        bf16_stochastic_round: bool = False,
    ):
        self.compile = torch_compile
        self.bf16_stochastic_round = bf16_stochastic_round
        defaults = dict(
            lr=lr,
            eps=eps,
            num_iters=num_iters,
            weight_decay=weight_decay,
            normalize_output=normalize_output,
            momentum=momentum,
            nesterov=nesterov,
            mode=mode,
            vector_mode=vector_mode,
        )
        super().__init__(params, defaults)
        self._sr_generator = torch.Generator()
        self._sr_generator.manual_seed(5489)
        self._sr_cuda_generators = {}  # device -> Generator, lazily created

    def add_param_group(self, param_group: dict):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if not isinstance(group["lr"], Tensor):
            group["lr"] = torch.tensor(group["lr"], dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        with torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                for p in group["params"]:
                    grad = p.grad
                    if grad is None:
                        continue

                    lr = group["lr"]
                    assert isinstance(
                        lr, Tensor
                    ), "Someone changed our lr to a non-Tensor!?"

                    eps = group["eps"]
                    num_iters = group["num_iters"]
                    weight_decay = group["weight_decay"]
                    normalize_output = group["normalize_output"]
                    mom = group["momentum"]
                    nesterov = group["nesterov"]
                    mode = group["mode"]
                    vector_mode = group["vector_mode"]
                    bf16_sr = self.bf16_stochastic_round

                    # Prepare CUDA generator for PyTorch SR path
                    if bf16_sr:
                        sr_seed = int(
                            torch.randint(
                                0,
                                2**31,
                                (1,),
                                generator=self._sr_generator,
                            ).item()
                        )
                    else:
                        sr_seed = 0

                    sr_cuda_gen = None
                    if bf16_sr and p.is_cuda:
                        device = p.device
                        if device not in self._sr_cuda_generators:
                            self._sr_cuda_generators[device] = torch.Generator(
                                device=device
                            )
                        sr_cuda_gen = self._sr_cuda_generators[device]
                        sr_cuda_gen.manual_seed(sr_seed)

                    # Handle momentum state
                    momentum_buffer = None
                    if mom > 0:
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        momentum_buffer = state["momentum_buffer"]

                    # Use standard PyTorch implementation
                    args = [
                        p.detach(),
                        grad,
                        lr,
                        eps,
                        num_iters,
                        weight_decay,
                        normalize_output,
                        mom,
                        nesterov,
                        mode,
                        vector_mode,
                        bf16_sr,
                        sr_cuda_gen,
                        momentum_buffer,
                    ]
                    if self.compile:
                        torch.compile(_sinkgd, fullgraph=True, dynamic=False)(*args)
                    else:
                        _sinkgd(*args)

        return loss

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["sr_generator_state"] = self._sr_generator.get_state()
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        sr_gen_state = state_dict.pop("sr_generator_state", None)
        super().load_state_dict(state_dict)
        if sr_gen_state is not None:
            self._sr_generator.set_state(sr_gen_state)


def _sinkgd(
    p: Tensor,
    grad: Tensor,
    lr: Tensor,
    eps: float,
    num_iters: int,
    weight_decay: float,
    normalize_output: bool,
    momentum: float,
    nesterov: bool,
    mode: str,
    vector_mode: str,
    bf16_stochastic_round: bool,
    sr_generator=None,
    momentum_buffer=None,
):
    # Decoupled weight decay (AdamW-style)
    if weight_decay > 0.0:
        p.add_(p, alpha=(-lr * weight_decay))

    if grad.dim() >= 2:
        update = _normalize_2d(grad, eps, num_iters, normalize_output, mode)
    elif grad.dim() == 1:
        update = _normalize_1d(grad, eps, vector_mode)
    else:
        # Scalar: use gradient directly
        update = grad.float()

    # Apply momentum
    if momentum > 0 and momentum_buffer is not None:
        momentum_buffer.lerp_(update, 1.0 - momentum)
        if nesterov:
            update = update + momentum * momentum_buffer
        else:
            update = momentum_buffer.clone()

        # Re-normalize after momentum to preserve constant-magnitude property
        if grad.dim() >= 2:
            rms_val = update.square().mean().sqrt()
            if normalize_output:
                m = grad[..., 0].numel()
                n = grad.shape[-1]
                target_rms = 1.0 / math.sqrt(m * n)
            else:
                target_rms = 1.0
            update = update * (target_rms / rms_val.clamp(min=eps))
        elif grad.dim() == 1 and vector_mode == "l2_norm":
            update = update / update.norm().clamp(min=eps)

    if p.dtype == update.dtype:
        p -= lr * update
    else:
        update = p.float() - lr * update
        if bf16_stochastic_round:
            update = fp32_to_bf16_stochastic_round(update, generator=sr_generator)
        p.copy_(update)


def _normalize_2d(
    grad: Tensor, eps: float, num_iters: int, normalize_output: bool, mode: str
) -> Tensor:
    orig_shape = grad.shape
    if grad.dim() > 2:
        grad = grad.reshape(-1, grad.shape[-1])

    update = grad.float()
    m, n = update.shape

    match mode:
        case "full":
            for _ in range(num_iters):
                sq = update.square()
                r = sq.sum(dim=1) + eps
                c = sq.sum(dim=0) + eps
                update = update * torch.outer(torch.rsqrt(r / r.sum()), torch.rsqrt(c))
        case "col_only":
            update = (
                update
                * update.square().mean(dim=0, keepdim=True).clamp(min=eps).rsqrt()
            )
        case "row_only":
            update = (
                update
                * update.square().mean(dim=1, keepdim=True).clamp(min=eps).rsqrt()
            )
        case "sparse":
            row_energy = update.square().sum(dim=1)
            active_mask = row_energy > 0
            if active_mask.any():
                active = update[active_mask]
                for _ in range(num_iters):
                    sq = active.square()
                    r = sq.sum(dim=1) + eps
                    c = sq.sum(dim=0) + eps
                    active = active * torch.outer(
                        torch.rsqrt(r / r.sum()), torch.rsqrt(c)
                    )
                update = update.clone()
                update[active_mask] = active
        case "rms":
            update = update * update.square().mean().clamp(min=eps).rsqrt()
        case "sgd":
            pass
        case _:
            raise ValueError(f"Unsupported mode: {mode}")

    if normalize_output:
        update = update * (1.0 / math.sqrt(m * n))

    if update.shape != orig_shape:
        update = update.reshape(orig_shape)
    return update


def _normalize_1d(grad: Tensor, eps: float, vector_mode: str) -> Tensor:
    update = grad.float()
    match vector_mode:
        case "l2_norm":
            update = update / update.norm().clamp(min=eps)
        case "rms":
            update = update * update.square().mean().clamp(min=eps).rsqrt()
        case "sign":
            update = update.sign()
        case "sgd":
            pass
        case _:
            raise ValueError(f"Unsupported mode: {mode}")

    return update
