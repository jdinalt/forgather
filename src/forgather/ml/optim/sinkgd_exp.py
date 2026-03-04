import math
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from .rounding_utils import fp32_to_bf16_stochastic_round


class SinkGD(Optimizer):
    """
    SinkGD
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        eps: float = 1e-15,
        num_iters: int = 5,
        clip_threshold: float = 1.0,
        torch_compile: bool = True,
        bf16_stochastic_round: bool = False,
    ):
        self.compile = torch_compile
        self.bf16_stochastic_round = bf16_stochastic_round
        defaults = dict(
            lr=lr,
            eps=eps,
            num_iters=num_iters,
            clip_threshold=clip_threshold,
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
    def step(self, closure: Callable = None):
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
                    clip_threshold = group["clip_threshold"]
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

                    # Use standard PyTorch implementation
                    args = [
                        p.detach(),
                        grad,
                        lr,
                        eps,
                        clip_threshold,
                        num_iters,
                        bf16_sr,
                        sr_cuda_gen,
                    ]
                    if self.compile:
                        torch.compile(_sinkgd, fullgraph=True, dynamic=False)(*args)
                    else:
                        _sinkgd(*args)

        return loss


def _sinkgd(
    p: Tensor,
    grad: Tensor,
    lr: Tensor,
    eps: float,
    clip_threshold: float,
    num_iters: int,
    bf16_stochastic_round: bool,
    sr_generator=None,
):
    update = grad.float()
    # Compute gradient magnitude before normalization
    # grad_rms = update.square().mean().sqrt()

    for _ in range(num_iters):
        sq = update.square()
        r = sq.sum(dim=1) + eps
        c = sq.sum(dim=0) + eps
        update = update * torch.outer(torch.rsqrt(r / r.sum()), torch.rsqrt(c))

    # Restore magnitude with optional clipping
    # update *= grad_rms.clamp(max=clip_threshold)

    if p.dtype == update.dtype:
        p -= lr * update
    else:
        update = p.float() - lr * update
        if bf16_stochastic_round:
            update = fp32_to_bf16_stochastic_round(update, generator=sr_generator)
        p.copy_(update)
