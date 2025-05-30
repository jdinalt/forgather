import math
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Optimizer

class Adafactor(Optimizer):
    """
    Adafactor
    """
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float=1e-3,
        decay_rate: float=-0.8,
        clip_threshold: float=1.0,
        betas: Tuple[float, float]=(0.9, 0.999),
        eps: Tuple[float, float]=(1e-30, 1e-3),
        weight_decay: float = 0.0,
        relative_step: bool = False,
        torch_compile: bool = False,
    ):
        self.compile = torch_compile
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            decay_rate=decay_rate,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            relative_step=relative_step,
        )
        super().__init__(params, defaults)

    def _init_state(self, state, group, p, grad):
        state["step"] = torch.tensor(0.0)
        if grad.dim() <= 1:
            state["row"] = torch.zeros_like(grad)
            state["col"] = None
        else:
            state["row"] = torch.zeros(grad.shape[0], dtype=torch.float32, device=grad.device)
            state["col"] = torch.zeros(grad.shape[1], dtype=torch.float32, device=grad.device)
        
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
                    beta1, beta2 = group["betas"]
                    eps1, eps2 = group["eps"]

                    args = [
                        p,
                        grad,
                        state["step"],
                        state["row"],
                        state["col"],
                        group["lr"],
                        beta1,
                        beta2,
                        group["decay_rate"],
                        group["clip_threshold"],
                        eps1,
                        eps2,
                        group["weight_decay"],
                        group["relative_step"],
                    ]
                    if self.compile:
                        torch.compile(_adafactor, fullgraph=True, dynamic=False)(*args)
                    else:
                        _adafactor(*args)

        return loss

"""
TODO: Implement Stochastic Rounding
https://arxiv.org/abs/2010.06192
# https://github.com/pytorch/ao/blob/main/torchao/optim/quant_utils.py#L120
"""

"""
Derivation of the implementation:

See: https://arxiv.org/pdf/1804.04235

It may not be obvious as to how the implementation relates to the original
formulation of the factorization in the paper, so...

This first implementation tries to stay as literal as possible to the paper.

# Returns factor by which gradient should be scaled
# Note: We don't include the 2nd-moment EPA in these examples for brevity.

def adagrad_update_ref(
    G, eps,
):
    # In the paper, these are 1n and 1m, where n and m are subscripts
    # Colmun vectors of ones.
    n, m = G.shape
    ones_n = torch.ones(n, 1, device=G.device)
    ones_m = torch.ones(m, 1, device=G.device)

    R = (G**2 + eps * ones_n @ ones_m.T) @ ones_m
    C = ones_n.T @ (G**2 + eps * ones_n @ ones_m.T)
    V_hat = (R @ C) / (ones_n.T @ R)
    return 1. / torch.sqrt(V_hat)


Breaking this down:

    - eps * ones_n @ tones_m.T
    This is the cross-product of ones-vectors, multiplies by eps. Effectively,
    this is just a n x m matrix with 'eps' in all elements. This is equivalent to
    'G**2 + eps,' using PyTorch broadcast symantics.
    
    - X @ ones_m
    X is a (m x n) matrix and ones_m is a (m x 1) column vector of ones. This operation is
    equivalent to multiplying each column of X by one and summing it, yielding a (m x 1)
    column vector, which is the same as: X.sum(dim=1, keepdim=True)

We can then simplify this to:

def adagrad_update_ref(
    G, eps,
):
    # Don't compute this twice
    G_sq = G**2 + eps
    
    R = G_sq.sum(dim=1, keepdim=True)
    C = G_sq.sum(dim=0, keepdim=True)
    V_hat = (R @ C) / R.sum()
    return 1. / torch.sqrt(V_hat)

Finally, we can compute the reciprocol-square-roots of the rows and columns before 
computing their outer product, which, in theory, saves a few floating-point ops.

Also note that 'sum()' can be replaced by 'mean(),' which you see in the Huggingface implementation,
as the ratios are the same.

def adagrad_update_opt2(
    G, eps,
):
    G_sq = G**2 + eps
    r_col_sum = G_sq.sum(dim=1)
    R_rsqrt = torch.rsqrt(r_col_sum / r_col_sum.sum())
    C_rsqrt = torch.rsqrt(G_sq.sum(dim=0))
    return torch.outer(R_rsqrt, C_rsqrt)

Surprisingly, when profing the above functions, torch seems pretty good at optimization and there's much 
less difference in performance than one would think.

---

The paper proposes replacing absolute step size (alpha, a.k.a learning-rate) with
a relative step size, p, which is proportional to the RMS of the parameter, with a
lower-bound of eps2, to allow escape from 0.

    alpha = max(eps2, rms(p)) * p

If relative_step is disabled, we just use 'lr,' as with Adam.

If enabled, we use the specified 'lr' becomes the relative step size, 'p'

There seems to be some confusion surounding this in the Pytoch implementation, which 
instead sets 'p' to min(lr, 1/sqrt(t)).

As best I can tell, this appears to be a misinterpretation. In Table 2, in the relative-step-size
column, you can see various approaches they tried:
    - alpha = 0.1 * s
    - p = s

    Where 's' was:

With Warmup: min(10-6*t, 1/sqrt(t))
Without Warmup: min(10-2, 1/sqrt(t))

In section 9, they explain that this is not part of their algorithm, but is a learning rate scheduler,
intended to replicate that of Vaswani et al. (2017).

My interpretation is that 's' is a function of the learning rate scheduler, which controls
'lr' in PyTorch and that when using 'relative_step': p = s, as per experiments O and P.

Thus, lr = max(eps2, rms(p)) * lr

Using their lr scheduler for reference, this suggests an lr ~= 1e-2

"""

def rms(x):
    return x.square().mean().sqrt()

def _adafactor(
    p: Tensor,
    grad: Tensor,
    step: Tensor,
    r: Tensor,
    c: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    decay_rate: float,
    clip_threshold: float,
    eps1: float,
    eps2: float,
    weight_decay: float,
    relative_step: bool,
):
    """
    Adafactor: Adaptive Learning Rates with Sublinear Memory Cost
    https://arxiv.org/pdf/1804.04235
    """

    if relative_step:
        lr = max(eps2, rms(p)) * lr
    
    """
    DECOUPLED WEIGHT DECAY REGULARIZATION
    https://arxiv.org/pdf/1711.05101

    We use the above method for implementing weight decay, which scales with lr.
    """
    if weight_decay > 0.0:
        p.add_(p, alpha=(-lr * weight_decay))

    grad32 = grad.float()
    update = grad32 ** 2 + eps1
    
    # We clamp to beta2, which is not from the paper, but appears to be a bit
    # more flexible that decaying to infinity..
    beta2t = (1.0 - step ** decay_rate).clamp(max=beta2)

    # Vectors and scalars are not factored.
    if c is None:
        r.lerp_(update, 1.0 - beta2t)
        update = grad32 / r.sqrt()
    # Matrix
    else:
        # See adagrad_update_ref() for explanation of this implementation
        r.lerp_(update.sum(dim=-1), 1.0 - beta2t)
        c.lerp_(update.sum(dim=-2), 1.0 - beta2t)
        update = grad32 * torch.outer(torch.rsqrt(r / r.sum()), torch.rsqrt(c))

    # Apply update clipping
    update /= (rms(update) / clip_threshold).clamp_(min=1.0)

    # Update parameter
    if p.dtype == update.dtype:
        p.add_(update, alpha=-lr)
    else:
        p.copy_(p.float() - lr * update)

        