import math
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from .rounding_utils import fp32_to_bf16_stochastic_round


class Adafactor(Optimizer):
    """
    Adafactor
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        decay_rate: float = -0.8,
        clip_threshold: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: Tuple[float, float] = (1e-30, 1e-3),
        weight_decay: float = 0.0,
        relative_step: bool = False,
        torch_compile: bool = False,
        bf16_stochastic_round: bool = False,
        use_triton: bool = False,
    ):
        self.compile = torch_compile
        self.use_triton = use_triton

        # Import Triton kernels if needed
        if use_triton:
            assert (
                relative_step == False
            ), "relative_step is not supported by Adafactor Triton kernel. Set use_triton = False"
            try:
                from . import adafactor_triton

                self.triton_module = adafactor_triton
            except ImportError as e:
                raise ImportError(
                    "Triton is required for use_triton=True. "
                    "Please install it with: pip install triton"
                ) from e

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            decay_rate=decay_rate,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            relative_step=relative_step,
            bf16_stochastic_round=bf16_stochastic_round,
        )
        super().__init__(params, defaults)

        # Dedicated generator for stochastic rounding. Using a fixed seed
        # ensures all DDP ranks produce identical rounding decisions,
        # preventing parameter divergence across ranks. The generator is
        # only advanced by SR draws (not shared with dropout, data loading,
        # etc.) so it stays in sync as long as all ranks process the same
        # parameters in the same order -- which DDP guarantees.
        self._sr_generator = torch.Generator()
        self._sr_generator.manual_seed(5489)
        self._sr_cuda_generators = {}  # device -> Generator, lazily created

    def _init_state(self, state, group, p, grad):
        state["step"] = torch.tensor(0.0, dtype=torch.float32)
        if grad.dim() <= 1:
            state["row"] = torch.zeros_like(grad, dtype=p.dtype)
            state["col"] = None
        else:
            state["row"] = torch.zeros(grad.shape[0], dtype=p.dtype, device=grad.device)
            state["col"] = torch.zeros(grad.shape[1], dtype=p.dtype, device=grad.device)

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

                    # Compute decay parameter for beta2
                    beta2t = (1.0 - state["step"] ** group["decay_rate"]).clamp(
                        max=beta2
                    )

                    # Draw SR seed from dedicated generator (same across DDP ranks)
                    bf16_sr = group["bf16_stochastic_round"]
                    if bf16_sr:
                        sr_seed = int(torch.randint(
                            0, 2**31, (1,), generator=self._sr_generator,
                        ).item())
                    else:
                        sr_seed = 0

                    # Route to Triton or PyTorch implementation
                    if self.use_triton and grad.is_cuda:
                        # Use Triton kernels
                        if state["col"] is None:
                            # 1D case
                            self.triton_module.adafactor_step_1d_triton(
                                p,
                                grad,
                                state["row"],
                                beta2t,
                                eps1,
                                group["lr"],
                                group["weight_decay"],
                                group["clip_threshold"],
                                bf16_sr,
                                sr_seed,
                            )
                        else:
                            # 2D case
                            self.triton_module.adafactor_step_2d_triton(
                                p,
                                grad,
                                state["row"],
                                state["col"],
                                beta2t,
                                eps1,
                                group["lr"],
                                group["weight_decay"],
                                group["clip_threshold"],
                                bf16_sr,
                                sr_seed,
                            )
                    else:
                        # Prepare CUDA generator for PyTorch SR path
                        sr_cuda_gen = None
                        if bf16_sr and p.is_cuda:
                            device = p.device
                            if device not in self._sr_cuda_generators:
                                self._sr_cuda_generators[device] = (
                                    torch.Generator(device=device)
                                )
                            sr_cuda_gen = self._sr_cuda_generators[device]
                            sr_cuda_gen.manual_seed(sr_seed)

                        # Use standard PyTorch implementation
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
                            bf16_sr,
                            sr_cuda_gen,
                        ]
                        if self.compile:
                            torch.compile(_adafactor, fullgraph=True, dynamic=False)(
                                *args
                            )
                        else:
                            _adafactor(*args)

        return loss

    def state_dict(self):
        """Return optimizer state handling conditional col=None."""
        state_dict = super().state_dict()

        # Validate state structure
        for param_id, param_state in state_dict["state"].items():
            expected_keys = {"step", "row", "col"}
            if not expected_keys.issubset(param_state.keys()):
                missing = expected_keys - param_state.keys()
                raise ValueError(
                    f"Adafactor state missing keys for param {param_id}: {missing}"
                )

            # Ensure col=None is handled correctly (not converted to tensor)
            if param_state["col"] is not None and not torch.is_tensor(
                param_state["col"]
            ):
                raise ValueError(
                    f"Adafactor col must be tensor or None, got {type(param_state['col'])}"
                )

        # Save SR generator state for deterministic resume
        state_dict["sr_generator_state"] = self._sr_generator.get_state()

        return state_dict

    def load_state_dict(self, state_dict):
        """Load optimizer state handling conditional col=None."""
        # Extract SR generator state before super() processes the dict
        sr_gen_state = state_dict.pop("sr_generator_state", None)

        # Validate structure
        for param_id, param_state in state_dict["state"].items():
            expected_keys = {"step", "row", "col"}
            if not expected_keys.issubset(param_state.keys()):
                missing = expected_keys - param_state.keys()
                raise ValueError(
                    f"Cannot load Adafactor: missing keys for param {param_id}: {missing}"
                )

        super().load_state_dict(state_dict)

        # Restore SR generator state for deterministic resume
        if sr_gen_state is not None:
            self._sr_generator.set_state(sr_gen_state)


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
    bf16_stochastic_round: bool,
    sr_generator=None,
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
    update = grad32**2 + eps1

    # We clamp to beta2, which is not from the paper, but appears to be a bit
    # more flexible that decaying to infinity..
    beta2t = (1.0 - step**decay_rate).clamp(max=beta2)

    # Vectors and scalars are not factored.
    r32 = r.float()
    if c is None:
        r32.lerp_(update, 1.0 - beta2t)
        update = grad32 / r32.sqrt()
    # Matrix
    else:
        c32 = c.float()
        # See adagrad_update_ref() for explanation of this implementation
        r32.lerp_(update.sum(dim=-1), 1.0 - beta2t)
        c32.lerp_(update.sum(dim=-2), 1.0 - beta2t)
        update = grad32 * torch.outer(torch.rsqrt(r32 / r32.sum()), torch.rsqrt(c32))
        if c32.dtype != c.dtype:
            c.copy_(c32)

    if r32.dtype != r.dtype:
        r.copy_(r32)

    # Apply update clipping
    update /= (rms(update) / clip_threshold).clamp_(min=1.0)

    # Update parameter
    if p.dtype == update.dtype:
        p.add_(update, alpha=-lr)
    else:
        update = p.float() - lr * update
        if bf16_stochastic_round:
            update = fp32_to_bf16_stochastic_round(update, generator=sr_generator)
        p.copy_(update)
