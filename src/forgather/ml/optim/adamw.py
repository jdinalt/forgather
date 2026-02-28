from typing import Callable, Iterable, Tuple

import torch
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

                    # Draw SR seed from dedicated generator (same across DDP ranks)
                    bf16_sr = group["bf16_stochastic_round"]
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

                    # Prepare CUDA generator for PyTorch SR path
                    sr_cuda_gen = None
                    if bf16_sr and p.is_cuda:
                        device = p.device
                        if device not in self._sr_cuda_generators:
                            self._sr_cuda_generators[device] = torch.Generator(
                                device=device
                            )
                        sr_cuda_gen = self._sr_cuda_generators[device]
                        sr_cuda_gen.manual_seed(sr_seed)

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
                        bf16_sr,
                        sr_cuda_gen,
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

        # Save SR generator state for deterministic resume
        state_dict["sr_generator_state"] = self._sr_generator.get_state()

        return state_dict

    def load_state_dict(self, state_dict):
        """Load optimizer state with validation."""
        # Shallow copy to avoid mutating caller's dict
        state_dict = dict(state_dict)
        # Extract SR generator state before super() processes the dict
        sr_gen_state = state_dict.pop("sr_generator_state", None)

        # Validate before loading
        for param_id, param_state in state_dict["state"].items():
            expected_keys = {"step", "m", "v"}
            if not expected_keys.issubset(param_state.keys()):
                missing = expected_keys - param_state.keys()
                raise ValueError(
                    f"Cannot load AdamW: missing keys for param {param_id}: {missing}"
                )

        super().load_state_dict(state_dict)

        # Restore SR generator state for deterministic resume
        if sr_gen_state is not None:
            self._sr_generator.set_state(sr_gen_state)


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
    sr_generator=None,
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
    # Upcast to float32 for numerical stability. When already float32,
    # .float() returns the same tensor and lerp_ operates in-place.
    grad32 = grad.float()
    m32 = m.float()
    v32 = v.float()

    m32.lerp_(grad32, 1.0 - beta1)
    v32.lerp_(grad32.square(), 1.0 - beta2)
    update = m32 / (v32.sqrt() + eps)

    # Write states back to storage precision
    if m32.dtype != m.dtype:
        if bf16_stochastic_round:
            m.copy_(fp32_to_bf16_stochastic_round(m32, generator=sr_generator))
            v.copy_(fp32_to_bf16_stochastic_round(v32, generator=sr_generator))
        else:
            m.copy_(m32)
            v.copy_(v32)

    # Bias correction
    alpha = alpha * torch.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)

    # Update parameter
    if p.dtype == update.dtype:
        p.add_(update, alpha=-alpha)
    else:
        update = p.float() - alpha * update
        if bf16_stochastic_round:
            update = fp32_to_bf16_stochastic_round(update, generator=sr_generator)
        p.copy_(update)
