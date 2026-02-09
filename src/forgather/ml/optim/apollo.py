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

    def state_dict(self):
        """Return optimizer state with serialized projector objects.

        Projector objects are converted to dicts containing only tensors and primitives
        to ensure proper checkpoint serialization.

        Note: The projector_factory in param_groups is removed since it's a
        non-serializable function. On load_state_dict, it must be provided
        via the optimizer constructor.
        """
        from forgather.ml.optim.subspace_proj import OnlinePCAProjector, RandProjector

        state_dict = super().state_dict()

        # Remove projector_factory from param_groups (can't pickle functions)
        for group in state_dict["param_groups"]:
            if "projector_factory" in group:
                del group["projector_factory"]

        # Serialize projector objects
        for param_id, param_state in state_dict["state"].items():
            if "projector" in param_state:
                proj = param_state["projector"]

                # Serialize based on projector type
                proj_dict = {
                    "_class": type(proj).__name__,
                    "rank": proj.rank,
                    "dim": proj.dim,
                    "proj_type": proj.proj_type,
                    "update_steps": proj.update_steps,
                    "_step": proj._step,
                    "scale": proj.scale,
                }

                # Add type-specific state
                if isinstance(proj, OnlinePCAProjector):
                    proj_dict["A"] = proj.A
                    # orthonormalize function is reconstructed from defaults

                elif isinstance(proj, RandProjector):
                    proj_dict["A"] = proj.A
                    proj_dict["init"] = proj.init
                    proj_dict["lazy"] = proj.lazy
                    proj_dict["seed"] = proj.seed
                    if hasattr(proj, "gen") and proj.gen is not None:
                        proj_dict["gen_state"] = proj.gen.get_state()
                    if hasattr(proj, "saved_gen_state"):
                        proj_dict["saved_gen_state"] = proj.saved_gen_state
                    if hasattr(proj, "device"):
                        proj_dict["device"] = str(proj.device)  # Serialize as string
                    if hasattr(proj, "dtype"):
                        proj_dict["dtype"] = str(proj.dtype)  # Serialize as string

                param_state["projector"] = proj_dict

        return state_dict

    def load_state_dict(self, state_dict):
        """Load optimizer state and reconstruct projector objects.

        Deserializes projector dicts back into projector objects.
        """
        from forgather.ml.optim.subspace_proj import OnlinePCAProjector, RandProjector

        # Reconstruct projector objects from serialized dicts
        for param_id, param_state in state_dict["state"].items():
            if "projector" in param_state:
                proj_dict = param_state["projector"]

                if not isinstance(proj_dict, dict):
                    raise ValueError(
                        f"Apollo projector state must be dict, got {type(proj_dict)}"
                    )

                proj_class_name = proj_dict.get("_class")

                # Reconstruct based on class type
                if proj_class_name == "OnlinePCAProjector":
                    # Note: orthag defaults to "none" in current implementation
                    proj = OnlinePCAProjector(
                        rank=proj_dict["rank"],
                        dim=proj_dict["dim"],
                        proj_type=proj_dict["proj_type"],
                        update_steps=proj_dict["update_steps"],
                    )
                    # Restore all attributes
                    proj.A = proj_dict["A"]
                    proj._step = proj_dict["_step"]
                    proj.scale = proj_dict["scale"]
                    # Note: proj_shape, einsum_* are set by __init__, orthonormalize defaults to identity

                elif proj_class_name == "RandProjector":
                    proj = RandProjector(
                        rank=proj_dict["rank"],
                        dim=proj_dict["dim"],
                        proj_type=proj_dict["proj_type"],
                        update_steps=proj_dict["update_steps"],
                        init=proj_dict["init"],
                        lazy=proj_dict["lazy"],
                        seed=proj_dict["seed"],
                    )
                    proj.A = proj_dict["A"]
                    proj._step = proj_dict["_step"]
                    proj.scale = proj_dict["scale"]

                    # Restore generator state if present
                    if "gen_state" in proj_dict:
                        # Need to create generator on correct device
                        device_str = proj_dict.get("device", "cpu")
                        device = torch.device(device_str.replace("cuda:", "cuda:"))
                        proj.gen = torch.Generator(device=device)
                        proj.gen.set_state(proj_dict["gen_state"])

                    if "saved_gen_state" in proj_dict:
                        proj.saved_gen_state = proj_dict["saved_gen_state"]
                    if "device" in proj_dict:
                        proj.device = torch.device(
                            proj_dict["device"].replace("cuda:", "cuda:")
                        )
                    if "dtype" in proj_dict:
                        # Convert string like "torch.float32" to dtype
                        dtype_str = proj_dict["dtype"].replace("torch.", "")
                        proj.dtype = getattr(torch, dtype_str)

                else:
                    raise ValueError(f"Unknown projector class: {proj_class_name}")

                param_state["projector"] = proj

        super().load_state_dict(state_dict)
