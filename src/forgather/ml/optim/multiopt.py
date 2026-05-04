from typing import Callable, Optional

import torch
from torch.optim import Optimizer


class Multiopt(Optimizer):
    """
    Composite optimizer that fans ``step``/``zero_grad`` out to a list of
    wrapped optimizers and aggregates their state dicts.

    The trainer constructs a ``Multiopt`` automatically when an
    ``optimizer_groups`` configuration declares more than one distinct
    optimizer factory (see ``forgather.ml.optim.opt_utils
    .build_optimizer_buckets``).
    """

    def __init__(
        self,
        optimizers: list,
    ):
        param_groups = []
        for opt in optimizers:
            for group in opt.param_groups:
                param_groups.append(group)

        super().__init__(param_groups, {})
        self.optimizers = optimizers

    @torch.no_grad()
    def step(  # type: ignore[override]
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        loss: Optional[float] = None
        if closure is not None:
            loss = closure()

        for opt in self.optimizers:
            opt.step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return aggregated state from all wrapped optimizers."""
        return {
            "optimizers": [
                {"index": i, "state_dict": opt.state_dict()}
                for i, opt in enumerate(self.optimizers)
            ]
        }

    def load_state_dict(self, state_dict):
        """Load aggregated state into wrapped optimizers."""
        if "optimizers" not in state_dict:
            raise ValueError("Multiopt state_dict must contain 'optimizers' key")

        opt_states = state_dict["optimizers"]

        if len(opt_states) != len(self.optimizers):
            raise ValueError(
                f"Multiopt optimizer count mismatch: expected {len(self.optimizers)}, "
                f"got {len(opt_states)}"
            )

        # Restore each optimizer by index
        for opt_state in opt_states:
            index = opt_state["index"]
            if index >= len(self.optimizers):
                raise ValueError(f"Invalid optimizer index in state_dict: {index}")

            self.optimizers[index].load_state_dict(opt_state["state_dict"])
