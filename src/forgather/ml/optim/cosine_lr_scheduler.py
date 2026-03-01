import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineLRScheduler(LRScheduler):
    """Cosine decay learning rate scheduler with optional linear warmup.

    Linearly warms up from 0 to base_lr over warmup_steps, then decays
    to 0 following a cosine curve over the remaining steps.

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Total number of training steps (warmup + decay).
        warmup_steps: Number of linear warmup steps. Default: 0.
        last_epoch: The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        assert total_steps > 0
        assert 0 <= warmup_steps < total_steps

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = total_steps - warmup_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / self.decay_steps
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [base_lr * scale for base_lr in self.base_lrs]
