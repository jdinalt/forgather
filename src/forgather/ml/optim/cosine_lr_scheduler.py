import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineLRScheduler(LRScheduler):
    """Cosine decay learning rate scheduler with optional linear warmup.

    Linearly warms the learning rate from 0 to ``base_lr`` over
    ``warmup_steps``, then applies a half-cosine decay from ``base_lr`` to
    ``min_lr`` over the remaining ``total_steps - warmup_steps`` steps.

    This is the standard schedule for fixed-budget training runs.  For
    continual pre-training without a predetermined budget, prefer
    `InfiniteLRScheduler` or `WSDScheduler`.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer whose ``param_groups`` LRs will be managed.
    total_steps : int
        Total number of training steps (warmup + decay combined).
    warmup_steps : int, optional
        Number of linear warmup steps before cosine decay begins.
        Default is 0.
    min_lr : float, optional
        Minimum learning rate at the end of cosine decay.  Default is 0.0.
    last_epoch : int, optional
        Index of the last epoch, used when resuming.  Default is -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        assert total_steps > 0
        assert 0 <= warmup_steps < total_steps
        assert min_lr >= 0.0

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = total_steps - warmup_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            scale = step / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / self.decay_steps
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * scale
                for base_lr in self.base_lrs
            ]
