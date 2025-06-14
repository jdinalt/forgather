from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
import math

"""
    https://arxiv.org/html/2503.02844v1
    Beyond Cosine Decay: On the effectiveness of Infinite Learning Rate Schedule for Continual Pre-training
"""
class InfiniteLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
        constant_lr: float = 3.75e-5,
        min_lr: float = 1e-8,
        tau: float = 1e4,
        checkpoint_step: int = -1,
        last_epoch: int = -1,
    ):
        assert warmup_steps >= 0
        assert cooldown_steps >= 0
        assert checkpoint_step < 0 or checkpoint_step >= warmup_steps + cooldown_steps
        assert tau > 0
        assert min_lr >= 0.
        assert constant_lr > 0.
        
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.constant_lr = constant_lr
        self.checkpoint_step = checkpoint_step
        self.min_lr = min_lr
        self.tau = tau
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return self._warmup_lr()
        elif self.last_epoch < self.warmup_steps + self.cooldown_steps:
            return self._cooldown_lr()
        elif self.checkpoint_step >= 0 and self.last_epoch >= self.checkpoint_step:
            return self._annealing_lr()
        else:
            return self._constant_lr()
        
    def _warmup_lr(self):
        # If cooldown phase, warmp up to self.base_lrs -- cooldown will go to constant_lr
        if self.cooldown_steps > 0:
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs)
            ]
        # Warm-up to constant_lr; otherwise this will just create a sudden drop.
        else:
            return [
                self.constant_lr * self.last_epoch / self.warmup_steps
                for group, base_lr in self.optimizer.param_groups
            ]

    def _cooldown_lr(self):
        return [
            self.constant_lr + ((base_lr - self.constant_lr) / 2) *
            (1. + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / self.cooldown_steps))
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs)
        ]

    def _constant_lr(self):
        return [ self.constant_lr for _ in self.optimizer.param_groups ]

    def _annealing_lr(self):
        last_epoch = self.last_epoch - self.checkpoint_step
        """
        Note: This differs from the exponential decay in the paper.

        Exponential decay from constant_lr to min_lr, with decay constant tau.
        See: https://en.wikipedia.org/wiki/Exponential_decay
        """
        return [
            self.min_lr + (self.constant_lr - self.min_lr) * math.exp(-last_epoch/self.tau)
            for group in self.optimizer.param_groups
        ]