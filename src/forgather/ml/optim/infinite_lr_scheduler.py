import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class InfiniteLRScheduler(LRScheduler):
    """Infinite Learning Rate Scheduler.

    Implements the Infinite Cosine Schedule from:
    "Beyond Cosine Decay: On the effectiveness of Infinite Learning Rate
    Schedule for Continual Pre-training"
    (https://arxiv.org/abs/2503.02844)

    The schedule consists of four phases (Eq. 1 in the paper):

        1. Warmup: Linear increase from 0 to base_lr over warmup_steps.
        2. Cooldown: Cosine decay from base_lr to constant_lr over
           cooldown_steps.
        3. Constant: Maintains constant_lr indefinitely (the "infinite"
           phase). This phase can run for an arbitrary number of steps,
           enabling continual pre-training without a predetermined budget.
        4. Annealing: Exponential decay from constant_lr toward min_lr,
           triggered at checkpoint_step when a converged checkpoint is
           needed.
    """

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
        """
        Args:
            optimizer: Wrapped optimizer.
            warmup_steps: Number of steps for linear warmup (phase 1).
            cooldown_steps: Number of steps for cosine cooldown from
                base_lr to constant_lr (phase 2).
            constant_lr: Learning rate maintained during the constant
                phase (phase 3) and the starting point for annealing
                (phase 4). Corresponds to eta_const in the paper.
            min_lr: Target minimum learning rate for the annealing phase.
                Must be > 0. Corresponds to eta_min in the paper.
            tau: Annealing step budget. Controls the decay rate in the
                annealing phase. Corresponds to t_a in the paper. The LR
                reaches exactly min_lr after (tau + checkpoint_step) steps
                past checkpoint_step.
            checkpoint_step: Step at which to begin annealing (phase 4).
                Set to -1 to disable annealing. Must be >=
                warmup_steps + cooldown_steps when enabled. Corresponds
                to N_d in the paper.
            last_epoch: The index of the last epoch. Used for resuming.
        """
        assert warmup_steps >= 0
        assert cooldown_steps >= 0
        assert checkpoint_step < 0 or checkpoint_step >= warmup_steps + cooldown_steps
        assert tau > 0
        assert min_lr > 0.0
        assert constant_lr > 0.0

        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.constant_lr = constant_lr
        self.checkpoint_step = checkpoint_step
        self.min_lr = min_lr
        self.tau = tau

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for the current step."""
        if self.last_epoch < self.warmup_steps:
            return self._warmup_lr()
        elif self.last_epoch < self.warmup_steps + self.cooldown_steps:
            return self._cooldown_lr()
        elif self.checkpoint_step >= 0 and self.last_epoch >= self.checkpoint_step:
            return self._annealing_lr()
        else:
            return self._constant_lr()

    def _warmup_lr(self):
        """Phase 1: Linear warmup from 0 to base_lr."""
        return [
            base_lr * self.last_epoch / self.warmup_steps
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs)
        ]

    def _cooldown_lr(self):
        """Phase 2: Cosine decay from base_lr to constant_lr."""
        return [
            self.constant_lr
            + ((base_lr - self.constant_lr) / 2)
            * (
                1.0
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_steps)
                    / self.cooldown_steps
                )
            )
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs)
        ]

    def _constant_lr(self):
        """Phase 3: Constant learning rate (the "infinite" phase)."""
        if self.cooldown_steps > 0:
            return [self.constant_lr for _ in self.optimizer.param_groups]
        else:
            return [base_lr for base_lr in self.base_lrs]

    def _annealing_lr(self):
        """Phase 4: Exponential decay from constant_lr toward min_lr.

        From Eq. 1 of the paper:
            eta(n) = eta_const * (eta_min / eta_const) ^ ((n - N_d) / (t_a + N_d))

        where n is the current step, N_d is checkpoint_step, and t_a is
        tau. The LR equals min_lr when n - N_d = t_a + N_d (i.e., after
        tau + checkpoint_step annealing steps).
        """
        steps_since_anneal = self.last_epoch - self.checkpoint_step
        exponent = steps_since_anneal / (self.tau + self.checkpoint_step)
        return [
            self.constant_lr * (self.min_lr / self.constant_lr) ** exponent
            for group in self.optimizer.param_groups
        ]
