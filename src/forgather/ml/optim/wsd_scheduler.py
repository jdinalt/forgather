from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WSDScheduler(LRScheduler):
    """Warmup-Stable-Decay learning rate scheduler.

    Implements the WSD-S protocol (Hu et al., arXiv:2410.05192).  The stable
    phase holds ``base_lr`` indefinitely, enabling training without a fixed
    step budget.  Decay is triggered on demand — by setting
    ``decay_start_step`` ahead of time or retroactively via ``start_decay=True``
    when resuming from a checkpoint — so multiple decayed checkpoints can be
    produced from a single stable-phase run.

    The schedule has three sequential phases:

    1. **Warmup** — linear ramp from 0 to ``base_lr`` over ``warmup_steps``.
    2. **Stable** — holds ``base_lr`` indefinitely until decay is triggered.
    3. **Decay** — harmonic/rational decay from ``base_lr`` to ``min_lr``
       over ``decay_steps`` using linear interpolation of inverse LR.  The
       curve drops quickly at first then slows (convex shape).

    Notes
    -----
    ``start_decay``, ``min_lr``, and ``decay_steps`` are *config-only* keys:
    they are taken from the constructor arguments and are not saved to or
    loaded from checkpoints.  This ensures backward compatibility and allows
    the decay policy to be changed when resuming.

    References
    ----------
    Hu, S. et al. (2024). Understanding Warmup-Stable-Decay Learning Rates:
    A River Valley Loss Landscape Perspective. arXiv:2410.05192.
    """

    # Config-only keys: set from constructor config, not saved/loaded
    # from checkpoints.
    _CONFIG_ONLY_KEYS = frozenset(("start_decay", "min_lr", "decay_steps"))

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        min_lr: float = 1e-8,
        decay_steps: int = 1,
        decay_start_step: int = -1,
        start_decay: bool = False,
        last_epoch: int = -1,
    ):
        """
        Parameters
        ----------
        optimizer : Optimizer
            Wrapped optimizer whose ``param_groups`` LRs will be managed.
        warmup_steps : int, optional
            Number of steps for linear warmup (phase 1).  Default is 0.
        min_lr : float, optional
            Target minimum learning rate reached at the end of decay.  Must
            be > 0.  Config-only: not saved in checkpoints.  Default is 1e-8.
        decay_steps : int, optional
            Total number of steps in the decay phase.  The LR reaches
            ``min_lr`` after exactly this many steps past
            ``decay_start_step``. Config-only: not saved in
            checkpoints.  Default is 1.
        decay_start_step : int, optional
            Step at which to begin decay (phase 3).  Set to ``-1`` to disable
            decay.  When enabled, must be >= ``warmup_steps``.  Default is -1.
        start_decay : bool, optional
            When ``True``, decay begins at the current step upon loading a
            checkpoint (provided ``decay_start_step`` is still ``-1``).  This
            allows triggering decay retroactively from any saved checkpoint.
            When ``False``, ``decay_start_step`` is restored from the
            constructor value.  Config-only: not saved in checkpoints.
            Default is ``False``.
        last_epoch : int, optional
            Index of the last epoch, used when resuming.  Default is -1.
        """
        assert warmup_steps >= 0
        assert min_lr > 0.0
        # decay_steps may be 0 only when the decay phase is disabled (no
        # decay_start_step and not starting decay now). Enabling decay needs a
        # positive decay window.
        assert decay_steps > 0 or (decay_start_step < 0 and not start_decay), (
            f"decay_steps must be > 0 to run a decay phase (got {decay_steps}); "
            f"set annealing_tokens > 0, or disable decay "
            f"(decay_start_step < 0 and start_decay=False)."
        )
        assert decay_start_step < 0 or decay_start_step >= warmup_steps, (
            f"decay_start_step ({decay_start_step}) must be >= warmup_steps "
            f"({warmup_steps}); annealing would otherwise overlap warmup. "
            f"Reduce annealing_tokens or warmup_tokens."
        )

        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.decay_steps = decay_steps
        self.decay_start_step = decay_start_step
        self.start_decay = start_decay

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for the current step."""
        if self.last_epoch < self.warmup_steps:
            return self._warmup_lr()
        elif self.decay_start_step >= 0 and self.last_epoch >= self.decay_start_step:
            return self._decay_lr()
        else:
            return self._stable_lr()

    def _warmup_lr(self):
        """Phase 1: Linear warmup from 0 to base_lr."""
        return [
            base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs
        ]

    def _stable_lr(self):
        """Phase 2: Constant learning rate at base_lr."""
        return list(self.base_lrs)

    def _decay_lr(self):
        """Phase 3: Harmonic/rational decay from base_lr to min_lr.

        Uses linear interpolation of inverse LR:
            t = step / decay_steps   (progress 0..1)
            lr = 1 / (t / min_lr + (1 - t) / base_lr)

        At step=0: lr = base_lr (smooth transition from stable phase).
        At step=decay_steps: lr = min_lr (exact target).
        Past decay_steps: clamped at min_lr.
        """
        step = self.last_epoch - self.decay_start_step
        t = min(step / self.decay_steps, 1.0)
        return [
            1.0 / (t / self.min_lr + (1.0 - t) / base_lr) for base_lr in self.base_lrs
        ]

    def state_dict(self):
        """Return state dict excluding config-only parameters."""
        return {
            key: value
            for key, value in super().state_dict().items()
            if key not in self._CONFIG_ONLY_KEYS
        }

    def load_state_dict(self, state_dict):
        """Load state dict, preserving config-only parameters.

        The start_decay flag controls how decay_start_step is resolved:

        - start_decay=True with loaded decay_start_step < 0:
          Begin decay at the current step (last_epoch).
        - start_decay=True with loaded decay_start_step >= 0:
          Resume decay from where it left off.
        - start_decay=False:
          Restore decay_start_step from the constructor value,
          ignoring whatever was saved in the checkpoint.
        """
        saved_config = {key: getattr(self, key) for key in self._CONFIG_ONLY_KEYS}
        saved_decay_start_step = self.decay_start_step

        super().load_state_dict(state_dict)

        for key, value in saved_config.items():
            setattr(self, key, value)

        if self.start_decay:
            if self.decay_start_step < 0:
                self.decay_start_step = self.last_epoch
        else:
            self.decay_start_step = saved_decay_start_step
