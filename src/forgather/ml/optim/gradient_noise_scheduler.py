"""
Adaptive LR scheduler that uses gradient norm standard deviation as a feedback
signal to maintain training stability in bf16 precision.

During bf16 training with Adafactor (no momentum, stochastic rounding),
accumulated quantization noise in the weights causes the gradient norm std to
grow over training, eventually degrading the signal-to-noise ratio. This
scheduler implements an integral feedback controller in log-LR space that
slowly adjusts the learning rate to keep the gradient norm std near a target.

Gradient spikes can corrupt the EMA statistics and cause the controller to
overreact. An optional spike filter (enabled by default) discards samples
that exceed ``current_mean + spike_threshold_std * current_std``. The filter
only activates after ``min_samples_for_spike_filter`` steps have been
collected, so that early noisy statistics don't cause false positives.

Usage:
    scheduler = GradientNoiseScheduler(optimizer, warmup_steps=5000)

    for step in training_loop:
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step(grad_norm)
"""

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class GradientNoiseScheduler(LRScheduler):
    """
    LR scheduler with integral feedback on gradient norm standard deviation.

    Operates in two phases:

    1. **Warmup** (first ``warmup_steps`` calls with a grad_norm value):
       Collects gradient norm statistics via EMA without modifying the LR.
       At the end of warmup, if ``target_std`` was not provided, it is
       automatically calibrated from the accumulated statistics.

    2. **Active** (after warmup): Uses an integral controller in log-LR space
       to adjust the LR multiplicatively. When the current gradient norm std
       exceeds the target, the LR is gradually reduced. When it is below the
       target, the LR is gradually increased (if ``max_lr_scale`` allows).

    The feedback operates on a very slow timescale (thousands of steps) to
    avoid destabilizing training from abrupt LR changes.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of steps (with grad_norm provided) to collect
            statistics before activating feedback. LR is unchanged during
            this window.
        target_std: Target gradient norm standard deviation. If ``None``,
            auto-calibrated from the EMA statistics at the end of warmup.
        ema_decay: Decay rate for exponential moving averages. Values close
            to 1.0 give longer smoothing windows. Default ``0.9999`` gives
            an effective window of ~10,000 steps.
        feedback_strength: Integral gain (ki). Controls how aggressively the
            LR responds to std deviations from target. Default ``1e-5`` gives
            very gradual adjustment.
        min_lr_scale: Floor on the multiplicative LR scale factor.
        max_lr_scale: Ceiling on the multiplicative LR scale factor. ``None``
            means no ceiling (LR can increase if std < target). Set to ``1.0``
            to prevent the LR from exceeding the base LR.
        spike_threshold_std: Number of standard deviations above the EMA mean
            beyond which a gradient norm sample is considered a spike and
            discarded. Default ``1.645`` corresponds to the ~95th percentile
            of a normal distribution. Set to ``None`` to disable spike
            filtering entirely.
        min_samples_for_spike_filter: Minimum number of feedback steps before
            spike filtering activates. Early statistics are too noisy for
            reliable outlier detection. Default ``100``.
        last_epoch: Standard PyTorch scheduler parameter for resuming.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 5000,
        target_std: Optional[float] = None,
        ema_decay: float = 0.9999,
        feedback_strength: float = 1e-5,
        min_lr_scale: float = 0.01,
        max_lr_scale: Optional[float] = None,
        spike_threshold_std: Optional[float] = 1.645,
        min_samples_for_spike_filter: int = 100,
        last_epoch: int = -1,
    ):
        assert warmup_steps >= 0, f"warmup_steps must be >= 0, got {warmup_steps}"
        assert 0.0 < ema_decay < 1.0, f"ema_decay must be in (0, 1), got {ema_decay}"
        assert (
            feedback_strength > 0.0
        ), f"feedback_strength must be > 0, got {feedback_strength}"
        assert min_lr_scale >= 0.0, f"min_lr_scale must be >= 0, got {min_lr_scale}"
        if max_lr_scale is not None:
            assert max_lr_scale > 0.0, f"max_lr_scale must be > 0, got {max_lr_scale}"
            assert (
                max_lr_scale >= min_lr_scale
            ), f"max_lr_scale ({max_lr_scale}) must be >= min_lr_scale ({min_lr_scale})"
        if target_std is not None:
            assert target_std > 0.0, f"target_std must be > 0, got {target_std}"
        if spike_threshold_std is not None:
            assert (
                spike_threshold_std > 0.0
            ), f"spike_threshold_std must be > 0, got {spike_threshold_std}"
        assert (
            min_samples_for_spike_filter >= 0
        ), f"min_samples_for_spike_filter must be >= 0, got {min_samples_for_spike_filter}"

        self.warmup_steps = warmup_steps
        self._target_std = target_std
        self.ema_decay = ema_decay
        self.feedback_strength = feedback_strength
        self.min_lr_scale = min_lr_scale
        self.max_lr_scale = max_lr_scale
        self.spike_threshold_std = spike_threshold_std
        self.min_samples_for_spike_filter = min_samples_for_spike_filter

        # EMA state
        self._gn_ema = 0.0
        self._gn_sq_ema = 0.0
        self._log_lr_scale = 0.0
        self._feedback_step = 0

        # Must be set before super().__init__() which calls get_lr()
        super().__init__(optimizer, last_epoch)

    def step(self, grad_norm=None, epoch=None):
        """
        Advance the scheduler by one step.

        Args:
            grad_norm: The total gradient norm for this step (pre-clipping
                recommended). Accepts float or scalar tensor. If ``None``,
                the feedback state is not updated but the base class
                bookkeeping (last_epoch, etc.) still advances.
            epoch: Deprecated PyTorch scheduler parameter.
        """
        if grad_norm is not None:
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
            self._update_feedback(grad_norm)
        super().step(epoch)

    def _update_feedback(self, gn: float):
        """Update EMA statistics and integral controller.

        If spike filtering is active and the sample exceeds the spike
        threshold, the sample is discarded entirely -- no state is updated.
        """
        # Spike filtering: discard outliers once sufficient samples exist
        if (
            self.spike_threshold_std is not None
            and self._feedback_step >= self.min_samples_for_spike_filter
        ):
            threshold = self._spike_threshold()
            if threshold > 0.0 and gn > threshold:
                return

        self._feedback_step += 1
        beta = self.ema_decay

        # Update EMAs
        self._gn_ema = beta * self._gn_ema + (1.0 - beta) * gn
        self._gn_sq_ema = beta * self._gn_sq_ema + (1.0 - beta) * gn * gn

        # Auto-calibrate target at end of warmup
        if self._feedback_step == self.warmup_steps and self._target_std is None:
            self._target_std = self._current_std()

        # No feedback during warmup
        if self._feedback_step <= self.warmup_steps:
            return

        # Need a valid target to apply feedback
        if self._target_std is None or self._target_std <= 0.0:
            return

        current_std = self._current_std()
        if current_std <= 0.0:
            return

        # Integral feedback in log space
        error = (current_std - self._target_std) / self._target_std
        self._log_lr_scale -= self.feedback_strength * error

        # Apply bounds
        if self.min_lr_scale > 0.0:
            self._log_lr_scale = max(self._log_lr_scale, math.log(self.min_lr_scale))
        if self.max_lr_scale is not None:
            self._log_lr_scale = min(self._log_lr_scale, math.log(self.max_lr_scale))

    def _current_std(self) -> float:
        """Compute bias-corrected gradient norm std from EMAs."""
        n = self._feedback_step
        if n == 0:
            return 0.0

        bc = 1.0 - self.ema_decay**n
        mean = self._gn_ema / bc
        mean_sq = self._gn_sq_ema / bc

        variance = mean_sq - mean * mean
        if variance <= 0.0:
            return 0.0
        return math.sqrt(variance)

    def _spike_threshold(self) -> float:
        """Compute the spike detection threshold from current EMA statistics."""
        mean = self.current_mean
        std = self._current_std()
        if std <= 0.0:
            return 0.0
        return mean + self.spike_threshold_std * std

    @property
    def spike_threshold(self) -> float:
        """Current spike detection threshold, or 0.0 if filtering is inactive."""
        if self.spike_threshold_std is None:
            return 0.0
        if self._feedback_step < self.min_samples_for_spike_filter:
            return 0.0
        return self._spike_threshold()

    def get_lr(self):
        """Compute LRs with adaptive feedback scale applied."""
        scale = math.exp(self._log_lr_scale)
        return [base_lr * scale for base_lr in self.base_lrs]

    @property
    def lr_scale(self) -> float:
        """Current multiplicative LR scale factor."""
        return math.exp(self._log_lr_scale)

    @property
    def current_std(self) -> float:
        """Current bias-corrected gradient norm standard deviation estimate."""
        return self._current_std()

    @property
    def current_mean(self) -> float:
        """Current bias-corrected gradient norm mean estimate."""
        if self._feedback_step == 0:
            return 0.0
        bc = 1.0 - self.ema_decay**self._feedback_step
        return self._gn_ema / bc

    @property
    def target_std(self) -> Optional[float]:
        """Target gradient norm std (None if not yet calibrated)."""
        return self._target_std

    def state_dict(self):
        """Return scheduler state for checkpointing."""
        state = super().state_dict()
        state["gn_ema"] = self._gn_ema
        state["gn_sq_ema"] = self._gn_sq_ema
        state["log_lr_scale"] = self._log_lr_scale
        state["feedback_step"] = self._feedback_step
        state["target_std"] = self._target_std
        state["spike_threshold_std"] = self.spike_threshold_std
        state["min_samples_for_spike_filter"] = self.min_samples_for_spike_filter
        return state

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self._gn_ema = state_dict.pop("gn_ema", 0.0)
        self._gn_sq_ema = state_dict.pop("gn_sq_ema", 0.0)
        self._log_lr_scale = state_dict.pop("log_lr_scale", 0.0)
        self._feedback_step = state_dict.pop("feedback_step", 0)
        self._target_std = state_dict.pop("target_std", self._target_std)
        self.spike_threshold_std = state_dict.pop(
            "spike_threshold_std", self.spike_threshold_std
        )
        self.min_samples_for_spike_filter = state_dict.pop(
            "min_samples_for_spike_filter", self.min_samples_for_spike_filter
        )
        super().load_state_dict(state_dict)
