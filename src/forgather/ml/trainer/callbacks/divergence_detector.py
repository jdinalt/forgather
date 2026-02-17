"""
Divergence detection callbacks for catching training instabilities early.

This module provides callbacks that monitor training metrics and detect divergence
(sustained loss increases, gradient explosions, etc.) before thousands of steps are wasted.
"""

import logging
from typing import Literal

from torch.distributed.checkpoint.stateful import Stateful

from ..trainer_types import TrainerCallback, TrainerControl

logger = logging.getLogger(__name__)


class DualTimeScaleDivergenceDetector(TrainerCallback, Stateful):
    """
    Detects training divergence using dual-timescale EMA filtering.

    Maintains two exponential moving averages:
    - Fast EMA (short_alpha): Responds quickly to loss changes
    - Slow EMA (long_alpha): Tracks long-term baseline

    Triggers when fast EMA significantly exceeds slow EMA, indicating sustained
    divergence rather than transient spikes.

    This is a classic signal processing approach for detecting sustained changes
    vs transient noise. The fast filter tracks current state, the slow filter
    tracks baseline - divergence is when they separate significantly.

    Example:
        >>> detector = DualTimeScaleDivergenceDetector(
        ...     short_alpha=0.1,      # ~10 step effective window
        ...     long_alpha=0.01,      # ~100 step effective window
        ...     threshold=2.0,        # Stop if short - long >= 2.0
        ...     action="stop",
        ...     use_eval_loss=True,   # Monitor eval_loss (more stable)
        ... )
        >>> trainer = Trainer(..., callbacks=[detector])
        >>> trainer.train()

    Tuning for quick detection of spikes (e.g., loss 2.8 → 8.0):
        >>> detector = DualTimeScaleDivergenceDetector(
        ...     short_alpha=0.2,      # Faster response (~5 steps)
        ...     long_alpha=0.01,      # Stable baseline (~100 steps)
        ...     threshold=1.5,        # More sensitive threshold
        ...     action="stop",
        ...     use_eval_loss=True,
        ... )
        # This will detect within ~10-20 eval steps instead of 50,000 train steps!
    """

    def __init__(
        self,
        short_alpha: float = 0.1,
        long_alpha: float = 0.01,
        threshold: float = 2.0,
        action: Literal["stop", "abort"] = "stop",
        use_eval_loss: bool = True,
        metric_key: str | None = None,
    ):
        """
        Initialize divergence detector.

        Args:
            short_alpha: EMA decay for fast filter (0-1). Higher = faster response.
                        Effective window ≈ 1/alpha steps.
            long_alpha: EMA decay for slow filter (0-1). Lower = slower response.
                        Effective window ≈ 1/alpha steps.
            threshold: Divergence threshold. Triggers when (short - long) >= threshold.
            action: What to do when divergence detected:
                   - "stop": Gracefully stop training (saves checkpoint first)
                   - "abort": Abort immediately without saving
            use_eval_loss: If True, monitor eval_loss. If False, monitor train loss.
                          eval_loss is more stable (no gradient noise).
            metric_key: Optional custom metric key to monitor (overrides use_eval_loss).
        """
        super().__init__()

        if not 0 < short_alpha <= 1:
            raise ValueError(f"short_alpha must be in (0, 1], got {short_alpha}")
        if not 0 < long_alpha <= 1:
            raise ValueError(f"long_alpha must be in (0, 1], got {long_alpha}")
        if short_alpha <= long_alpha:
            logger.warning(
                f"short_alpha ({short_alpha}) should be > long_alpha ({long_alpha}) "
                "for proper dual-timescale filtering"
            )

        self.short_alpha = short_alpha
        self.long_alpha = long_alpha
        self.threshold = threshold
        self.action = action
        self.use_eval_loss = use_eval_loss
        self.metric_key = metric_key

        # EMA state
        self.short_ema: float | None = None
        self.long_ema: float | None = None
        self.initialized = False

    def _check_divergence(self, args, state, control, logs=None, metrics=None):
        """
        Check for divergence given metrics dict.

        Can be called from either on_log or on_evaluate.
        """
        data = logs or metrics
        if not data:
            return control

        # Determine which metric to monitor
        if self.metric_key:
            loss = data.get(self.metric_key)
            key_name = self.metric_key
        elif self.use_eval_loss:
            loss = data.get("eval_loss")
            key_name = "eval_loss"
        else:
            loss = data.get("loss")
            key_name = "loss"

        if loss is None:
            return control

        # Initialize EMAs on first observation
        if not self.initialized:
            self.short_ema = loss
            self.long_ema = loss
            self.initialized = True
            if state is None or state.is_world_process_zero:
                logger.info(
                    f"DualTimeScaleDivergenceDetector initialized with {key_name}={loss:.4f}"
                )
            return control

        # Update EMAs (both are guaranteed non-None after the initialization branch above)
        assert self.short_ema is not None
        assert self.long_ema is not None
        new_short = self.short_alpha * loss + (1 - self.short_alpha) * self.short_ema
        new_long = self.long_alpha * loss + (1 - self.long_alpha) * self.long_ema
        self.short_ema = new_short
        self.long_ema = new_long

        # Check for divergence
        divergence = new_short - new_long

        logger.debug(
            f"Divergence detector: {key_name}={loss:.4f}, "
            f"short_ema={self.short_ema:.4f}, long_ema={self.long_ema:.4f}, "
            f"divergence={divergence:.4f}"
        )

        if divergence >= self.threshold:
            if state is None or state.is_world_process_zero:
                logger.error(
                    f"Training divergence detected! {key_name}={loss:.4f}\n"
                    f"Short EMA: {self.short_ema:.4f} (window ~{1/self.short_alpha:.0f} steps)\n"
                    f"Long EMA: {self.long_ema:.4f} (window ~{1/self.long_alpha:.0f} steps)\n"
                    f"Divergence: {divergence:.4f} >= threshold {self.threshold:.4f}\n"
                    f"Action: {self.action}"
                )

            if self.action == "stop":
                control.should_training_stop = True
            elif self.action == "abort":
                # Set a custom control flag for aborting without save
                # Trainers can check this and skip final checkpoint save
                control.should_training_stop = True
                if hasattr(control, "should_abort_without_save"):
                    control.should_abort_without_save = True

        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Update EMA filters when training metrics are logged."""
        return self._check_divergence(args, state, control, logs=logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Update EMA filters when evaluation metrics are available."""
        return self._check_divergence(args, state, control, metrics=metrics)

    def state_dict(self):
        """Return callback state to save with checkpoint."""
        return {
            "short_ema": self.short_ema,
            "long_ema": self.long_ema,
            "initialized": self.initialized,
        }

    def load_state_dict(self, state_dict):
        """Restore callback state from checkpoint."""
        self.short_ema = state_dict["short_ema"]
        self.long_ema = state_dict["long_ema"]
        self.initialized = state_dict["initialized"]
        logger.debug(
            f"Restored DualTimeScaleDivergenceDetector state: "
            f"short_ema={self.short_ema:.4f}, long_ema={self.long_ema:.4f}"
        )


class DualWindowDivergenceDetector(TrainerCallback, Stateful):
    """
    Detects training divergence using dual-window moving averages (FIR filter).

    Similar to DualTimeScaleDivergenceDetector but uses finite windows instead
    of exponential decay. This can be more intuitive to tune ("last 10 steps"
    vs "alpha=0.1") and has exact cutoff at window size.

    Example:
        >>> detector = DualWindowDivergenceDetector(
        ...     short_window=10,     # Recent average
        ...     long_window=100,     # Long-term average
        ...     threshold=2.0,
        ... )
    """

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 100,
        threshold: float = 2.0,
        action: Literal["stop", "abort"] = "stop",
        use_eval_loss: bool = True,
        metric_key: str | None = None,
    ):
        """
        Initialize dual-window divergence detector.

        Args:
            short_window: Number of recent observations for short average
            long_window: Number of observations for long-term average
            threshold: Divergence threshold. Triggers when (short - long) >= threshold.
            action: What to do when divergence detected ("stop" or "abort")
            use_eval_loss: If True, monitor eval_loss. If False, monitor train loss.
            metric_key: Optional custom metric key to monitor (overrides use_eval_loss).
        """
        super().__init__()

        if short_window <= 0:
            raise ValueError(f"short_window must be > 0, got {short_window}")
        if long_window <= 0:
            raise ValueError(f"long_window must be > 0, got {long_window}")
        if short_window >= long_window:
            logger.warning(
                f"short_window ({short_window}) should be < long_window ({long_window}) "
                "for proper dual-timescale filtering"
            )

        self.short_window = short_window
        self.long_window = long_window
        self.threshold = threshold
        self.action = action
        self.use_eval_loss = use_eval_loss
        self.metric_key = metric_key

        # Window state (circular buffers)
        self.short_buffer: list[float] = []
        self.long_buffer: list[float] = []

    def _check_divergence(self, args, state, control, logs=None, metrics=None):
        """
        Check for divergence given metrics dict.

        Can be called from either on_log or on_evaluate.
        """
        data = logs or metrics
        if not data:
            return control

        # Determine which metric to monitor
        if self.metric_key:
            loss = data.get(self.metric_key)
            key_name = self.metric_key
        elif self.use_eval_loss:
            loss = data.get("eval_loss")
            key_name = "eval_loss"
        else:
            loss = data.get("loss")
            key_name = "loss"

        if loss is None:
            return control

        # Add to buffers
        self.short_buffer.append(loss)
        self.long_buffer.append(loss)

        # Trim to window size
        if len(self.short_buffer) > self.short_window:
            self.short_buffer.pop(0)
        if len(self.long_buffer) > self.long_window:
            self.long_buffer.pop(0)

        # Wait until long buffer is full before checking divergence
        if len(self.long_buffer) < self.long_window:
            return control

        # Compute averages
        short_avg = sum(self.short_buffer) / len(self.short_buffer)
        long_avg = sum(self.long_buffer) / len(self.long_buffer)

        # Check for divergence
        divergence = short_avg - long_avg

        logger.debug(
            f"Divergence detector: {key_name}={loss:.4f}, "
            f"short_avg={short_avg:.4f}, long_avg={long_avg:.4f}, "
            f"divergence={divergence:.4f}"
        )

        if divergence >= self.threshold:
            if state is None or state.is_world_process_zero:
                logger.error(
                    f"Training divergence detected! {key_name}={loss:.4f}\n"
                    f"Short average (last {self.short_window} steps): {short_avg:.4f}\n"
                    f"Long average (last {self.long_window} steps): {long_avg:.4f}\n"
                    f"Divergence: {divergence:.4f} >= threshold {self.threshold:.4f}\n"
                    f"Action: {self.action}"
                )

            if self.action == "stop":
                control.should_training_stop = True
            elif self.action == "abort":
                control.should_training_stop = True
                if hasattr(control, "should_abort_without_save"):
                    control.should_abort_without_save = True

        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Update moving averages when training metrics are logged."""
        return self._check_divergence(args, state, control, logs=logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Update moving averages when evaluation metrics are available."""
        return self._check_divergence(args, state, control, metrics=metrics)

    def state_dict(self):
        """Return callback state to save with checkpoint."""
        return {
            "short_buffer": self.short_buffer,
            "long_buffer": self.long_buffer,
        }

    def load_state_dict(self, state_dict):
        """Restore callback state from checkpoint."""
        self.short_buffer = state_dict["short_buffer"]
        self.long_buffer = state_dict["long_buffer"]
        logger.debug(
            f"Restored DualWindowDivergenceDetector state: "
            f"{len(self.short_buffer)} short samples, {len(self.long_buffer)} long samples"
        )
