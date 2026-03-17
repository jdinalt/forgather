"""
Divergence detection callbacks for catching training instabilities early.

This module provides callbacks that monitor training metrics and detect divergence
(sustained loss increases, gradient explosions, etc.) before thousands of steps are wasted.
"""

import logging
import math
from typing import Literal

from torch.distributed.checkpoint.stateful import Stateful

from ..trainer_types import TrainerCallback

logger = logging.getLogger(__name__)


class DivergenceDetector(TrainerCallback, Stateful):
    """
    Detects training divergence by comparing smoothed loss against its best observed value.

    Maintains a smoothed loss (EMA) and tracks its running minimum. Triggers when
    the smoothed loss exceeds the baseline minimum by a configurable threshold,
    sustained for ``patience`` consecutive observations.

    Supports absolute threshold (smoothed - best >= threshold), relative threshold
    (smoothed >= best * factor), or both simultaneously (triggers on whichever
    fires first).

    Also detects NaN/Inf loss values immediately (no patience required).

    Defaults are calibrated against real training runs where loss decreases from
    ~10 to ~3.8 then spikes to ~9.7 on divergence. With default settings
    (smoothing=0.3, threshold=1.0, patience=3), divergence is detected within
    3 log entries (~96 training steps at 32-step log intervals) of the spike,
    with zero false positives on healthy runs.

    Example:
        >>> detector = DivergenceDetector(
        ...     smoothing=0.3,        # EMA alpha (higher = more responsive)
        ...     threshold=1.0,        # Absolute: stop if smoothed - best >= 1.0
        ...     patience=3,           # Require 3 consecutive observations
        ...     action="abort",
        ... )
        >>> trainer = Trainer(..., callbacks=[detector])
        >>> trainer.train()

    Using relative threshold (e.g., 50% increase from best):
        >>> detector = DivergenceDetector(
        ...     smoothing=0.3,
        ...     relative_threshold=1.5,  # Stop if smoothed >= 1.5 * best
        ...     patience=3,
        ...     action="stop",
        ... )
    """

    def __init__(
        self,
        smoothing: float = 0.3,
        threshold: float | None = 1.0,
        relative_threshold: float | None = None,
        patience: int = 3,
        warmup: int = 10,
        action: Literal["stop", "abort"] = "stop",
        use_eval_loss: bool = False,
        metric_key: str | None = None,
    ):
        """
        Initialize divergence detector.

        Args:
            smoothing: EMA alpha for smoothing raw loss (0-1). Higher = more responsive
                      to recent values. Effective window ~ 1/alpha observations.
            threshold: Absolute divergence threshold. Triggers when
                      (smoothed_loss - best_smoothed_loss) >= threshold.
                      Set to None to disable absolute threshold.
            relative_threshold: Relative divergence threshold. Triggers when
                               smoothed_loss >= best_smoothed_loss * relative_threshold.
                               For example, 1.5 means "50% increase from best".
                               Set to None to disable relative threshold.
            patience: Number of consecutive observations above threshold required
                     before triggering. Higher values reduce false positives from
                     transient spikes. Set to 1 for immediate triggering.
            warmup: Number of initial observations to skip before checking divergence.
                   Avoids false positives from the high-loss early training phase.
            action: What to do when divergence detected:
                   - "stop": Gracefully stop training (saves checkpoint first)
                   - "abort": Abort immediately without saving
            use_eval_loss: If True, monitor eval_loss. If False, monitor train loss.
                          Defaults to False because train loss is logged much more
                          frequently, enabling faster detection. Set to True if you
                          have frequent evaluation and want less noisy signals.
            metric_key: Optional custom metric key to monitor (overrides use_eval_loss).
        """
        super().__init__()

        if not 0 < smoothing <= 1:
            raise ValueError(f"smoothing must be in (0, 1], got {smoothing}")
        if threshold is not None and threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if relative_threshold is not None and relative_threshold <= 1:
            raise ValueError(
                f"relative_threshold must be > 1, got {relative_threshold}"
            )
        if threshold is None and relative_threshold is None:
            raise ValueError(
                "At least one of threshold or relative_threshold must be set"
            )
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")

        self.smoothing = smoothing
        self.threshold = threshold
        self.relative_threshold = relative_threshold
        self.patience = patience
        self.warmup = warmup
        self.action = action
        self.use_eval_loss = use_eval_loss
        self.metric_key = metric_key

        # State
        self.smoothed_loss: float | None = None
        self.best_smoothed_loss: float | None = None
        self.observation_count: int = 0
        self.consecutive_above: int = 0

    def _get_metric(self, data):
        """Extract the monitored metric from a data dict. Returns (value, key_name)."""
        if self.metric_key:
            return data.get(self.metric_key), self.metric_key
        elif self.use_eval_loss:
            return data.get("eval_loss"), "eval_loss"
        else:
            return data.get("loss"), "loss"

    def _check_divergence(self, args, state, control, logs=None, metrics=None):
        """Check for divergence given metrics dict."""
        data = logs or metrics
        if not data:
            return control

        loss, key_name = self._get_metric(data)
        if loss is None:
            return control

        # Detect NaN/Inf immediately
        if math.isnan(loss) or math.isinf(loss):
            if state is None or state.is_world_process_zero:
                logger.error(
                    f"Training divergence detected! {key_name}={loss} (NaN/Inf)\n"
                    f"Action: {self.action}"
                )
            self._trigger_action(control)
            return control

        self.observation_count += 1

        # Update smoothed loss
        if self.smoothed_loss is None:
            self.smoothed_loss = loss
            self.best_smoothed_loss = loss
            if state is None or state.is_world_process_zero:
                logger.info(
                    f"DivergenceDetector initialized with {key_name}={loss:.4f}"
                )
            return control

        # Both are guaranteed non-None after the initialization branch above
        assert self.smoothed_loss is not None
        assert self.best_smoothed_loss is not None

        smoothed = self.smoothing * loss + (1 - self.smoothing) * self.smoothed_loss
        self.smoothed_loss = smoothed

        # Update best (only track improvements)
        best = self.best_smoothed_loss
        if smoothed < best:
            best = smoothed
            self.best_smoothed_loss = best

        # Skip warmup period
        if self.observation_count <= self.warmup:
            return control

        # Check thresholds
        above = False
        abs_divergence = smoothed - best
        rel_ratio = smoothed / best if best > 0 else 0.0

        if self.threshold is not None and abs_divergence >= self.threshold:
            above = True
        if self.relative_threshold is not None and rel_ratio >= self.relative_threshold:
            above = True

        logger.debug(
            f"Divergence detector: {key_name}={loss:.4f}, "
            f"smoothed={smoothed:.4f}, best={best:.4f}, "
            f"abs_div={abs_divergence:.4f}, rel={rel_ratio:.4f}, "
            f"consecutive={self.consecutive_above}/{self.patience}"
        )

        if above:
            self.consecutive_above += 1
            if self.consecutive_above >= self.patience:
                if state is None or state.is_world_process_zero:
                    parts = [
                        f"Training divergence detected! {key_name}={loss:.4f}",
                        f"Smoothed loss: {smoothed:.4f}",
                        f"Best smoothed loss: {best:.4f}",
                    ]
                    if self.threshold is not None:
                        parts.append(
                            f"Absolute divergence: {abs_divergence:.4f} "
                            f"(threshold: {self.threshold:.4f})"
                        )
                    if self.relative_threshold is not None:
                        parts.append(
                            f"Relative ratio: {rel_ratio:.4f} "
                            f"(threshold: {self.relative_threshold:.4f})"
                        )
                    parts.append(f"Action: {self.action}")
                    logger.error("\n".join(parts))

                self._trigger_action(control)
        else:
            self.consecutive_above = 0

        return control

    def _trigger_action(self, control):
        """Apply the configured action to the trainer control."""
        if self.action == "stop":
            control.should_training_stop = True
        elif self.action == "abort":
            control.should_training_stop = True
            if hasattr(control, "should_abort_without_save"):
                control.should_abort_without_save = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Check for divergence when training metrics are logged."""
        return self._check_divergence(args, state, control, logs=logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check for divergence when evaluation metrics are available."""
        return self._check_divergence(args, state, control, metrics=metrics)

    def state_dict(self):
        """Return callback state to save with checkpoint."""
        return {
            "smoothed_loss": self.smoothed_loss,
            "best_smoothed_loss": self.best_smoothed_loss,
            "observation_count": self.observation_count,
            "consecutive_above": self.consecutive_above,
        }

    def load_state_dict(self, state_dict):
        """Restore callback state from checkpoint."""
        self.smoothed_loss = state_dict["smoothed_loss"]
        self.best_smoothed_loss = state_dict["best_smoothed_loss"]
        self.observation_count = state_dict["observation_count"]
        self.consecutive_above = state_dict["consecutive_above"]
        if self.smoothed_loss is not None and self.best_smoothed_loss is not None:
            logger.debug(
                f"Restored DivergenceDetector state: "
                f"smoothed={self.smoothed_loss:.4f}, best={self.best_smoothed_loss:.4f}, "
                f"observations={self.observation_count}"
            )


# Keep old names as aliases for backward compatibility.
# These are deprecated; prefer DivergenceDetector.
DualTimeScaleDivergenceDetector = DivergenceDetector
DualWindowDivergenceDetector = DivergenceDetector
