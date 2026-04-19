"""
Divergence Detection Examples

Demonstrates how to detect training divergence early using stateful callbacks.
"""

from forgather.ml.trainer import Trainer, TrainingArguments
from forgather.ml.trainer.callbacks import (
    DualTimeScaleDivergenceDetector,
    DualWindowDivergenceDetector,
)


def example_basic_divergence_detection():
    """Basic divergence detection with dual-timescale EMA."""

    # Configure divergence detector
    divergence_detector = DualTimeScaleDivergenceDetector(
        short_alpha=0.1,         # Fast EMA (~10 step effective window)
        long_alpha=0.01,         # Slow EMA (~100 step effective window)
        threshold=2.0,           # Stop if short - long >= 2.0
        action="stop",           # Stop gracefully (saves checkpoint first)
        use_eval_loss=True,      # Monitor eval_loss (more stable than train loss)
    )

    args = TrainingArguments(
        output_dir="output_models/my_model",
        save_strategy="steps",
        save_steps=1000,
        preserve_best_model=True,
        preserve_n_best=2,
        eval_strategy="steps",
        eval_steps=500,
        eval_on_save=True,
    )

    # trainer = Trainer(
    #     model=model,
    #     args=args,
    #     train_dataset=train_ds,
    #     eval_dataset=eval_ds,
    #     callbacks=[divergence_detector],
    # )
    # trainer.train()

    # If loss spikes (e.g., 2.8 → 8.0):
    # - Detector triggers within ~10-20 eval steps
    # - Training stops gracefully
    # - Final checkpoint saved (includes detector state)
    # - Best checkpoint preserved
    print("✓ Basic divergence detection configured")


def example_quick_spike_detection():
    """Tune divergence detector for fast response to loss spikes."""

    # For case where loss spikes from 2.8 to 8.0 and stays high:
    divergence_detector = DualTimeScaleDivergenceDetector(
        short_alpha=0.2,         # Faster response (~5 steps)
        long_alpha=0.01,         # Stable baseline (~100 steps)
        threshold=1.5,           # More sensitive (stop if short - long >= 1.5)
        action="stop",
        use_eval_loss=True,
    )

    args = TrainingArguments(
        output_dir="output_models/my_model",
        save_steps=1000,
        eval_steps=500,          # Eval every 500 steps
        eval_on_save=True,
        preserve_best_model=True,
    )

    # This will detect within ~10-20 eval steps (5,000-10,000 train steps)
    # instead of 50,000 train steps!
    print("✓ Quick spike detection configured")


def example_window_based_detection():
    """Use window-based moving averages instead of EMA."""

    # More intuitive than EMA - specify exact window sizes
    divergence_detector = DualWindowDivergenceDetector(
        short_window=10,         # Recent average (last 10 observations)
        long_window=100,         # Long-term average (last 100 observations)
        threshold=2.0,
        action="stop",
        use_eval_loss=True,
    )

    args = TrainingArguments(
        output_dir="output_models/my_model",
        save_steps=1000,
        eval_steps=500,
        eval_on_save=True,
        preserve_best_model=True,
    )

    print("✓ Window-based detection configured")


def example_custom_metric_monitoring():
    """Monitor custom metrics instead of loss."""

    # Monitor gradient norm for instability
    divergence_detector = DualTimeScaleDivergenceDetector(
        short_alpha=0.1,
        long_alpha=0.01,
        threshold=5.0,           # Higher threshold for grad norm
        action="stop",
        use_eval_loss=False,     # Don't use eval_loss
        metric_key="grad_norm",  # Monitor custom metric
    )

    args = TrainingArguments(
        output_dir="output_models/my_model",
        save_steps=1000,
        eval_steps=500,
        eval_on_save=True,
        preserve_best_model=True,
    )

    print("✓ Custom metric monitoring configured")


def example_custom_stateful_callback():
    """Implement custom divergence detector with Stateful protocol."""
    from forgather.ml.trainer.trainer_types import TrainerCallback
    from torch.distributed.checkpoint.stateful import Stateful
    import logging

    logger = logging.getLogger(__name__)

    class GradualDivergenceDetector(TrainerCallback, Stateful):
        """Detect slow degradation over long time horizons."""

        def __init__(self, lookback_steps=100, degradation_threshold=0.1):
            self.lookback_steps = lookback_steps
            self.degradation_threshold = degradation_threshold
            self.loss_history = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or 'eval_loss' not in logs:
                return control

            loss = logs['eval_loss']
            self.loss_history.append(loss)

            if len(self.loss_history) > self.lookback_steps:
                self.loss_history.pop(0)

            if len(self.loss_history) == self.lookback_steps:
                # Check if recent average > early average by threshold
                early_avg = sum(self.loss_history[:20]) / 20
                recent_avg = sum(self.loss_history[-20:]) / 20

                if recent_avg > early_avg * (1 + self.degradation_threshold):
                    logger.error(f"Gradual divergence: {early_avg:.3f} → {recent_avg:.3f}")
                    control.should_training_stop = True

            return control

        def state_dict(self):
            """Save callback state to checkpoint."""
            return {'loss_history': self.loss_history}

        def load_state_dict(self, state_dict):
            """Restore callback state from checkpoint."""
            self.loss_history = state_dict['loss_history']

    # Usage
    detector = GradualDivergenceDetector(lookback_steps=100, degradation_threshold=0.1)

    # trainer = Trainer(..., callbacks=[detector])
    print("✓ Custom stateful callback configured")


if __name__ == "__main__":
    print("=" * 60)
    print("Divergence Detection Examples")
    print("=" * 60)
    print()

    example_basic_divergence_detection()
    example_quick_spike_detection()
    example_window_based_detection()
    example_custom_metric_monitoring()
    example_custom_stateful_callback()

    print()
    print("=" * 60)
    print("All examples configured successfully!")
    print("=" * 60)
