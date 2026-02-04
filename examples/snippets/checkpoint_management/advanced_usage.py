"""
Advanced Checkpoint Management Examples

Demonstrates advanced patterns combining multiple features for robust training.
"""

from forgather.ml.trainer import Trainer, TrainingArguments
from forgather.ml.trainer.callbacks import (
    DualTimeScaleDivergenceDetector,
    DualWindowDivergenceDetector,
    ProgressCallback,
)


def example_multiple_safeguards():
    """Combine multiple detection strategies for robust training."""

    # Combine multiple safeguards
    callbacks = [
        # Fast spike detection (EMA-based)
        DualTimeScaleDivergenceDetector(
            short_alpha=0.2,
            long_alpha=0.01,
            threshold=1.5,
            action="stop",
            use_eval_loss=True,
        ),

        # Gradual degradation detection (window-based)
        DualWindowDivergenceDetector(
            short_window=10,
            long_window=100,
            threshold=0.5,  # More sensitive for gradual changes
            action="stop",
            use_eval_loss=True,
        ),

        # Progress reporting
        ProgressCallback(),
    ]

    args = TrainingArguments(
        output_dir="output_models/my_model",
        save_steps=1000,
        save_total_limit=5,          # Keep 5 recent checkpoints
        preserve_best_model=True,
        preserve_n_best=3,           # Plus preserve 3 best (total: up to 8)
        eval_steps=250,              # Frequent evaluation for divergence detection
        eval_on_save=True,
    )

    # trainer = Trainer(model=model, args=args, callbacks=callbacks, ...)

    # After training:
    # - Best 3 checkpoints preserved (even if older than last 5)
    # - Training stopped early if divergence detected
    # - Callback states saved in each checkpoint for correct resumption
    print("✓ Multiple safeguards configured")


def example_production_training():
    """Production-ready configuration with balanced safeguards."""

    detector = DualTimeScaleDivergenceDetector(
        short_alpha=0.1,
        long_alpha=0.01,
        threshold=2.0,
        use_eval_loss=True,
    )

    args = TrainingArguments(
        output_dir="output_models/my_model",
        preserve_best_model=True,
        preserve_n_best=2,
        save_total_limit=5,
        eval_on_save=True,
        eval_steps=500,
        save_steps=1000,

        # Additional production settings
        logging_steps=100,
        save_safetensors=True,
        save_optimizer_state=True,
        save_scheduler_state=True,
        save_dataset_state=True,
        save_rng_state=True,
    )

    print("✓ Production training configured")


def example_experimentation():
    """Configuration optimized for rapid experimentation."""

    args = TrainingArguments(
        output_dir="output_models/my_model",
        preserve_best_model=True,
        preserve_n_best=5,      # Keep top 5 for comparison
        save_total_limit=10,    # Recent 10 checkpoints
        eval_on_save=True,
        eval_steps=100,         # Frequent eval for quick feedback
        save_steps=500,

        # Fast iteration settings
        logging_steps=50,
        max_eval_steps=100,     # Limit eval time
    )

    print("✓ Experimentation configured")


def example_checkpoint_resume_with_callbacks():
    """Resume training with divergence detection state preserved."""

    # Configure same callbacks as original training
    callbacks = [
        DualTimeScaleDivergenceDetector(
            short_alpha=0.1,
            long_alpha=0.01,
            threshold=2.0,
            use_eval_loss=True,
        ),
    ]

    args = TrainingArguments(
        output_dir="output_models/my_model",
        resume_from_checkpoint=True,  # Auto-finds latest checkpoint
        preserve_best_model=True,
        preserve_n_best=2,
        eval_on_save=True,
    )

    # trainer = Trainer(model=model, args=args, callbacks=callbacks, ...)
    # trainer.train()

    # Callback state (EMA values) automatically restored from checkpoint
    # Training continues with correct divergence detection state
    print("✓ Checkpoint resume with callbacks configured")


def example_different_metrics_for_checkpointing():
    """Use different metrics for different purposes."""

    # Track best by loss, but save frequently based on accuracy
    args = TrainingArguments(
        output_dir="output_models/my_model",

        # Best model by loss
        preserve_best_model=True,
        best_model_metric="loss",
        best_model_greater_is_better=False,
        preserve_n_best=2,

        # Save every 1000 steps
        save_steps=1000,
        save_total_limit=5,
        eval_on_save=True,
    )

    # Also monitor accuracy with divergence detector
    detector = DualTimeScaleDivergenceDetector(
        short_alpha=0.1,
        long_alpha=0.01,
        threshold=0.1,  # Accuracy drop threshold
        use_eval_loss=False,
        metric_key="eval_accuracy",
    )

    print("✓ Different metrics configured")


def example_custom_checkpoint_preservation_logic():
    """Implement custom logic for checkpoint preservation."""
    from forgather.ml.trainer.trainer_types import TrainerCallback

    class TopKCheckpointCallback(TrainerCallback):
        """Keep top K checkpoints by custom metric."""

        def __init__(self, k=3, metric="eval_f1"):
            self.k = k
            self.metric = metric
            self.checkpoints = []  # List of (metric_value, checkpoint_path)

        def on_save(self, args, state, control, **kwargs):
            """Track checkpoints after save."""
            # Get current metric value from state.log_history
            if state.log_history:
                logs = state.log_history[-1]
                metric_value = logs.get(self.metric)

                if metric_value is not None:
                    checkpoint_path = f"{args.output_dir}/checkpoints/checkpoint-{state.global_step}"
                    self.checkpoints.append((metric_value, checkpoint_path))

                    # Sort and keep top K
                    self.checkpoints.sort(reverse=True)  # Descending
                    self.checkpoints = self.checkpoints[:self.k]

                    # Mark best checkpoints for preservation
                    # (would need to integrate with CheckpointManager)
                    print(f"Top {self.k} checkpoints: {[cp[1] for cp in self.checkpoints]}")

            return control

    callback = TopKCheckpointCallback(k=3, metric="eval_f1")
    print("✓ Custom preservation logic configured")


if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Checkpoint Management Examples")
    print("=" * 60)
    print()

    example_multiple_safeguards()
    example_production_training()
    example_experimentation()
    example_checkpoint_resume_with_callbacks()
    example_different_metrics_for_checkpointing()
    example_custom_checkpoint_preservation_logic()

    print()
    print("=" * 60)
    print("All advanced examples configured successfully!")
    print("=" * 60)
