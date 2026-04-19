"""
Checkpoint Preservation Examples

Demonstrates how to prevent best checkpoints from being deleted during training.
"""

from forgather.ml.trainer import Trainer, TrainingArguments


def example_basic_preservation():
    """Basic checkpoint preservation - keep best checkpoint safe."""
    args = TrainingArguments(
        output_dir="output_models/my_model",

        # Checkpoint management
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,           # Keep only 3 recent checkpoints

        # Preserve best model (new!)
        preserve_best_model=True,     # Don't delete best checkpoint
        best_model_metric="loss",
        preserve_n_best=1,            # Keep top 1 checkpoint

        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        eval_on_save=True,            # Force eval at save steps (new!)
    )

    # trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
    # trainer.train()

    # After training:
    # - Best checkpoint preserved (not deleted by save_total_limit)
    # - Evaluation automatically runs before saves (eval_on_save=True)
    print("✓ Basic preservation configured")


def example_n_best_checkpoints():
    """Track top N checkpoints for ensembling or comparison."""
    args = TrainingArguments(
        output_dir="output_models/my_model",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,          # Keep 5 recent checkpoints

        preserve_best_model=True,
        preserve_n_best=3,           # Plus preserve 3 best (total: up to 8)
        best_model_metric="eval_accuracy",
        best_model_greater_is_better=True,  # Higher accuracy is better

        eval_strategy="steps",
        eval_steps=250,              # Frequent evaluation
        eval_on_save=True,
    )

    # After training:
    # - Top 3 checkpoints preserved (by eval_accuracy)
    # - Last 5 checkpoints preserved (by recency)
    # - Total: up to 8 checkpoints (if best are not in recent 5)
    print("✓ N best checkpoints configured")


def example_decoupled_eval_save():
    """Different eval and save frequencies without alignment requirement."""
    args = TrainingArguments(
        output_dir="output_models/my_model",

        # Different frequencies (no longer requires save_steps % eval_steps == 0)
        save_steps=1000,             # Save every 1000 steps
        eval_steps=250,              # Eval every 250 steps
        eval_on_save=True,           # Force eval when saving (new!)

        preserve_best_model=True,
        best_model_metric="loss",
    )

    # Eval schedule: steps 250, 500, 750, 1000, 1250, ...
    # Save schedule: steps 1000, 2000, 3000, ...
    # At step 1000: eval forced (eval_on_save=True), then save with metrics
    print("✓ Decoupled eval/save configured")


def example_backward_compatibility():
    """Old load_best_model_at_end still works with deprecation warning."""
    args = TrainingArguments(
        output_dir="output_models/my_model",

        # Old API (deprecated but still works)
        load_best_model_at_end=True,
        metric_for_best_model="loss",

        save_steps=1000,
        eval_steps=1000,  # No longer requires strict alignment with eval_on_save
    )

    # Prints deprecation warning and auto-migrates to:
    # - preserve_best_model=True
    # - best_model_metric="loss"
    # - eval_on_save=True
    print("✓ Backward compatibility configured")


if __name__ == "__main__":
    print("=" * 60)
    print("Checkpoint Preservation Examples")
    print("=" * 60)
    print()

    example_basic_preservation()
    example_n_best_checkpoints()
    example_decoupled_eval_save()
    example_backward_compatibility()

    print()
    print("=" * 60)
    print("All examples configured successfully!")
    print("=" * 60)
