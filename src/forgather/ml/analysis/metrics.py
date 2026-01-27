"""Compute summary statistics from training logs."""

from typing import Any, Dict, Optional

from .log_parser import TrainingLog


def compute_summary_statistics(log: TrainingLog) -> Dict[str, Any]:
    """Compute summary statistics from a training log.

    Args:
        log: TrainingLog object

    Returns:
        Dictionary containing summary statistics
    """
    train_records = log.get_training_records()
    eval_records = log.get_eval_records()
    final_record = log.get_final_record()

    summary = {
        "run_name": log.run_name,
        "log_path": str(log.log_path),
    }

    # Training progress
    if train_records:
        summary["total_steps"] = train_records[-1].get("global_step", 0)
        summary["final_epoch"] = train_records[-1].get("epoch", 0)

    # Training metrics
    if train_records:
        losses = log.get_metric_values("loss", train_records)
        if losses:
            summary["final_loss"] = losses[-1]
            summary["avg_loss"] = sum(losses) / len(losses)
            summary["min_loss"] = min(losses)
            best_loss_step, best_loss = log.find_best_step("loss", mode="min")
            summary["best_loss"] = best_loss
            summary["best_loss_step"] = best_loss_step

        # Gradient statistics
        grad_norms = log.get_metric_values("grad_norm", train_records)
        if grad_norms:
            summary["avg_grad_norm"] = sum(grad_norms) / len(grad_norms)
            summary["max_grad_norm_value"] = max(grad_norms)
            max_idx = grad_norms.index(max(grad_norms))
            summary["max_grad_norm_step"] = train_records[max_idx]["global_step"]

        # Learning rate
        learning_rates = log.get_metric_values("learning_rate", train_records)
        if learning_rates:
            summary["initial_lr"] = learning_rates[0]
            summary["final_lr"] = learning_rates[-1]

    # Evaluation metrics
    if eval_records:
        eval_losses = log.get_metric_values("eval_loss", eval_records)
        if eval_losses:
            summary["final_eval_loss"] = eval_losses[-1]
            best_eval_step, best_eval_loss = log.find_best_step("eval_loss", mode="min")
            summary["best_eval_loss"] = best_eval_loss
            summary["best_eval_loss_step"] = best_eval_step

    # Training performance
    if final_record:
        summary["train_runtime"] = final_record.get("train_runtime")
        summary["train_samples"] = final_record.get("train_samples")
        summary["train_samples_per_second"] = final_record.get("train_samples_per_second")
        summary["train_steps_per_second"] = final_record.get("train_steps_per_second")
        summary["effective_batch_size"] = final_record.get("effective_batch_size")

    return summary


def format_summary_text(summary: Dict[str, Any]) -> str:
    """Format summary statistics as human-readable text.

    Args:
        summary: Summary statistics dictionary

    Returns:
        Formatted text string
    """
    lines = []
    lines.append("Training Run Summary")
    lines.append("=" * 60)
    lines.append(f"Run: {summary.get('run_name', 'Unknown')}")

    if summary.get("train_runtime"):
        lines.append(f"Duration: {summary['train_runtime']:.2f}s")

    if summary.get("total_steps"):
        lines.append(f"Total Steps: {summary['total_steps']}")

    if summary.get("final_epoch") is not None:
        lines.append(f"Final Epoch: {summary['final_epoch']:.4f}")

    lines.append("")

    # Training metrics
    if any(k.endswith("_loss") for k in summary):
        lines.append("Metrics:")

        if summary.get("final_loss") is not None:
            lines.append(f"  Final Loss: {summary['final_loss']:.4f}")

        if summary.get("best_loss") is not None:
            lines.append(
                f"  Best Loss: {summary['best_loss']:.4f} "
                f"(step {summary.get('best_loss_step', 'N/A')})"
            )

        if summary.get("avg_loss") is not None:
            lines.append(f"  Average Loss: {summary['avg_loss']:.4f}")

        if summary.get("final_eval_loss") is not None:
            lines.append(
                f"  Final Eval Loss: {summary['final_eval_loss']:.4f} "
                f"(step {summary.get('best_eval_loss_step', 'N/A')})"
            )

        if summary.get("best_eval_loss") is not None:
            lines.append(
                f"  Best Eval Loss: {summary['best_eval_loss']:.4f} "
                f"(step {summary.get('best_eval_loss_step', 'N/A')})"
            )

        lines.append("")

    # Training speed
    if any(k.startswith("train_") for k in summary):
        lines.append("Training Speed:")

        if summary.get("train_samples_per_second"):
            lines.append(f"  Samples/sec: {summary['train_samples_per_second']:.2f}")

        if summary.get("train_steps_per_second"):
            lines.append(f"  Steps/sec: {summary['train_steps_per_second']:.2f}")

        if summary.get("effective_batch_size"):
            lines.append(f"  Effective Batch Size: {summary['effective_batch_size']}")

        lines.append("")

    # Gradient statistics
    if summary.get("avg_grad_norm") is not None:
        lines.append("Gradient Statistics:")
        lines.append(f"  Average Grad Norm: {summary['avg_grad_norm']:.4f}")

        if summary.get("max_grad_norm_value"):
            lines.append(
                f"  Max Grad Norm: {summary['max_grad_norm_value']:.4f} "
                f"(step {summary.get('max_grad_norm_step', 'N/A')})"
            )

        lines.append("")

    # Learning rate
    if summary.get("initial_lr") is not None:
        lines.append("Learning Rate:")
        lines.append(f"  Initial: {summary['initial_lr']:.6f}")

        if summary.get("final_lr") is not None:
            lines.append(f"  Final: {summary['final_lr']:.6f}")

        lines.append("")

    return "\n".join(lines)


def format_summary_markdown(summary: Dict[str, Any]) -> str:
    """Format summary statistics as markdown.

    Args:
        summary: Summary statistics dictionary

    Returns:
        Markdown formatted string
    """
    lines = []
    lines.append("# Training Run Summary")
    lines.append("")
    lines.append(f"**Run:** {summary.get('run_name', 'Unknown')}")

    if summary.get("train_runtime"):
        lines.append(f"**Duration:** {summary['train_runtime']:.2f}s")

    if summary.get("total_steps"):
        lines.append(f"**Total Steps:** {summary['total_steps']}")

    if summary.get("final_epoch") is not None:
        lines.append(f"**Final Epoch:** {summary['final_epoch']:.4f}")

    lines.append("")

    # Training metrics table
    if any(k.endswith("_loss") for k in summary):
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Value | Step |")
        lines.append("|--------|-------|------|")

        if summary.get("final_loss") is not None:
            lines.append(f"| Final Loss | {summary['final_loss']:.4f} | {summary.get('total_steps', 'N/A')} |")

        if summary.get("best_loss") is not None:
            lines.append(
                f"| Best Loss | {summary['best_loss']:.4f} | "
                f"{summary.get('best_loss_step', 'N/A')} |"
            )

        if summary.get("final_eval_loss") is not None:
            lines.append(
                f"| Final Eval Loss | {summary['final_eval_loss']:.4f} | "
                f"{summary.get('best_eval_loss_step', 'N/A')} |"
            )

        if summary.get("best_eval_loss") is not None:
            lines.append(
                f"| Best Eval Loss | {summary['best_eval_loss']:.4f} | "
                f"{summary.get('best_eval_loss_step', 'N/A')} |"
            )

        lines.append("")

    # Training speed
    if any(k.startswith("train_") for k in summary):
        lines.append("## Training Speed")
        lines.append("")

        if summary.get("train_samples_per_second"):
            lines.append(f"- **Samples/sec:** {summary['train_samples_per_second']:.2f}")

        if summary.get("train_steps_per_second"):
            lines.append(f"- **Steps/sec:** {summary['train_steps_per_second']:.2f}")

        if summary.get("effective_batch_size"):
            lines.append(f"- **Effective Batch Size:** {summary['effective_batch_size']}")

        lines.append("")

    return "\n".join(lines)


def format_summary_oneline(summary: Dict[str, Any]) -> str:
    """Format summary statistics as a single line.

    Args:
        summary: Summary statistics dictionary

    Returns:
        Single line formatted string with key metrics
    """
    # Extract key metrics
    run_name = summary.get('run_name', 'Unknown')[:30]
    steps = summary.get('total_steps', 0)
    duration = summary.get('train_runtime', 0)
    final_loss = summary.get('final_loss')
    best_eval = summary.get('best_eval_loss')
    samples_sec = summary.get('train_samples_per_second')

    # Format duration as MM:SS
    if duration:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        duration_str = f"{minutes:02d}:{seconds:02d}"
    else:
        duration_str = "N/A"

    # Build the one-line summary
    parts = [
        f"{run_name:<32}",
        f"steps={steps:<5}",
        f"time={duration_str:<6}",
    ]

    if final_loss is not None:
        parts.append(f"loss={final_loss:.4f}")

    if best_eval is not None:
        parts.append(f"eval={best_eval:.4f}")

    if samples_sec is not None:
        parts.append(f"samp/s={samples_sec:.1f}")

    return " | ".join(parts)
