"""Compute summary statistics from training logs."""

import math
from typing import Any, Dict, Optional

from .log_parser import TrainingLog


def get_perplexity(value, metrics=None):
    """Compute perplexity as ``exp(value)``.

    Returns ``float('inf')`` on overflow rather than raising an exception.
    The *metrics* parameter is accepted for compatibility with TBLogger
    transform callback signatures but is not used.

    Parameters
    ----------
    value : float
        A loss value (nats) to convert to perplexity.
    metrics : any, optional
        Ignored.  Present only for API compatibility.

    Returns
    -------
    float
        ``math.exp(value)``, or ``float('inf')`` if *value* is too large.
    """
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


def get_bpb(loss_nats: float, tokens_per_byte: float) -> float:
    """Convert per-token cross-entropy (in nats) to bits-per-byte.

    ``bpb = loss_nats * tokens_per_byte / ln(2)``. The result is the average
    number of bits the model needs to encode one byte of UTF-8 source text,
    which is independent of the tokenizer's vocabulary size and therefore
    comparable across models that use different tokenizers.

    Parameters
    ----------
    loss_nats : float
        Mean cross-entropy loss per predicted token, in nats.
    tokens_per_byte : float
        Predicted tokens divided by source bytes over the same corpus prefix.
        Must be strictly positive; otherwise ``float('nan')`` is returned.

    Returns
    -------
    float
        Bits-per-byte.
    """
    if not (tokens_per_byte > 0) or not math.isfinite(loss_nats):
        return float("nan")
    return loss_nats * tokens_per_byte / math.log(2)


def get_bpc(loss_nats: float, tokens_per_char: float) -> float:
    """Convert per-token cross-entropy (in nats) to bits-per-character.

    Same idea as :func:`get_bpb` but normalised by Unicode code points
    instead of UTF-8 bytes.
    """
    if not (tokens_per_char > 0) or not math.isfinite(loss_nats):
        return float("nan")
    return loss_nats * tokens_per_char / math.log(2)


def compute_summary_statistics(log: TrainingLog) -> Dict[str, Any]:
    """Compute summary statistics from a training log.

    Aggregates training-step records, evaluation records, and the final
    summary record into a flat dictionary of key metrics.  Keys are only
    present when the underlying data exists; callers should use
    ``summary.get(key)`` rather than direct indexing.

    Parameters
    ----------
    log : TrainingLog
        Parsed training log to summarise.

    Returns
    -------
    dict
        Dictionary with a subset of the following keys, depending on what
        data is available in *log*:

        ``run_name`` : str or None
            Name of the training run.
        ``log_path`` : str
            String representation of the log file path.
        ``total_steps`` : int
            Global step number of the last training record.
        ``final_epoch`` : float
            Epoch number at the last training step.
        ``final_loss`` : float
            Training loss at the last recorded step.
        ``avg_loss`` : float
            Mean training loss over all recorded steps.
        ``min_loss`` : float
            Minimum training loss observed during the run.
        ``best_loss`` : float
            Training loss at the step where it was lowest (same as
            ``min_loss`` but paired with ``best_loss_step``).
        ``best_loss_step`` : int
            Global step at which ``best_loss`` was achieved.
        ``avg_grad_norm`` : float
            Mean gradient norm over all training steps that recorded it.
        ``max_grad_norm_value`` : float
            Peak gradient norm observed during training.
        ``max_grad_norm_step`` : int
            Global step at which ``max_grad_norm_value`` was observed.
        ``initial_lr`` : float
            Learning rate at the first training step.
        ``final_lr`` : float
            Learning rate at the last training step.
        ``final_eval_loss`` : float
            Evaluation loss from the most recent evaluation checkpoint.
        ``best_eval_loss`` : float
            Lowest evaluation loss observed.
        ``best_eval_loss_step`` : int
            Global step at which ``best_eval_loss`` was achieved.
        ``train_runtime`` : float
            Total training wall-clock time in seconds.
        ``train_samples`` : int
            Total number of training samples processed.
        ``train_samples_per_second`` : float
            Average throughput in samples per second.
        ``train_steps_per_second`` : float
            Average throughput in optimizer steps per second.
        ``effective_batch_size`` : int
            Effective batch size (local batch x gradient accumulation x
            world size).

    Examples
    --------
    >>> from forgather.ml.analysis import TrainingLog, compute_summary_statistics
    >>> log = TrainingLog.from_file("output_models/my_model/runs/run_001/trainer_logs.json")
    >>> summary = compute_summary_statistics(log)
    >>> print(f"Best loss: {summary['best_loss']:.4f} at step {summary['best_loss_step']}")
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
        summary["train_samples_per_second"] = final_record.get(
            "train_samples_per_second"
        )
        summary["train_steps_per_second"] = final_record.get("train_steps_per_second")
        summary["effective_batch_size"] = final_record.get("effective_batch_size")

    return summary


def format_summary_text(summary: Dict[str, Any]) -> str:
    """Format summary statistics as a human-readable plain-text block.

    Parameters
    ----------
    summary : dict
        Dictionary returned by :func:`compute_summary_statistics`.

    Returns
    -------
    str
        Multi-line plain-text string with sections for training progress,
        metrics, training speed, gradient statistics, and learning rate.
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
    """Format summary statistics as a Markdown document.

    Parameters
    ----------
    summary : dict
        Dictionary returned by :func:`compute_summary_statistics`.

    Returns
    -------
    str
        Markdown-formatted string with a header, metrics table, and
        training-speed section suitable for rendering in notebooks or
        documentation.
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
            lines.append(
                f"| Final Loss | {summary['final_loss']:.4f} | {summary.get('total_steps', 'N/A')} |"
            )

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
            lines.append(
                f"- **Samples/sec:** {summary['train_samples_per_second']:.2f}"
            )

        if summary.get("train_steps_per_second"):
            lines.append(f"- **Steps/sec:** {summary['train_steps_per_second']:.2f}")

        if summary.get("effective_batch_size"):
            lines.append(
                f"- **Effective Batch Size:** {summary['effective_batch_size']}"
            )

        lines.append("")

    return "\n".join(lines)


def format_summary_oneline(summary: Dict[str, Any]) -> str:
    """Format summary statistics as a compact single-line string.

    Useful for tabular displays when comparing many runs side-by-side (e.g.
    ``forgather logs summary --all --format one-line``).

    Parameters
    ----------
    summary : dict
        Dictionary returned by :func:`compute_summary_statistics`.

    Returns
    -------
    str
        Pipe-delimited one-liner containing run name, step count, wall-clock
        duration, final training loss, best eval loss, and throughput (where
        available).
    """
    # Extract key metrics
    run_name = summary.get("run_name", "Unknown")[:30]
    steps = summary.get("total_steps", 0)
    duration = summary.get("train_runtime", 0)
    final_loss = summary.get("final_loss")
    best_eval = summary.get("best_eval_loss")
    samples_sec = summary.get("train_samples_per_second")

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
