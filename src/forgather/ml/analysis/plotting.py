"""Generate plots from training logs."""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .log_parser import TrainingLog

# Color palette optimized for distinguishing multiple runs
_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def _get_color(index: int) -> str:
    """Get color for run index, cycling through palette."""
    return _COLORS[index % len(_COLORS)]


def _get_x_values(log, records, x_axis):
    """Extract x-axis values and label from log records.

    Returns:
        (x_values, x_label)
    """
    if x_axis == "step":
        return log.get_steps(records), "Global Step"
    elif x_axis == "epoch":
        return log.get_epochs(records), "Epoch"
    elif x_axis == "time":
        timestamps = log.get_timestamps(records)
        x_values = [(t - timestamps[0]) / 60 for t in timestamps]
        return x_values, "Time (minutes)"
    else:
        raise ValueError(f"Invalid x_axis: {x_axis}")


def smooth_values(values: List[float], window_size: int = 10) -> List[float]:
    """Apply moving average smoothing to values.

    Args:
        values: List of values to smooth
        window_size: Size of smoothing window

    Returns:
        Smoothed values
    """
    if window_size <= 1 or len(values) < window_size:
        return values

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))

    return smoothed


def plot_training_metrics(
    logs: List[TrainingLog],
    metrics: Optional[List[str]] = None,
    x_axis: str = "step",
    smooth_window: Optional[int] = None,
    log_scale: bool = False,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    show: bool = False,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot training metrics from one or more logs.

    Args:
        logs: List of TrainingLog objects to plot
        metrics: List of metrics to plot (e.g., ['loss', 'eval_loss', 'learning_rate'])
                 If None, plots loss, eval_loss, and learning_rate
        x_axis: X-axis variable ('step', 'epoch', or 'time')
        smooth_window: Size of smoothing window, or None for no smoothing
        log_scale: Use log scale for y-axis
        output_path: Path to save figure, or None to not save
        figsize: Figure size as (width, height)
        show: Whether to display the plot
        title: Optional custom plot title

    Returns:
        Matplotlib figure object
    """
    if metrics is None:
        metrics = ["loss", "eval_loss", "learning_rate"]

    # Determine subplot layout
    n_metrics = len(metrics)
    n_rows = (n_metrics + 1) // 2 if n_metrics > 1 else 1
    n_cols = 2 if n_metrics > 1 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    x_label = "Global Step"

    # Plot each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        for log_idx, log in enumerate(logs):
            color = _get_color(log_idx)
            label = log.get_label(log_idx)

            # Get appropriate records for this metric
            if metric == "eval_loss":
                records = log.get_eval_records()
            elif metric in ["loss", "grad_norm", "learning_rate", "max_grad_norm"]:
                records = log.get_training_records()
            else:
                # Try to find in any records
                records = [r for r in log.records if metric in r]

            if not records:
                continue

            # Get x and y values
            y_values = log.get_metric_values(metric, records)
            x_values, x_label = _get_x_values(log, records, x_axis)

            # Apply smoothing if requested
            if smooth_window and smooth_window > 1:
                y_values_smooth = smooth_values(y_values, smooth_window)
                # Plot both original (faint) and smoothed
                ax.plot(x_values, y_values, alpha=0.15, linewidth=0.5, color=color)
                ax.plot(
                    x_values, y_values_smooth,
                    label=label, linewidth=2, color=color,
                )
            else:
                ax.plot(x_values, y_values, label=label, linewidth=2, color=color)

        # Format subplot
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} vs {x_label}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if log_scale:
            ax.set_yscale("log")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_loss_curves(
    logs: List[TrainingLog],
    x_axis: str = "step",
    smooth_window: Optional[int] = None,
    output_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot loss curves with learning rate on secondary axis.

    When comparing multiple runs, train and eval loss are split into
    separate subplots for readability. Single-run plots use a combined view.

    Args:
        logs: List of TrainingLog objects to plot
        x_axis: X-axis variable ('step', 'epoch', or 'time')
        smooth_window: Size of smoothing window, or None for no smoothing
        output_path: Path to save figure, or None to not save
        show: Whether to display the plot
        title: Optional custom plot title

    Returns:
        Matplotlib figure object
    """
    multi_run = len(logs) > 1

    if multi_run:
        fig = _plot_loss_curves_multi(logs, x_axis, smooth_window, title)
    else:
        fig = _plot_loss_curves_single(logs, x_axis, smooth_window, title)

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # Show if requested
    if show:
        plt.show()

    return fig


def _plot_loss_curves_single(logs, x_axis, smooth_window, title):
    """Single-run loss curves with dual y-axes (loss + LR)."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x_label = "Global Step"

    for log_idx, log in enumerate(logs):
        label_prefix = log.get_label(log_idx)

        # Plot training loss
        train_records = log.get_training_records()
        if train_records:
            losses = log.get_metric_values("loss", train_records)
            x_values, x_label = _get_x_values(log, train_records, x_axis)

            if smooth_window and smooth_window > 1:
                losses_smooth = smooth_values(losses, smooth_window)
                ax1.plot(x_values, losses, alpha=0.2, linewidth=0.5, color="tab:blue")
                ax1.plot(
                    x_values, losses_smooth,
                    label=f"{label_prefix} Train Loss", linewidth=2, color="tab:blue",
                )
            else:
                ax1.plot(
                    x_values, losses,
                    label=f"{label_prefix} Train Loss", linewidth=2, color="tab:blue",
                )

        # Plot eval loss
        eval_records = log.get_eval_records()
        if eval_records:
            eval_losses = log.get_metric_values("eval_loss", eval_records)
            eval_x, _ = _get_x_values(log, eval_records, x_axis)
            ax1.plot(
                eval_x, eval_losses,
                label=f"{label_prefix} Eval Loss",
                marker="o", linewidth=2, markersize=6, color="tab:green",
            )

        # Plot learning rate on secondary axis
        if train_records:
            learning_rates = log.get_metric_values("learning_rate", train_records)
            if learning_rates:
                ax2.plot(
                    x_values, learning_rates,
                    label=f"{label_prefix} LR",
                    linestyle="--", alpha=0.7, linewidth=1.5, color="tab:orange",
                )

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.set_ylabel("Learning Rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.legend(loc="upper right")

    plt.title(title or "Training Progress")
    plt.tight_layout()
    return fig


def _plot_loss_curves_multi(logs, x_axis, smooth_window, title):
    """Multi-run comparison: train loss and eval loss in separate subplots."""
    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(16, 6))

    x_label = "Global Step"
    has_eval = False

    for log_idx, log in enumerate(logs):
        color = _get_color(log_idx)
        label = log.get_label(log_idx)

        # Plot training loss
        train_records = log.get_training_records()
        if train_records:
            losses = log.get_metric_values("loss", train_records)
            x_values, x_label = _get_x_values(log, train_records, x_axis)

            if smooth_window and smooth_window > 1:
                losses_smooth = smooth_values(losses, smooth_window)
                ax_train.plot(
                    x_values, losses, alpha=0.15, linewidth=0.5, color=color,
                )
                ax_train.plot(
                    x_values, losses_smooth,
                    label=label, linewidth=2, color=color,
                )
            else:
                ax_train.plot(
                    x_values, losses, label=label, linewidth=2, color=color,
                )

        # Plot eval loss
        eval_records = log.get_eval_records()
        if eval_records:
            has_eval = True
            eval_losses = log.get_metric_values("eval_loss", eval_records)
            eval_x, _ = _get_x_values(log, eval_records, x_axis)
            ax_eval.plot(
                eval_x, eval_losses,
                label=label, marker="o", linewidth=2, markersize=4, color=color,
            )

    # Format train loss subplot
    ax_train.set_xlabel(x_label)
    ax_train.set_ylabel("Loss")
    ax_train.set_title("Train Loss")
    ax_train.grid(True, alpha=0.3)
    ax_train.legend()

    # Format eval loss subplot
    ax_eval.set_xlabel(x_label)
    ax_eval.set_ylabel("Eval Loss")
    ax_eval.set_title("Eval Loss")
    ax_eval.grid(True, alpha=0.3)
    if has_eval:
        ax_eval.legend()

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
