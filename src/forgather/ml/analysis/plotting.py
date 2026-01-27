"""Generate plots from training logs."""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .log_parser import TrainingLog


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

    # Plot each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        for log_idx, log in enumerate(logs):
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

            if x_axis == "step":
                x_values = log.get_steps(records)
                x_label = "Global Step"
            elif x_axis == "epoch":
                x_values = log.get_epochs(records)
                x_label = "Epoch"
            elif x_axis == "time":
                timestamps = log.get_timestamps(records)
                # Convert to relative time in minutes
                x_values = [(t - timestamps[0]) / 60 for t in timestamps]
                x_label = "Time (minutes)"
            else:
                raise ValueError(f"Invalid x_axis: {x_axis}")

            # Apply smoothing if requested
            if smooth_window and smooth_window > 1:
                y_values_smooth = smooth_values(y_values, smooth_window)
                label = f"{log.run_name or f'Run {log_idx+1}'} (smoothed)"
                # Plot both original (faint) and smoothed
                ax.plot(x_values, y_values, alpha=0.2, linewidth=0.5)
                ax.plot(x_values, y_values_smooth, label=label, linewidth=2)
            else:
                label = log.run_name or f"Run {log_idx+1}"
                ax.plot(x_values, y_values, label=label, linewidth=2)

        # Format subplot
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} vs {x_label}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if log_scale:
            ax.set_yscale("log")

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
) -> plt.Figure:
    """Plot loss curves with learning rate on secondary axis.

    Args:
        logs: List of TrainingLog objects to plot
        x_axis: X-axis variable ('step', 'epoch', or 'time')
        smooth_window: Size of smoothing window, or None for no smoothing
        output_path: Path to save figure, or None to not save
        show: Whether to display the plot

    Returns:
        Matplotlib figure object
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Create secondary y-axis for learning rate
    ax2 = ax1.twinx()

    for log_idx, log in enumerate(logs):
        label_prefix = log.run_name or f"Run {log_idx+1}"

        # Plot training loss
        train_records = log.get_training_records()
        if train_records:
            losses = log.get_metric_values("loss", train_records)

            if x_axis == "step":
                x_values = log.get_steps(train_records)
                x_label = "Global Step"
            elif x_axis == "epoch":
                x_values = log.get_epochs(train_records)
                x_label = "Epoch"
            elif x_axis == "time":
                timestamps = log.get_timestamps(train_records)
                x_values = [(t - timestamps[0]) / 60 for t in timestamps]
                x_label = "Time (minutes)"
            else:
                raise ValueError(f"Invalid x_axis: {x_axis}")

            if smooth_window and smooth_window > 1:
                losses_smooth = smooth_values(losses, smooth_window)
                ax1.plot(x_values, losses, alpha=0.2, linewidth=0.5)
                ax1.plot(
                    x_values,
                    losses_smooth,
                    label=f"{label_prefix} Train Loss",
                    linewidth=2,
                )
            else:
                ax1.plot(x_values, losses, label=f"{label_prefix} Train Loss", linewidth=2)

        # Plot eval loss
        eval_records = log.get_eval_records()
        if eval_records:
            eval_losses = log.get_metric_values("eval_loss", eval_records)

            if x_axis == "step":
                eval_x = log.get_steps(eval_records)
            elif x_axis == "epoch":
                eval_x = log.get_epochs(eval_records)
            elif x_axis == "time":
                eval_timestamps = log.get_timestamps(eval_records)
                eval_x = [(t - timestamps[0]) / 60 for t in eval_timestamps]

            ax1.plot(
                eval_x,
                eval_losses,
                label=f"{label_prefix} Eval Loss",
                marker="o",
                linewidth=2,
                markersize=6,
            )

        # Plot learning rate on secondary axis
        if train_records:
            learning_rates = log.get_metric_values("learning_rate", train_records)
            if learning_rates:
                ax2.plot(
                    x_values,
                    learning_rates,
                    label=f"{label_prefix} LR",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                )

    # Format axes
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.set_ylabel("Learning Rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.legend(loc="upper right")

    plt.title("Training Progress")
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
