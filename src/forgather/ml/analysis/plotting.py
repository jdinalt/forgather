"""Generate plots from training logs."""

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

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

# Metric names (lowercased) that get outlier-aware y-scaling by default.
_LOSS_LIKE_METRICS = {
    "loss",
    "train_loss",
    "eval_loss",
    "grad_norm",
    "max_grad_norm",
}

# Metrics that should be converted to perplexity when --perplexity is set.
_PERPLEXITY_METRICS = {"loss", "train_loss", "eval_loss"}


def _get_color(index: int) -> str:
    """Get color for run index, cycling through palette."""
    return _COLORS[index % len(_COLORS)]


def _get_x_values(log, records, x_axis):
    """Extract x-axis values and label from log records."""
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
    """Apply a centred moving-average to a sequence of values.

    When *values* has fewer elements than *window_size*, or *window_size* is
    1 or less, the original list is returned unchanged.

    Parameters
    ----------
    values : list of float
        Raw metric values to smooth.
    window_size : int, optional
        Number of points to average over.  Default is 10.

    Returns
    -------
    list of float
        Smoothed values of the same length as *values*.
    """
    if window_size <= 1 or len(values) < window_size:
        return values

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))

    return smoothed


def _is_perplexity_metric(metric: str) -> bool:
    return metric.lower() in _PERPLEXITY_METRICS


def _is_loss_like_metric(metric: str) -> bool:
    return metric.lower() in _LOSS_LIKE_METRICS


def _apply_perplexity(values: Sequence[float]) -> List[float]:
    """Map loss values to exp(loss). NaN/inf propagate through."""
    out = []
    for v in values:
        try:
            out.append(math.exp(v))
        except (OverflowError, ValueError):
            out.append(float("inf"))
    return out


def _metric_display_label(metric: str, perplexity: bool) -> str:
    """Return a nicely-cased label for axis titles."""
    if perplexity and _is_perplexity_metric(metric):
        if metric.lower() == "eval_loss":
            return "Eval Perplexity"
        if metric.lower() == "train_loss":
            return "Train Perplexity"
        return "Perplexity"
    return metric.replace("_", " ").title()


def _clip_to_x_window(
    x_values: Sequence[float],
    y_values: Sequence[float],
    x_min: Optional[float],
    x_max: Optional[float],
) -> Tuple[List[float], List[float]]:
    """Trim (x, y) pairs to the [x_min, x_max] domain window."""
    if x_min is None and x_max is None:
        return list(x_values), list(y_values)
    xs, ys = [], []
    for x, y in zip(x_values, y_values):
        if x_min is not None and x < x_min:
            continue
        if x_max is not None and x > x_max:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def _percentile_ylim(
    series: Sequence[Sequence[float]],
    log_scale: bool,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> Optional[Tuple[float, float]]:
    """Compute an outlier-aware (y_lo, y_hi) from one or more plotted series.

    Combines series by taking the min of per-series lo and max of per-series hi,
    so every series stays visible but single-series outliers don't dominate.
    Returns None if there is not enough data to compute a window.
    """
    lows, highs = [], []
    for values in series:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if log_scale:
            arr = arr[arr > 0]
        if arr.size < 2:
            continue
        p_lo = float(np.nanpercentile(arr, lo_pct))
        p_hi = float(np.nanpercentile(arr, hi_pct))
        if not (math.isfinite(p_lo) and math.isfinite(p_hi)):
            continue
        if p_hi <= p_lo:
            # Degenerate window; widen a bit so matplotlib doesn't complain.
            spread = max(abs(p_hi), 1e-9) * 0.1
            p_lo -= spread
            p_hi += spread
        span = p_hi - p_lo
        lows.append(p_lo - 0.05 * span)
        highs.append(p_hi + 0.15 * span)
    if not lows or not highs:
        return None
    lo = min(lows)
    hi = max(highs)
    if log_scale:
        # Percentiles were computed on positive-only data; guarantee lo > 0.
        lo = max(lo, 1e-12)
        if hi <= lo:
            hi = lo * 10
    return lo, hi


def _apply_ylim(
    ax,
    series: Sequence[Sequence[float]],
    log_scale: bool,
    ignore_outliers: bool,
    y_min: Optional[float],
    y_max: Optional[float],
):
    """Set ax y-limits per user options, combining outlier-aware auto-scale with
    explicit y_min/y_max overrides.

    On a log y-axis, outlier-aware percentile clipping is suppressed: log-scale
    plots exist precisely to handle data spanning orders of magnitude, so
    clipping to a p5/p95 window works against that intent. Explicit y_min/y_max
    overrides are still honoured.
    """
    # Detect log axis via both the explicit flag and the matplotlib axis state,
    # in case some caller sets ax.set_yscale("log") without passing log_scale=True.
    axis_is_log = log_scale or ax.get_yscale() == "log"

    auto_lo, auto_hi = None, None
    if ignore_outliers and not axis_is_log:
        window = _percentile_ylim(series, log_scale)
        if window is not None:
            auto_lo, auto_hi = window

    lo = y_min if y_min is not None else auto_lo
    hi = y_max if y_max is not None else auto_hi
    if lo is not None or hi is not None:
        ax.set_ylim(bottom=lo, top=hi)


def plot_training_metrics(
    logs: List[TrainingLog],
    metrics: Optional[List[str]] = None,
    x_axis: str = "step",
    smooth_window: Optional[int] = None,
    log_scale: bool = False,
    output_path: Union[str, Path, None] = None,
    figsize: Tuple[int, int] = (12, 8),
    show: bool = False,
    title: Optional[str] = None,
    ignore_outliers: bool = True,
    perplexity: bool = False,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Figure:
    """Plot one or more training metrics from one or more training logs.

    Creates a grid of subplots (up to two columns) with one panel per metric.
    When multiple logs are supplied each run is drawn in a distinct colour with
    a legend entry.  For loss-like metrics (``loss``, ``eval_loss``,
    ``grad_norm``) the y-axis is automatically clipped to the 5th–95th
    percentile window to suppress early-training outliers; pass
    ``ignore_outliers=False`` to disable this.

    Parameters
    ----------
    logs : list of TrainingLog
        One or more parsed training logs to plot.
    metrics : list of str, optional
        Metric keys to plot.  Each element must be a key present in at least
        some log records (e.g. ``'loss'``, ``'eval_loss'``,
        ``'learning_rate'``, ``'grad_norm'``).  Default is
        ``['loss', 'eval_loss', 'learning_rate']``.
    x_axis : {'step', 'epoch', 'time'}, optional
        X-axis variable.  ``'step'`` uses ``global_step``, ``'epoch'`` uses
        ``epoch``, and ``'time'`` converts timestamps to elapsed minutes.
        Default is ``'step'``.
    smooth_window : int, optional
        When greater than 1, draws the raw series at low opacity and overlays
        a centred moving-average with the given window size.  Default is
        ``None`` (no smoothing).
    log_scale : bool, optional
        Use a logarithmic y-axis.  Outlier-aware auto-scaling is suppressed
        on log axes.  Default is ``False``.
    output_path : str or Path, optional
        If provided, the figure is saved to this path at 300 dpi.  Parent
        directories are created automatically.
    figsize : tuple of int, optional
        ``(width, height)`` in inches passed to ``plt.subplots``.  Default is
        ``(12, 8)``.
    show : bool, optional
        Call ``plt.show()`` after rendering.  Default is ``False``.
    title : str, optional
        Figure-level suptitle.  When ``None`` no title is added.
    ignore_outliers : bool, optional
        Apply percentile-based y-axis clipping for loss-like metrics.
        Default is ``True``.
    perplexity : bool, optional
        Convert loss values to perplexity (``exp(loss)``) for ``loss``,
        ``train_loss``, and ``eval_loss`` metrics.  Default is ``False``.
    x_min : float, optional
        Clip data and set the left x-axis limit to this value.
    x_max : float, optional
        Clip data and set the right x-axis limit to this value.
    y_min : float, optional
        Override the bottom y-axis limit.  Takes priority over auto-scaling.
    y_max : float, optional
        Override the top y-axis limit.  Takes priority over auto-scaling.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.  The caller is responsible for closing it when
        no longer needed (``plt.close(fig)``).

    Examples
    --------
    >>> from forgather.ml.analysis import TrainingLog
    >>> from forgather.ml.analysis.plotting import plot_training_metrics
    >>> log = TrainingLog.from_file("output_models/my_model/runs/run_001/trainer_logs.json")
    >>> fig = plot_training_metrics([log], metrics=["loss", "eval_loss"], smooth_window=20)
    >>> fig.savefig("training.png", dpi=150)
    """
    if metrics is None:
        metrics = ["loss", "eval_loss", "learning_rate"]

    n_metrics = len(metrics)
    n_rows = (n_metrics + 1) // 2 if n_metrics > 1 else 1
    n_cols = 2 if n_metrics > 1 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    x_label = "Global Step"

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        plotted_series: List[List[float]] = []
        metric_is_perplexity = perplexity and _is_perplexity_metric(metric)

        for log_idx, log in enumerate(logs):
            color = _get_color(log_idx)
            label = log.get_label(log_idx)

            if metric == "eval_loss":
                records = log.get_eval_records()
            elif metric in ["loss", "grad_norm", "learning_rate", "max_grad_norm"]:
                records = log.get_training_records()
            else:
                records = [r for r in log.records if metric in r]

            if not records:
                continue

            y_values = log.get_metric_values(metric, records)
            x_values, x_label = _get_x_values(log, records, x_axis)

            if metric_is_perplexity:
                y_values = _apply_perplexity(y_values)

            x_values, y_values = _clip_to_x_window(x_values, y_values, x_min, x_max)
            if not x_values:
                continue

            if smooth_window and smooth_window > 1:
                y_values_smooth = smooth_values(y_values, smooth_window)
                ax.plot(x_values, y_values, alpha=0.15, linewidth=0.5, color=color)
                ax.plot(
                    x_values,
                    y_values_smooth,
                    label=label,
                    linewidth=2,
                    color=color,
                )
                plotted_series.append(list(y_values_smooth))
            else:
                ax.plot(x_values, y_values, label=label, linewidth=2, color=color)
                plotted_series.append(list(y_values))

        ax.set_xlabel(x_label)
        ax.set_ylabel(_metric_display_label(metric, perplexity))
        ax.set_title(f"{_metric_display_label(metric, perplexity)} vs {x_label}")
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

        if log_scale:
            ax.set_yscale("log")

        if x_min is not None or x_max is not None:
            ax.set_xlim(left=x_min, right=x_max)

        # Only auto-clip y for loss-like metrics; don't squash LR.
        if plotted_series and (_is_loss_like_metric(metric) or metric_is_perplexity):
            _apply_ylim(
                ax,
                plotted_series,
                log_scale,
                ignore_outliers,
                y_min,
                y_max,
            )
        elif y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_loss_curves(
    logs: List[TrainingLog],
    x_axis: str = "step",
    smooth_window: Optional[int] = None,
    output_path: Union[str, Path, None] = None,
    show: bool = False,
    title: Optional[str] = None,
    log_scale: bool = False,
    ignore_outliers: bool = True,
    perplexity: bool = False,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Figure:
    """Plot training and evaluation loss curves, with learning rate overlay.

    For a single log the figure contains one panel with train loss and eval
    loss on the primary y-axis and learning rate on a secondary y-axis.  When
    multiple logs are supplied the figure splits into two side-by-side panels
    (train loss and eval loss), one line per run, for easy comparison.

    Parameters
    ----------
    logs : list of TrainingLog
        One or more parsed training logs to plot.
    x_axis : {'step', 'epoch', 'time'}, optional
        X-axis variable.  Default is ``'step'``.
    smooth_window : int, optional
        Moving-average window size.  When greater than 1, raw values are shown
        at low opacity and a smoothed overlay is drawn.  Default is ``None``.
    output_path : str or Path, optional
        If provided, the figure is saved to this path at 300 dpi.
    show : bool, optional
        Call ``plt.show()`` after rendering.  Default is ``False``.
    title : str, optional
        Figure title.  When ``None`` a default title is used.
    log_scale : bool, optional
        Use a logarithmic y-axis for loss.  Default is ``False``.
    ignore_outliers : bool, optional
        Apply percentile-based y-axis clipping to loss series.
        Default is ``True``.
    perplexity : bool, optional
        Convert loss values to ``exp(loss)`` before plotting.
        Default is ``False``.
    x_min : float, optional
        Clip data and set the left x-axis limit to this value.
    x_max : float, optional
        Clip data and set the right x-axis limit to this value.
    y_min : float, optional
        Override the bottom y-axis limit for loss axes.
    y_max : float, optional
        Override the top y-axis limit for loss axes.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    multi_run = len(logs) > 1

    kwargs = dict(
        x_axis=x_axis,
        smooth_window=smooth_window,
        title=title,
        log_scale=log_scale,
        ignore_outliers=ignore_outliers,
        perplexity=perplexity,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    if multi_run:
        fig = _plot_loss_curves_multi(logs, **kwargs)
    else:
        fig = _plot_loss_curves_single(logs, **kwargs)

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _loss_axis_label(perplexity: bool, which: str) -> str:
    if perplexity:
        return {
            "loss": "Perplexity",
            "train": "Train Perplexity",
            "eval": "Eval Perplexity",
        }[which]
    return {"loss": "Loss", "train": "Train Loss", "eval": "Eval Loss"}[which]


def _plot_loss_curves_single(
    logs,
    x_axis,
    smooth_window,
    title,
    log_scale,
    ignore_outliers,
    perplexity,
    x_min,
    x_max,
    y_min,
    y_max,
):
    """Single-run loss curves with dual y-axes (loss + LR)."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x_label = "Global Step"
    loss_series: List[List[float]] = []

    for log_idx, log in enumerate(logs):
        label_prefix = log.get_label(log_idx)

        train_records = log.get_training_records()
        train_x: List[float] = []
        if train_records:
            losses = log.get_metric_values("loss", train_records)
            x_values, x_label = _get_x_values(log, train_records, x_axis)
            if perplexity:
                losses = _apply_perplexity(losses)
            x_values, losses = _clip_to_x_window(x_values, losses, x_min, x_max)
            train_x = x_values

            if losses:
                if smooth_window and smooth_window > 1:
                    losses_smooth = smooth_values(losses, smooth_window)
                    ax1.plot(
                        x_values, losses, alpha=0.2, linewidth=0.5, color="tab:blue"
                    )
                    ax1.plot(
                        x_values,
                        losses_smooth,
                        label=f"{label_prefix} {_loss_axis_label(perplexity, 'train')}",
                        linewidth=2,
                        color="tab:blue",
                    )
                    loss_series.append(list(losses_smooth))
                else:
                    ax1.plot(
                        x_values,
                        losses,
                        label=f"{label_prefix} {_loss_axis_label(perplexity, 'train')}",
                        linewidth=2,
                        color="tab:blue",
                    )
                    loss_series.append(list(losses))

        eval_records = log.get_eval_records()
        if eval_records:
            eval_losses = log.get_metric_values("eval_loss", eval_records)
            eval_x, _ = _get_x_values(log, eval_records, x_axis)
            if perplexity:
                eval_losses = _apply_perplexity(eval_losses)
            eval_x, eval_losses = _clip_to_x_window(eval_x, eval_losses, x_min, x_max)
            if eval_losses:
                ax1.plot(
                    eval_x,
                    eval_losses,
                    label=f"{label_prefix} {_loss_axis_label(perplexity, 'eval')}",
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    color="tab:green",
                )
                loss_series.append(list(eval_losses))

        if train_records and train_x:
            learning_rates = log.get_metric_values("learning_rate", train_records)
            raw_x, _ = _get_x_values(log, train_records, x_axis)
            lr_x, lr_y = _clip_to_x_window(raw_x, learning_rates, x_min, x_max)
            if lr_y:
                ax2.plot(
                    lr_x,
                    lr_y,
                    label=f"{label_prefix} LR",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                    color="tab:orange",
                )

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(_loss_axis_label(perplexity, "loss"), color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(loc="upper left")

    ax2.set_ylabel("Learning Rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(loc="upper right")

    if log_scale:
        ax1.set_yscale("log")
    if x_min is not None or x_max is not None:
        ax1.set_xlim(left=x_min, right=x_max)
    if loss_series:
        _apply_ylim(ax1, loss_series, log_scale, ignore_outliers, y_min, y_max)

    plt.title(
        title
        or ("Training Progress" if not perplexity else "Training Progress (Perplexity)")
    )
    plt.tight_layout()
    return fig


def _plot_loss_curves_multi(
    logs,
    x_axis,
    smooth_window,
    title,
    log_scale,
    ignore_outliers,
    perplexity,
    x_min,
    x_max,
    y_min,
    y_max,
):
    """Multi-run comparison: train loss and eval loss in separate subplots."""
    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(16, 6))

    x_label = "Global Step"
    has_eval = False
    train_series: List[List[float]] = []
    eval_series: List[List[float]] = []

    for log_idx, log in enumerate(logs):
        color = _get_color(log_idx)
        label = log.get_label(log_idx)

        train_records = log.get_training_records()
        if train_records:
            losses = log.get_metric_values("loss", train_records)
            x_values, x_label = _get_x_values(log, train_records, x_axis)
            if perplexity:
                losses = _apply_perplexity(losses)
            x_values, losses = _clip_to_x_window(x_values, losses, x_min, x_max)

            if losses:
                if smooth_window and smooth_window > 1:
                    losses_smooth = smooth_values(losses, smooth_window)
                    ax_train.plot(
                        x_values,
                        losses,
                        alpha=0.15,
                        linewidth=0.5,
                        color=color,
                    )
                    ax_train.plot(
                        x_values,
                        losses_smooth,
                        label=label,
                        linewidth=2,
                        color=color,
                    )
                    train_series.append(list(losses_smooth))
                else:
                    ax_train.plot(
                        x_values,
                        losses,
                        label=label,
                        linewidth=2,
                        color=color,
                    )
                    train_series.append(list(losses))

        eval_records = log.get_eval_records()
        if eval_records:
            eval_losses = log.get_metric_values("eval_loss", eval_records)
            eval_x, _ = _get_x_values(log, eval_records, x_axis)
            if perplexity:
                eval_losses = _apply_perplexity(eval_losses)
            eval_x, eval_losses = _clip_to_x_window(eval_x, eval_losses, x_min, x_max)
            if eval_losses:
                has_eval = True
                if smooth_window and smooth_window > 1:
                    eval_smooth = smooth_values(eval_losses, smooth_window)
                    ax_eval.plot(
                        eval_x,
                        eval_losses,
                        alpha=0.15,
                        linewidth=0.5,
                        color=color,
                    )
                    ax_eval.plot(
                        eval_x,
                        eval_smooth,
                        label=label,
                        marker="o",
                        linewidth=2,
                        markersize=4,
                        color=color,
                    )
                    eval_series.append(list(eval_smooth))
                else:
                    ax_eval.plot(
                        eval_x,
                        eval_losses,
                        label=label,
                        marker="o",
                        linewidth=2,
                        markersize=4,
                        color=color,
                    )
                    eval_series.append(list(eval_losses))

    ax_train.set_xlabel(x_label)
    ax_train.set_ylabel(_loss_axis_label(perplexity, "loss"))
    ax_train.set_title(_loss_axis_label(perplexity, "train"))
    ax_train.grid(True, alpha=0.3)
    ax_train.legend()

    ax_eval.set_xlabel(x_label)
    ax_eval.set_ylabel(_loss_axis_label(perplexity, "eval"))
    ax_eval.set_title(_loss_axis_label(perplexity, "eval"))
    ax_eval.grid(True, alpha=0.3)
    if has_eval:
        ax_eval.legend()

    if log_scale:
        ax_train.set_yscale("log")
        ax_eval.set_yscale("log")
    if x_min is not None or x_max is not None:
        ax_train.set_xlim(left=x_min, right=x_max)
        ax_eval.set_xlim(left=x_min, right=x_max)
    if train_series:
        _apply_ylim(ax_train, train_series, log_scale, ignore_outliers, y_min, y_max)
    if eval_series:
        _apply_ylim(ax_eval, eval_series, log_scale, ignore_outliers, y_min, y_max)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_grad_norm(
    logs: List[TrainingLog],
    x_axis: str = "step",
    smooth_window: Optional[int] = None,
    log_scale: bool = False,
    output_path: Union[str, Path, None] = None,
    show: bool = False,
    title: Optional[str] = None,
    ignore_outliers: bool = True,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Figure:
    """Plot gradient norm over training for one or more runs.

    Produces a single-panel figure of the ``grad_norm`` metric.  Only
    training steps that recorded ``grad_norm`` are included (some trainers
    log it at a lower frequency than loss).

    Parameters
    ----------
    logs : list of TrainingLog
        One or more parsed training logs to plot.
    x_axis : {'step', 'epoch', 'time'}, optional
        X-axis variable.  Default is ``'step'``.
    smooth_window : int, optional
        Moving-average window size.  Default is ``None`` (no smoothing).
    log_scale : bool, optional
        Use a logarithmic y-axis.  Default is ``False``.
    output_path : str or Path, optional
        If provided, the figure is saved to this path at 300 dpi.
    show : bool, optional
        Call ``plt.show()`` after rendering.  Default is ``False``.
    title : str, optional
        Plot title.  Defaults to ``'Gradient Norm'``.
    ignore_outliers : bool, optional
        Apply percentile-based y-axis clipping.  Default is ``True``.
    x_min : float, optional
        Clip data and set the left x-axis limit to this value.
    x_max : float, optional
        Clip data and set the right x-axis limit to this value.
    y_min : float, optional
        Override the bottom y-axis limit.
    y_max : float, optional
        Override the top y-axis limit.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x_label = "Global Step"
    plotted_series: List[List[float]] = []
    any_data = False

    for log_idx, log in enumerate(logs):
        color = _get_color(log_idx)
        label = log.get_label(log_idx)
        records = log.get_training_records()
        if not records:
            continue

        values = log.get_metric_values("grad_norm", records)
        if not values:
            continue

        # Records with grad_norm may be a subset of training records.
        gn_records = [r for r in records if "grad_norm" in r]
        x_values, x_label = _get_x_values(log, gn_records, x_axis)
        x_values, values = _clip_to_x_window(x_values, values, x_min, x_max)
        if not x_values:
            continue
        any_data = True

        if smooth_window and smooth_window > 1:
            smoothed = smooth_values(values, smooth_window)
            ax.plot(x_values, values, alpha=0.15, linewidth=0.5, color=color)
            ax.plot(
                x_values,
                smoothed,
                label=label,
                linewidth=2,
                color=color,
            )
            plotted_series.append(list(smoothed))
        else:
            ax.plot(x_values, values, label=label, linewidth=2, color=color)
            plotted_series.append(list(values))

    ax.set_xlabel(x_label)
    ax.set_ylabel("Grad Norm")
    ax.set_title(title or "Gradient Norm")
    ax.grid(True, alpha=0.3)
    if any_data:
        ax.legend()

    if log_scale:
        ax.set_yscale("log")
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if plotted_series:
        _apply_ylim(ax, plotted_series, log_scale, ignore_outliers, y_min, y_max)

    plt.tight_layout()

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig
