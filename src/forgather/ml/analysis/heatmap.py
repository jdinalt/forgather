"""Heatmap visualization for per-parameter diagnostic logs."""

import json
import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Map from JSON record key to display label
_METRIC_LABELS = {
    "norm": "L2 Norm",
    "spectral_norm": "Spectral Norm",
    "grad_norm": "Gradient L2 Norm",
}

# Map from metric name to the JSON record key that contains the data dict
_METRIC_KEYS = {
    "norm": "norms",
    "spectral_norm": "spectral_norms",
    "grad_norm": "grad_norms",
}


def _load_diagnostic_log(log_path: str | Path) -> list[dict]:
    """Load a diagnostic JSON log file, handling common corruption modes."""
    log_path = Path(log_path)
    content = log_path.read_text()
    content = content.strip()
    if not content:
        return []

    # Try direct parse
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try adding closing bracket (unclean shutdown)
    if not content.endswith("]"):
        trimmed = content.rstrip(",\n\r\t ")
        try:
            result = json.loads(trimmed + "\n]")
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Last record may be partially written
        last_brace = trimmed.rfind("}")
        if last_brace > 0:
            attempt = trimmed[: last_brace + 1].rstrip(",\n\r\t ") + "\n]"
            try:
                result = json.loads(attempt)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

    # Handle trailing comma inside otherwise valid array
    if content.endswith("]"):
        inner = content[:-1].rstrip(",\n\r\t ") + "\n]"
        try:
            result = json.loads(inner)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse diagnostic log %s, returning empty list", log_path)
    return []


def _detect_metric(records: list[dict]) -> str | None:
    """Auto-detect the metric type from record keys."""
    if not records:
        return None
    first = records[0]
    for metric_name, key in _METRIC_KEYS.items():
        if key in first:
            return metric_name
    return None


def _extract_heatmap_data(
    records: list[dict],
    metric: str,
    step_stride: int = 1,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Extract a 2D array from diagnostic log records.

    Returns:
        (data, param_names, steps) where data has shape (n_params, n_steps).
    """
    data_key = _METRIC_KEYS[metric]

    # Filter records that have the metric key and apply stride
    filtered = []
    for i, rec in enumerate(records):
        if data_key not in rec:
            continue
        if step_stride > 1 and i % step_stride != 0:
            continue
        filtered.append(rec)

    if not filtered:
        raise ValueError(f"No records found with metric key '{data_key}'")

    # Collect all parameter names (preserving insertion order from first record)
    param_names = list(filtered[0][data_key].keys())

    # Build 2D array
    steps = []
    data_rows = []
    for rec in filtered:
        steps.append(rec.get("global_step", 0))
        values = rec[data_key]
        row = [values.get(name, float("nan")) for name in param_names]
        data_rows.append(row)

    # Shape: (n_params, n_steps) — parameters on y-axis, steps on x-axis
    data = np.array(data_rows, dtype=np.float64).T

    return data, param_names, steps


def plot_parameter_heatmap(
    log_path: str | Path,
    metric: str | None = None,
    step_stride: int = 1,
    log_scale: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
) -> matplotlib.figure.Figure:
    """Plot a per-parameter heatmap from a diagnostic log file.

    Generates a grid with parameter FQN labels on the y-axis, training
    steps on the x-axis, and cells color-coded by the metric value.
    Follows the visualization format from the appendix of arxiv 2510.04212.

    Args:
        log_path: Path to a parameter_norms.json or gradient_norms.json file.
        metric: Which metric to plot ("norm", "spectral_norm", "grad_norm").
                Auto-detected from file content if None.
        step_stride: Plot every Nth step (default: 1, all eval steps).
        log_scale: Use log scale for color mapping.
        vmin: Manual minimum for color range.
        vmax: Manual maximum for color range.
        title: Custom plot title.
        output_path: Path to save the figure. None to skip saving.
        figsize: Figure size as (width, height). Auto-scaled if None.
        show: Whether to display the plot interactively.

    Returns:
        Matplotlib figure object.
    """
    records = _load_diagnostic_log(log_path)
    if not records:
        raise ValueError(f"No records found in {log_path}")

    if metric is None:
        metric = _detect_metric(records)
        if metric is None:
            raise ValueError(
                f"Could not auto-detect metric from {log_path}. "
                f"Expected keys: {list(_METRIC_KEYS.values())}"
            )

    data, param_names, steps = _extract_heatmap_data(records, metric, step_stride)
    n_params, n_steps = data.shape

    # Auto-scale figure size
    if figsize is None:
        width = max(10, n_steps * 0.15 + 2)
        height = max(6, n_params * 0.25 + 2)
        # Cap at reasonable sizes
        width = min(width, 40)
        height = min(height, 60)
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize)

    # Color normalization
    norm = None
    if log_scale:
        # Avoid log of zero/negative
        data_min = np.nanmin(data[data > 0]) if np.any(data > 0) else 1e-10
        effective_vmin = vmin if vmin is not None else data_min
        effective_vmax = vmax if vmax is not None else np.nanmax(data)
        norm = mcolors.LogNorm(vmin=effective_vmin, vmax=effective_vmax)
    else:
        if vmin is not None or vmax is not None:
            norm = mcolors.Normalize(
                vmin=vmin if vmin is not None else np.nanmin(data),
                vmax=vmax if vmax is not None else np.nanmax(data),
            )

    # Plot heatmap
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="viridis",
        norm=norm,
        interpolation="nearest",
    )

    # Colorbar
    metric_label = _METRIC_LABELS.get(metric, metric)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(metric_label)

    # Y-axis: parameter names
    ax.set_yticks(range(n_params))
    # Truncate long names for readability
    display_names = []
    for name in param_names:
        if len(name) > 60:
            display_names.append("..." + name[-57:])
        else:
            display_names.append(name)
    ax.set_yticklabels(display_names, fontsize=max(4, min(8, 200 // n_params)))

    # X-axis: step numbers
    if n_steps <= 30:
        ax.set_xticks(range(n_steps))
        ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right", fontsize=8)
    else:
        # Show subset of tick labels
        tick_stride = max(1, n_steps // 20)
        tick_positions = list(range(0, n_steps, tick_stride))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [str(steps[i]) for i in tick_positions],
            rotation=45,
            ha="right",
            fontsize=8,
        )

    ax.set_xlabel("Global Step")
    ax.set_ylabel("Parameter")

    if title is None:
        title = f"{metric_label} per Parameter"
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig
