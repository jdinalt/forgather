"""Render the README plots for examples/pretrain/small-llm.

Renders thin-line, line-only (no markers) versions of the 10x and 1x
comparison plots, the matching eval-perplexity plots, and the single-run LR
trace. The CLI `forgather logs plot` defaults are too noisy on the train-loss
panel and the marker-per-eval-point style obscures detail when several runs are
closely matched, so we render directly with matplotlib here.

Reads the committed ``curves.csv`` (produced by ``extract_curves.py``), so it
works on a clean checkout with no ``output_models/`` present. The train/LR
series in ``curves.csv`` are downsampled (~MAX_POINTS/run); the train-smoothing
windows below are sized for that downsampled resolution rather than the raw
per-step logs (the eval series are full-resolution).

Run from the project directory:

    python docs/plots/render.py

Outputs are written to docs/plots/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from curves_data import load_curves, series

from forgather.ml.analysis.plotting import smooth_values

OUT = Path(__file__).resolve().parent

LINEWIDTH = 1.1
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

# Run names as stored in curves.csv (the 1x ``default`` is the dense 1x slice of
# ten_chinchilla; see extract_curves.py).
RUNS_10X = ["ten_chinchilla", "long_cooldown", "tiny_x_small_lm", "wds", "final"]
RUNS_1X = ["default", "bf16", "bf16_adafactor", "high_lr", "canon", "muon", "deepone"]


def percentile_ylim(series_list, lo=5, hi=95, pad=0.05):
    flat = [v for s in series_list for v in s if np.isfinite(v)]
    if not flat:
        return None
    p_lo = float(np.percentile(flat, lo))
    p_hi = float(np.percentile(flat, hi))
    span = p_hi - p_lo
    return (p_lo - span * pad, p_hi + span * pad)


def render_train_eval(
    data,
    runs,
    out_path,
    train_smooth,
    eval_smooth,
    x_max=None,
    train_ylim=None,
    eval_ylim=None,
):
    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(16, 6))
    train_series, eval_series = [], []

    for idx, run in enumerate(runs):
        color = PALETTE[idx % len(PALETTE)]

        steps, losses = series(data, run, "train_loss", x_max=x_max)
        if steps:
            smooth = smooth_values(losses, train_smooth)
            ax_t.plot(steps, smooth, label=run, linewidth=LINEWIDTH, color=color)
            train_series.append(list(smooth))

        steps, losses = series(data, run, "eval_loss", x_max=x_max)
        if steps:
            smooth = smooth_values(losses, eval_smooth)
            ax_e.plot(steps, smooth, label=run, linewidth=LINEWIDTH, color=color)
            eval_series.append(list(smooth))

    for ax, title, ylabel, sl, override in (
        (ax_t, "Train Loss", "Loss", train_series, train_ylim),
        (ax_e, "Eval Loss", "Eval Loss", eval_series, eval_ylim),
    ):
        ax.set_title(title)
        ax.set_xlabel("Global Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if override:
            ax.set_ylim(*override)
        else:
            ylim = percentile_ylim(sl)
            if ylim:
                ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print("wrote", out_path)


def render_eval_perplexity(data, runs, out_path, eval_smooth, x_max=None, ylim=None):
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
    eval_series = []

    for idx, run in enumerate(runs):
        color = PALETTE[idx % len(PALETTE)]
        steps, losses = series(data, run, "eval_loss", x_max=x_max)
        if not steps:
            continue
        ppl = [float(np.exp(v)) for v in losses]
        smooth = smooth_values(ppl, eval_smooth)
        ax.plot(steps, smooth, label=run, linewidth=LINEWIDTH, color=color)
        eval_series.append(list(smooth))

    ax.set_title("Eval Perplexity vs Global Step")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Eval Perplexity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if ylim:
        ax.set_ylim(*ylim)
    else:
        auto = percentile_ylim(eval_series, lo=2, hi=98, pad=0.1)
        if auto:
            ax.set_ylim(*auto)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print("wrote", out_path)


def render_single_lr(data, run, out_path, train_smooth, eval_smooth=1, ylim=None):
    """Single-run plot: train+eval loss on left axis, LR on right axis."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    loss_series = []

    steps, losses = series(data, run, "train_loss")
    if steps:
        smooth = smooth_values(losses, train_smooth)
        ax1.plot(
            steps,
            smooth,
            label=f"{run} Train Loss",
            linewidth=LINEWIDTH,
            color="tab:blue",
        )
        loss_series.append(list(smooth))
    lr_steps, lrs = series(data, run, "learning_rate")
    if lr_steps:
        ax2.plot(
            lr_steps,
            lrs,
            label=f"{run} LR",
            linestyle="--",
            alpha=0.8,
            linewidth=1.0,
            color="tab:orange",
        )

    e_steps, e_losses = series(data, run, "eval_loss")
    if e_steps:
        smooth = smooth_values(e_losses, eval_smooth) if eval_smooth > 1 else e_losses
        ax1.plot(
            e_steps,
            smooth,
            label=f"{run} Eval Loss",
            linewidth=LINEWIDTH,
            color="tab:green",
        )
        loss_series.append(list(smooth))

    if ylim:
        ax1.set_ylim(*ylim)
    else:
        auto = percentile_ylim(loss_series)
        if auto:
            ax1.set_ylim(*auto)

    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax2.set_ylabel("Learning Rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.legend(loc="upper right")

    plt.title("Training Progress")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print("wrote", out_path)


def main():
    data = load_curves()
    # Train-smoothing windows are sized for the downsampled curves.csv
    # resolution: the raw 10x logs (~25k pts) used window 1500, downsampled ~21x
    # here -> ~72; the 1x logs (~2.3k pts) used 100, downsampled ~2x -> ~52.
    render_train_eval(
        data,
        RUNS_10X,
        OUT / "10x_chinchilla_comparison.png",
        train_smooth=72,
        eval_smooth=5,
        eval_ylim=(2.20, 2.85),
    )
    render_eval_perplexity(
        data, RUNS_10X, OUT / "10x_eval_perplexity.png", eval_smooth=5, ylim=(9.0, 17.0)
    )
    render_train_eval(
        data,
        RUNS_1X,
        OUT / "1x_chinchilla_comparison.png",
        train_smooth=52,
        eval_smooth=3,
        x_max=36448,
    )
    render_eval_perplexity(
        data,
        RUNS_1X,
        OUT / "1x_eval_perplexity.png",
        eval_smooth=3,
        x_max=36448,
        ylim=(13.5, 25.0),
    )
    render_single_lr(
        data,
        "ten_chinchilla",
        OUT / "10x_ten_chinchilla_lr.png",
        train_smooth=72,
        ylim=(2.25, 3.10),
    )


if __name__ == "__main__":
    main()
