"""Render the README plots for examples/pretrain/small-llm.

Renders thin-line, line-only (no markers) versions of the 10x and 1x
comparison plots and the matching eval-perplexity plots. The CLI
`forgather logs plot` defaults are too noisy on the train-loss panel and
the marker-per-eval-point style obscures detail when several runs are
closely matched, so we render directly with matplotlib here.

Run from the project directory:

    python docs/plots/render.py

Outputs are written to docs/plots/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forgather.ml.analysis import TrainingLog
from forgather.ml.analysis.plotting import smooth_values

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "plots"

LINEWIDTH = 1.1
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

RUNS_10X = [
    (
        "ten_chinchilla",
        "output_models/ten_chinchilla/runs/log_2026-04-02T00-02-58/trainer_logs.json",
    ),
    (
        "long_cooldown",
        "output_models/long_cooldown/runs/log_2026-04-06T07-14-55/trainer_logs.json",
    ),
    (
        "tiny_x_small_lm",
        "output_models/tiny_x_small_lm/runs/log_2026-04-19T22-50-21/trainer_logs.json",
    ),
    ("wds", "output_models/wds/runs/log_2026-04-22T04-05-02/trainer_logs.json"),
    ("final", "output_models/final/runs/log_2026-04-26T11-00-31/trainer_logs.json"),
]

RUNS_1X = [
    (
        "default",
        "output_models/ten_chinchilla/runs/log_2026-04-02T00-02-58/trainer_logs.json",
    ),
    ("bf16", "output_models/bf16/runs/log_2026-04-05T00-51-49/trainer_logs.json"),
    (
        "bf16_adafactor",
        "output_models/bf16_adafactor/runs/log_2026-04-04T20-52-11/trainer_logs.json",
    ),
    ("high_lr", "output_models/high_lr/runs/log_2026-04-08T14-47-03/trainer_logs.json"),
    ("canon", "output_models/canon/runs/log_2026-04-05T20-04-51/trainer_logs.json"),
    ("muon", "output_models/muon/runs/log_2026-04-28T08-22-52/trainer_logs.json"),
    ("deepone", "output_models/deepone/runs/log_2026-04-24T09-25-27/trainer_logs.json"),
]


def percentile_ylim(series_list, lo=5, hi=95, pad=0.05):
    flat = [v for s in series_list for v in s if np.isfinite(v)]
    if not flat:
        return None
    p_lo = float(np.percentile(flat, lo))
    p_hi = float(np.percentile(flat, hi))
    span = p_hi - p_lo
    return (p_lo - span * pad, p_hi + span * pad)


def clip_xy(steps, values, x_max):
    if x_max is None:
        return list(steps), list(values)
    out_s, out_v = [], []
    for s, v in zip(steps, values):
        if s <= x_max:
            out_s.append(s)
            out_v.append(v)
    return out_s, out_v


def render_train_eval(
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

    for idx, (label, path) in enumerate(runs):
        log = TrainingLog.from_file(str(ROOT / path))
        color = PALETTE[idx % len(PALETTE)]

        train = log.get_training_records()
        if train:
            steps = [r["global_step"] for r in train]
            losses = [r["loss"] for r in train]
            steps, losses = clip_xy(steps, losses, x_max)
            smooth = smooth_values(losses, train_smooth)
            ax_t.plot(steps, smooth, label=label, linewidth=LINEWIDTH, color=color)
            train_series.append(list(smooth))

        evals = log.get_eval_records()
        if evals:
            steps = [r["global_step"] for r in evals]
            losses = [r["eval_loss"] for r in evals]
            steps, losses = clip_xy(steps, losses, x_max)
            smooth = smooth_values(losses, eval_smooth)
            ax_e.plot(steps, smooth, label=label, linewidth=LINEWIDTH, color=color)
            eval_series.append(list(smooth))

    for ax, title, ylabel, series, override in (
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
            ylim = percentile_ylim(series)
            if ylim:
                ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print("wrote", out_path)


def render_eval_perplexity(runs, out_path, eval_smooth, x_max=None, ylim=None):
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
    eval_series = []

    for idx, (label, path) in enumerate(runs):
        log = TrainingLog.from_file(str(ROOT / path))
        color = PALETTE[idx % len(PALETTE)]
        evals = log.get_eval_records()
        if not evals:
            continue
        steps = [r["global_step"] for r in evals]
        losses = [r["eval_loss"] for r in evals]
        steps, losses = clip_xy(steps, losses, x_max)
        ppl = [float(np.exp(v)) for v in losses]
        smooth = smooth_values(ppl, eval_smooth)
        ax.plot(steps, smooth, label=label, linewidth=LINEWIDTH, color=color)
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


def render_single_lr(label, path, out_path, train_smooth, eval_smooth=1, ylim=None):
    """Single-run plot: train+eval loss on left axis, LR on right axis.

    Mirrors `forgather logs plot --loss-curves` for one run, but with thin
    lines and no markers on the eval series.
    """
    log = TrainingLog.from_file(str(ROOT / path))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    loss_series = []

    train = log.get_training_records()
    if train:
        steps = [r["global_step"] for r in train]
        losses = [r["loss"] for r in train]
        smooth = smooth_values(losses, train_smooth)
        ax1.plot(
            steps,
            smooth,
            label=f"{label} Train Loss",
            linewidth=LINEWIDTH,
            color="tab:blue",
        )
        loss_series.append(list(smooth))
        lrs = [r.get("learning_rate") for r in train]
        if any(v is not None for v in lrs):
            ax2.plot(
                steps,
                lrs,
                label=f"{label} LR",
                linestyle="--",
                alpha=0.8,
                linewidth=1.0,
                color="tab:orange",
            )

    evals = log.get_eval_records()
    if evals:
        steps = [r["global_step"] for r in evals]
        losses = [r["eval_loss"] for r in evals]
        smooth = smooth_values(losses, eval_smooth) if eval_smooth > 1 else losses
        ax1.plot(
            steps,
            smooth,
            label=f"{label} Eval Loss",
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
    render_train_eval(
        RUNS_10X,
        OUT / "10x_chinchilla_comparison.png",
        train_smooth=1500,
        eval_smooth=5,
        eval_ylim=(2.20, 2.85),
    )
    render_eval_perplexity(
        RUNS_10X, OUT / "10x_eval_perplexity.png", eval_smooth=5, ylim=(9.0, 17.0)
    )
    render_train_eval(
        RUNS_1X,
        OUT / "1x_chinchilla_comparison.png",
        train_smooth=100,
        eval_smooth=3,
        x_max=36448,
    )
    render_eval_perplexity(
        RUNS_1X,
        OUT / "1x_eval_perplexity.png",
        eval_smooth=3,
        x_max=36448,
        ylim=(13.5, 25.0),
    )
    render_single_lr(
        "ten_chinchilla",
        "output_models/ten_chinchilla/runs/log_2026-04-02T00-02-58/trainer_logs.json",
        OUT / "10x_ten_chinchilla_lr.png",
        train_smooth=1500,
        ylim=(2.25, 3.10),
    )


if __name__ == "__main__":
    main()
