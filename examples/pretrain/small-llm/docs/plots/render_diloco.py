"""Render the DiLoCo-vs-DDPx4 comparison plot for examples/pretrain/small-llm.

Overlays the 4-worker shared-memory DiLoCo run (mean across workers, with a
min/max band) against the DDPx4 ``default`` baseline (the first 1x-Chinchilla
slice of ``ten_chinchilla``). Same model (medium.yaml 162M), dataset, and
per-rank step schedule; the only variable is 4 workers syncing every H=20
steps via outer SGD vs DDP all-reduce every step.

Reads the committed ``curves.csv`` (produced by ``extract_curves.py``), so it
works on a clean checkout with no ``output_models/`` present.

Also renders ``diloco_11x_comparison.png``: the 4-worker DiLoCo 11x run (mean
across workers) against the five DDPx4 10x runs.

Run from the project directory:

    python docs/plots/render_diloco.py

Writes docs/plots/diloco_1x_comparison.png + diloco_11x_comparison.png and
prints summary tables.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from curves_data import load_curves, series

from forgather.ml.analysis.plotting import smooth_values

OUT = Path(__file__).resolve().parent
X_MAX = 36460  # 1x-Chinchilla slice of the 10x baseline
LINEWIDTH = 1.2
WORKERS = ["diloco_w0", "diloco_w1", "diloco_w2", "diloco_w3"]


def worker_mean(data, metric):
    """Mean across workers at each common step (workers eval/log in lockstep)."""
    from collections import defaultdict

    by_step = defaultdict(list)
    for w in WORKERS:
        steps, vals = series(data, w, metric, x_max=X_MAX)
        for s, v in zip(steps, vals):
            by_step[s].append(v)
    steps = sorted(by_step)
    mean = [float(np.mean(by_step[s])) for s in steps]
    lo = [float(np.min(by_step[s])) for s in steps]
    hi = [float(np.max(by_step[s])) for s in steps]
    return steps, mean, lo, hi


def main():
    data = load_curves()

    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(16, 6))

    for metric, ax, smooth_w in (("train_loss", ax_t, 50), ("eval_loss", ax_e, 1)):
        # DiLoCo workers -> mean + band
        steps, mean, lo, hi = worker_mean(data, metric)
        if smooth_w > 1:
            mean = smooth_values(mean, smooth_w)
            lo = smooth_values(lo, smooth_w)
            hi = smooth_values(hi, smooth_w)
        ax.fill_between(steps, lo, hi, color="#d62728", alpha=0.18, linewidth=0)
        ax.plot(
            steps,
            mean,
            color="#d62728",
            linewidth=LINEWIDTH,
            label="DiLoCo 4-worker (H=20, shm)",
        )

        # DDPx4 baseline (1x slice of ten_chinchilla)
        bs, bv = series(data, "default", metric, x_max=X_MAX)
        if smooth_w > 1:
            bv = smooth_values(bv, smooth_w)
        ax.plot(
            bs,
            bv,
            color="#1f77b4",
            linewidth=LINEWIDTH,
            label="DDPx4 baseline (default)",
        )

        title = "Train" if metric == "train_loss" else "Eval"
        ax.set_title(f"{title} Loss vs step (1x Chinchilla)")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # focus eval y-axis on the informative tail
    ax_e.set_ylim(2.5, 3.6)
    fig.tight_layout()
    out = OUT / "diloco_1x_comparison.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")

    # summary
    def best_last(run):
        steps, ev = series(data, run, "eval_loss", x_max=X_MAX)
        return (min(ev), (steps[-1], ev[-1])) if ev else (None, None)

    print("\n=== eval-loss summary (1x slice, step <= %d) ===" % X_MAX)
    bests, lasts = [], []
    for w in WORKERS:
        b, l = best_last(w)
        bests.append(b)
        lasts.append(l[1])
        print(f"  DiLoCo {w}: best={b:.4f} last={l[1]:.4f}@{l[0]}")
    print(
        f"  DiLoCo group: best(mean)={np.mean(bests):.4f}  last(mean)={np.mean(lasts):.4f}"
    )
    bb, bl = best_last("default")
    print(f"  DDPx4 baseline: best={bb:.4f} last={bl[1]:.4f}@{bl[0]}")
    print(f"  delta (DiLoCo best mean - baseline best): {np.mean(bests) - bb:+.4f}")


RUNS_10X_DDP = ["ten_chinchilla", "long_cooldown", "tiny_x_small_lm", "wds", "final"]
PALETTE_10X = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]


def render_11x(data):
    """DiLoCo 11x (4-worker mean) vs the five DDPx4 10x runs — train + eval."""
    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(16, 6))
    for metric, ax, smooth_w in (("train_loss", ax_t, 72), ("eval_loss", ax_e, 3)):
        for i, run in enumerate(RUNS_10X_DDP):
            steps, vals = series(data, run, metric)
            if not steps:
                continue
            sm = smooth_values(vals, smooth_w) if smooth_w > 1 else vals
            ax.plot(
                steps, sm, color=PALETTE_10X[i], linewidth=1.0, alpha=0.85, label=run
            )
        steps, vals = series(data, "diloco11x", metric)
        sm = smooth_values(vals, smooth_w) if smooth_w > 1 else vals
        ax.plot(
            steps, sm, color="#d62728", linewidth=2.2, label="DiLoCo 11x (4-worker)"
        )
        ax.set_title(
            f"{'Train' if metric == 'train_loss' else 'Eval'} Loss vs step (10x + 1x anneal)"
        )
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax_t.set_ylim(2.55, 2.95)
    ax_e.set_ylim(2.2, 2.6)  # focus on the informative tail
    fig.tight_layout()
    out = OUT / "diloco_11x_comparison.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")

    def best(run):
        _, ev = series(data, run, "eval_loss")
        return min(ev) if ev else None

    print("\n=== 11x best-eval ranking ===")
    for b, r in sorted((best(r), r) for r in RUNS_10X_DDP + ["diloco11x"]):
        tag = "  <- DiLoCo" if r == "diloco11x" else ""
        print(f"  {r:<18} {b:.4f}{tag}")


if __name__ == "__main__":
    main()
    render_11x(load_curves())
