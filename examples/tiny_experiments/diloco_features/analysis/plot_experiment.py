#!/usr/bin/env python3
"""Plot the diloco_features comparison sweep from ``assets/curves.csv``.

``curves.csv`` (tidy: series, metric, step, value) is produced by
``analysis/harvest.py`` and is the committed source of truth. This renders the
comparison plots into ``assets/`` and prints the final metrics. Run from the
project directory:

    python analysis/plot_experiment.py

Two figures:
  * ``loss_comparison.png`` — train loss, eval loss, and grad-norm panels. The
    grad-norm panel is the "is training healthy?" check (decreasing/stable, not
    flat or exploding).
  * ``eval_tail.png`` — eval loss over the second half (the feature deltas are
    small; this shows the endgame separation vs the baseline).
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")

# Baseline in black; features in a distinct color each.
LABELS = {
    "baseline": "Baseline (sync, H=100)",
    "streaming": "Streaming (2 fragments)",
    "async": "Async (no DN — diverges)",
    "async_dn": "Async + DN buffer",
    "async_dn_dylu": "Async + DN + DyLU",
}
COLORS = {
    "baseline": "#000000",
    "streaming": "#1f6fb2",
    "async": "#d9772b",
    "async_dn": "#2ca25f",
    "async_dn_dylu": "#c44e52",
}


def load():
    data = defaultdict(lambda: defaultdict(list))  # series -> metric -> [(step, val)]
    path = os.path.join(ASSETS, "curves.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            data[row["series"]][row["metric"]].append(
                (int(row["step"]), float(row["value"]))
            )
    for s in data:
        for m in data[s]:
            data[s][m].sort()
    return data


def main():
    data = load()
    present = [s for s in LABELS if s in data]
    if not present:
        print("No data in assets/curves.csv — run analysis/harvest.py first.")
        return

    hdr = f"{'experiment':<26}{'final train':>13}{'final eval':>13}{'best eval':>13}"
    print(hdr)
    for s in present:
        tr = data[s].get("train_loss", [])
        ev = data[s].get("eval_loss", [])
        ft = tr[-1][1] if tr else float("nan")
        fe = ev[-1][1] if ev else float("nan")
        be = min((v for _, v in ev), default=float("nan"))
        print(f"{LABELS[s]:<26}{ft:>13.4f}{fe:>13.4f}{be:>13.4f}")

    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

    def panel(ax, metric, title, ylabel, markers=False, ylim=None, logy=False):
        for s in present:
            d = data[s].get(metric, [])
            if not d:
                continue
            lw = 2.0 if s == "baseline" else 1.3
            ax.plot(
                [x for x, _ in d],
                [v for _, v in d],
                color=COLORS[s],
                lw=lw,
                marker=("o" if markers else None),
                ms=3,
                label=LABELS[s],
            )
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    # Main comparison: train loss, eval loss, grad norm.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    panel(axes[0], "train_loss", "Train loss", "loss")
    panel(axes[1], "eval_loss", "Eval loss", "loss", markers=True)
    panel(axes[2], "grad_norm", "Grad norm (stability)", "||g||", logy=True)
    fig.suptitle(
        "DiLoCo feature comparison — small Llama (34.4M), ~1B tokens, H=100",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "loss_comparison.png"), bbox_inches="tight")

    # Zoomed eval tail — the feature deltas among the *converged* runs are small;
    # show the endgame. Diverged runs (best eval far above the pack) are excluded
    # so they don't blow out the y-range.
    converged = [
        s
        for s in present
        if min((v for _, v in data[s].get("eval_loss", [])), default=1e9) < 4.0
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    tails = []
    for s in converged:
        ev = data[s].get("eval_loss", [])
        if len(ev) > 5:
            tails += [v for _, v in ev[len(ev) // 2 :]]
    present = converged
    if tails:
        lo, hi = min(tails), max(tails)
        pad = (hi - lo) * 0.1 or 0.01
        panel(
            ax,
            "eval_loss",
            "Eval loss (second half — endgame)",
            "loss",
            markers=True,
            ylim=(lo - pad, hi + pad),
        )
        fig.tight_layout()
        fig.savefig(os.path.join(ASSETS, "eval_tail.png"), bbox_inches="tight")

    print("\nwrote loss_comparison.png + eval_tail.png to", ASSETS)


if __name__ == "__main__":
    main()
