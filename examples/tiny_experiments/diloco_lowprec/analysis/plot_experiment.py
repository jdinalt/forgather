#!/usr/bin/env python3
"""Plot the low-precision DiLoCo sweep from ``assets/curves.csv``.

``curves.csv`` (tidy: series, metric, step, value) is produced by
``analysis/harvest.py`` from the run logs and is the committed source of truth.
This script renders the comparison plots into ``assets/`` and prints the final
metrics. Run from the project directory:

    python analysis/plot_experiment.py
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")

# Plot order, labels, colors. AMP variants in blues/greens; bf16-weights in reds.
LABELS = {
    "b0": "B0 baseline (down fp32)",
    "e1": "E1 down bf16",
    "e2": "E2 down bf16+SR",
    "e3": "E3 up bf16+SR",
    "e4": "E4 bf16w, down bf16+SR",
    "e5": "E5 bf16w, down bf16",
}
COLORS = {
    "b0": "#1f6fb2",
    "e1": "#d9772b",
    "e2": "#2ca25f",
    "e3": "#9467bd",
    "e4": "#c44e52",
    "e5": "#8c8c8c",
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

    print(f"{'experiment':<26}{'final train':>13}{'final eval':>13}{'best eval':>13}")
    for s in present:
        tr = data[s].get("train_loss", [])
        ev = data[s].get("eval_loss", [])
        ft = tr[-1][1] if tr else float("nan")
        fe = ev[-1][1] if ev else float("nan")
        be = min((v for _, v in ev), default=float("nan"))
        print(f"{LABELS[s]:<26}{ft:>13.4f}{fe:>13.4f}{be:>13.4f}")

    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

    def panel(ax, metric, title, ylabel, markers=False, ylim=None):
        for s in present:
            d = data[s].get(metric, [])
            if not d:
                continue
            ax.plot(
                [x for x, _ in d],
                [v for _, v in d],
                color=COLORS[s],
                lw=1.3,
                marker=("o" if markers else None),
                ms=3,
                label=LABELS[s],
            )
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    # Full curves.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    panel(axes[0], "train_loss", "Train loss", "loss")
    panel(axes[1], "eval_loss", "Eval loss", "loss", markers=True)
    fig.suptitle(
        "Low-precision DiLoCo — small Llama (34.4M), ~1B tokens, H=100",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "loss_comparison.png"), bbox_inches="tight")

    # Zoomed eval tail — the differences are small, so show the endgame.
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    tails = []
    for s in present:
        ev = data[s].get("eval_loss", [])
        if len(ev) > 5:
            tails += [v for _, v in ev[len(ev) // 2 :]]
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
