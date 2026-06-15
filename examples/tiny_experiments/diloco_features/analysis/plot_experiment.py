#!/usr/bin/env python3
"""Plot the diloco_features comparison sweep from ``assets/curves.csv``.

``curves.csv`` (tidy: series, metric, step, value) is produced by
``analysis/harvest.py`` and is the committed source of truth. This renders the
comparison plots into ``assets/`` and prints the final metrics. Run from the
project directory:

    python analysis/plot_experiment.py

Two figures, two panels each (so each panel stays readable when the docs compress
a figure to the text-column width):
  * ``loss_comparison.png`` — eval loss (full) + the converged-runs endgame zoom.
  * ``training_health.png`` — train loss + grad norm (the "is training healthy?"
    check: decreasing/stable, not flat or exploding).
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")

# The ASYNC + DN cluster only (equal-speed workers + injected jitter): sync
# baseline vs async-no-DN (diverges) vs the DN-buffer depth sweep N=4/8/16. The
# other axes are reported separately — they are NOT directly comparable to these:
# streaming (synchronous, fragmented comm) lives in analysis/streaming.py +
# plot_walltime.py; the DyLU on/off A/B (run under a speed spread, not equal speed)
# in analysis/dylu_control.py; the warm-start follow-up in analysis/warm_compare.py.
LABELS = {
    "baseline": "Baseline (sync, H=100)",
    "async_nodn": "Async (no DN)",
    "async_dn4": "Async + DN (N=4)",
    "async_dn8": "Async + DN (N=8)",
    "async_dn16": "Async + DN (N=16)",
}
COLORS = {
    "baseline": "#000000",
    "async_nodn": "#d9772b",
    "async_dn4": "#74c476",
    "async_dn8": "#2ca25f",
    "async_dn16": "#006d2c",
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

    # Two panels per figure (not three) so each renders wide enough to read when
    # the docs compress a figure to the text-column width.
    def panel(
        ax, metric, title, ylabel, series_list, markers=False, ylim=None, logy=False
    ):
        for s in series_list:
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
        ax.set_xlabel("local step (∝ total tokens)")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    suptitle = "small Llama (34.4M), 4 workers, 2B total tokens, H=100"

    # Endgame (converged runs only): the deltas among the converged runs are
    # small, so a zoom shows the separation; diverged runs would blow out the
    # y-range, so exclude them.
    CONVERGED_EVAL_MAX = 4.0  # best-eval above this == didn't converge here
    converged = [
        s
        for s in present
        if min((v for _, v in data[s].get("eval_loss", [])), default=1e9)
        < CONVERGED_EVAL_MAX
    ]
    for s in present:
        if s not in converged:
            print(f"  endgame panel: excluding non-converged series '{s}'")

    # Figure 1 (headline): eval loss — full, and the converged-runs endgame zoom.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panel(axes[0], "eval_loss", "Eval loss", "loss", present, markers=True)
    tails = [
        v
        for s in converged
        for _, v in data[s].get("eval_loss", [])[
            len(data[s].get("eval_loss", [])) // 2 :
        ]
    ]
    if tails:
        lo, hi = min(tails), max(tails)
        pad = (hi - lo) * 0.1 or 0.01
        # restrict the endgame x-range to the second half too
        for s in converged:
            ev = data[s].get("eval_loss", [])
            if ev:
                axes[1].set_xlim(ev[len(ev) // 2][0], ev[-1][0])
        panel(
            axes[1],
            "eval_loss",
            "Eval loss (endgame — converged)",
            "loss",
            converged,
            markers=True,
            ylim=(lo - pad, hi + pad),
        )
    fig.suptitle(f"Async + DN — {suptitle}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "loss_comparison.png"), bbox_inches="tight")

    # Figure 2 (training health): train loss + grad norm.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panel(axes[0], "train_loss", "Train loss", "loss", present)
    panel(axes[1], "grad_norm", "Grad norm (stability)", "||g||", present, logy=True)
    fig.suptitle(f"Async + DN — training health — {suptitle}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "training_health.png"), bbox_inches="tight")

    print("\nwrote loss_comparison.png + training_health.png to", ASSETS)


if __name__ == "__main__":
    main()
