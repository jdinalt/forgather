#!/usr/bin/env python3
"""DN-buffer depth sweep: how deep should the Delayed-Nesterov buffer be?

The async paper's Algorithm 3 defaults the DN buffer to ``N = k`` (the number of
workers, 4 here). That stabilizes async (vs the no-DN control, which diverges),
but from scratch with induced staleness it leaves a large gap to the sync
baseline. This sweep varies *only* the buffer depth — N ∈ {4, 8, 16}, everything
else fixed (async, jitter-induced staleness ~3, 4 workers, H=100, 2B tokens) — to
ask whether depth is the lever.

Finding (single seed): the relationship is **non-monotonic** — N=8 is the
empirical optimum, closing roughly half the from-scratch async gap, while N=16
regresses. So the paper's N=k default is *under*-buffered for from-scratch +
staleness, but "deeper is always better" is false: there is a sweet spot near
2x the mean staleness. The sync baseline is drawn for reference.

Reads the committed ``assets/curves.csv`` (produced by analysis/harvest.py), using
series ``async_dn4`` / ``async_dn8`` / ``async_dn16`` (plus ``baseline``).

    python analysis/dn_sweep.py
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")

# (curves.csv series key, label, color) — buffer depth deepens the green.
RUNS = [
    ("async_dn4", "Async + DN (N=4, = #workers)", "#74c476"),
    ("async_dn8", "Async + DN (N=8)", "#2ca25f"),
    ("async_dn16", "Async + DN (N=16)", "#006d2c"),
]


def load():
    data = defaultdict(lambda: defaultdict(list))  # series -> metric -> [(step, v)]
    with open(os.path.join(ASSETS, "curves.csv")) as f:
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
    base_ev = data.get("baseline", {}).get("eval_loss", [])

    present = [(s, lbl, c) for s, lbl, c in RUNS if data.get(s, {}).get("eval_loss")]
    if not present:
        print("Need async_dn{4,8,16} in curves.csv — run harvest.py")
        return

    print(f"{'run':<34}{'final eval':>12}{'best ppl':>12}")
    fe = {}
    for s, lbl, _ in present:
        ev = data[s]["eval_loss"]
        best = min(v for _, v in ev)
        fe[s] = ev[-1][1]
        print(f"{lbl:<34}{ev[-1][1]:>12.4f}{2.718281828 ** best:>12.2f}")
    if base_ev:
        print(f"{'Sync baseline (reference)':<34}{base_ev[-1][1]:>12.4f}")
        best_async = min(fe, key=fe.get)
        print(
            f"\nbest async = {best_async} (eval {fe[best_async]:.4f}); "
            f"gap to baseline = {fe[best_async] - base_ev[-1][1]:+.4f}"
        )

    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if base_ev:
        ax.plot(
            [s for s, _ in base_ev],
            [v for _, v in base_ev],
            color="#000000",
            lw=1.6,
            ls="--",
            label="Sync baseline (reference)",
        )
    for s, lbl, color in present:
        ev = data[s]["eval_loss"]
        ax.plot(
            [x for x, _ in ev],
            [v for _, v in ev],
            color=color,
            lw=1.5,
            marker="o",
            ms=3,
            label=lbl,
        )
    ax.set_title(
        "DN-buffer depth sweep — N=8 optimal, N=16 regresses (non-monotonic)",
        fontweight="bold",
    )
    ax.set_xlabel("local step (∝ total tokens)")
    ax.set_ylabel("eval loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(ASSETS, "dn_sweep.png")
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
