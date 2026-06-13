#!/usr/bin/env python3
"""Does DyLU actually help? Eval loss with and without DyLU, everything else fixed.

Isolates DyLU's effect with a controlled A/B: both runs use the **same**
per-worker speed spread (0 / 0.05 / 0.10 / 0.15 s), the same async path, and the
same small N=4 DN buffer — they differ only in whether ``--dylu`` is on. With
DyLU off (the control) eval lands ~5.07; with DyLU on it reaches ~4.30, so the
adaptive per-worker ``sync_every`` buys the gap under genuinely uneven workers.
The sync baseline is drawn for reference.

Reads the committed ``assets/curves.csv`` (produced by analysis/harvest.py),
using the series ``dylu`` and ``dylu_control`` (plus ``baseline``).

    python analysis/dylu_control.py
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")

# (curves.csv series key, label, color)
RUNS = [
    ("dylu_control", "Async + DN (N=4), spread, no DyLU", "#d9772b"),
    ("dylu", "Async + DN (N=4), spread, + DyLU", "#c44e52"),
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
    if len(present) < 2:
        print("Need both 'dylu' and 'dylu_control' in curves.csv — run harvest.py")
        return

    print(f"{'run':<38}{'final eval':>12}")
    for s, lbl, _ in present:
        ev = data[s]["eval_loss"]
        print(f"{lbl:<38}{ev[-1][1]:>12.4f}")
    fe = {s: data[s]["eval_loss"][-1][1] for s, _, _ in present}
    if "dylu" in fe and "dylu_control" in fe:
        print(f"\nDyLU effect (control - dylu): {fe['dylu_control'] - fe['dylu']:+.4f}")

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
        "Does DyLU help? Same speed spread + N=4 buffer, DyLU off vs on",
        fontweight="bold",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("eval loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(ASSETS, "dylu_control.png")
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
