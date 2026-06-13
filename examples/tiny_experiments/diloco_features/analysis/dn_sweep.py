#!/usr/bin/env python3
"""Plot the async DN-buffer-size sweep: how async convergence depends on N.

Async DiLoCo needs the Delayed-Nesterov buffer (`--dn-buffer-size N`) to be
stable, but the buffer size is also a convergence knob. This sweeps N over
{4, 8, 16} (= 1x / 2x / 4x the worker count, 4 workers) at the same budget/seed/
data and shows the eval-loss trajectory and final loss vs N, against the sync
baseline.

Reads the committed ``assets/curves.csv`` (produced by analysis/harvest.py),
using the series ``async_dn_b4`` (N=4), ``async_dn_b8`` (N=8), ``async_dn_b16``
(N=16), plus ``baseline``.

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

# (N, curves.csv series key, color)
SWEEP = [
    (4, "async_dn_b4", "#2ca25f"),
    (8, "async_dn_b8", "#d9772b"),
    (16, "async_dn_b16", "#c44e52"),
]


def _curves():
    data = defaultdict(lambda: defaultdict(list))  # series -> metric -> [(step, v)]
    with open(os.path.join(ASSETS, "curves.csv")) as f:
        for row in csv.DictReader(f):
            data[row["series"]][row["metric"]].append(
                (int(row["step"]), float(row["value"]))
            )
    return data


_DATA = _curves()


def load(series_key):
    d = _DATA.get(series_key, {})
    return sorted(d.get("train_loss", [])), sorted(d.get("eval_loss", []))


def main():
    base_tr, base_ev = load("baseline")
    points = []  # (N, final_eval)
    series = []  # (N, color, eval_curve)
    for n, run_dir, color in SWEEP:
        tr, ev = load(run_dir)
        if not ev:
            print(f"  skip N={n}: no data at runs/{run_dir}/worker0.log")
            continue
        series.append((n, color, ev))
        points.append((n, ev[-1][1]))

    if not series:
        print("No DN-sweep data — run `./experiment.sh dnsweep` first.")
        return

    print(f"{'DN buffer N':<14}{'final eval':>12}{'vs sync':>10}")
    base_fe = base_ev[-1][1] if base_ev else float("nan")
    print(f"{'sync (no async)':<14}{base_fe:>12.4f}{'—':>10}")
    for n, fe in points:
        print(f"{'N=' + str(n):<14}{fe:>12.4f}{fe - base_fe:>+10.4f}")

    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
    fig, (axc, axs) = plt.subplots(1, 2, figsize=(12, 4.4))

    # Left: eval-loss trajectories.
    if base_ev:
        axc.plot(
            [s for s, _ in base_ev],
            [v for _, v in base_ev],
            color="#000000",
            lw=2.0,
            label="sync baseline",
        )
    for n, color, ev in series:
        axc.plot(
            [s for s, _ in ev],
            [v for _, v in ev],
            color=color,
            lw=1.4,
            marker="o",
            ms=3,
            label=f"async, DN N={n}",
        )
    axc.set_title("Eval loss vs DN buffer size")
    axc.set_xlabel("step")
    axc.set_ylabel("eval loss")
    axc.legend(fontsize=8)
    axc.grid(alpha=0.25)

    # Right: final eval vs N.
    if points:
        ns = [n for n, _ in points]
        fes = [fe for _, fe in points]
        axs.plot(ns, fes, color="#1f6fb2", lw=1.6, marker="o", ms=6, label="async + DN")
        if base_ev:
            axs.axhline(
                base_fe, color="#000000", lw=1.5, ls="--", label="sync baseline"
            )
        axs.set_title("Final eval loss vs DN buffer size")
        axs.set_xlabel("DN buffer size N  (worker count = 4)")
        axs.set_ylabel("final eval loss")
        axs.set_xticks(ns)
        axs.legend(fontsize=8)
        axs.grid(alpha=0.25)

    fig.suptitle(
        "Async DN-buffer-size sweep — small Llama (34.4M), 4 workers x 520M tokens, H=100",
        fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(ASSETS, "dn_sweep.png")
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
