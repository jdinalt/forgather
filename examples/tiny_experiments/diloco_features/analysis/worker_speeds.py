#!/usr/bin/env python3
"""Verify the DyLU speed-spread assumption from the captured run data.

The DyLU arms inject a *fixed* per-worker step delay
(`DILOCO_DEBUG_STEP_DELAY`, set from `experiment.sh`'s `DYLU_SPREAD`) so the four
workers run at different average speeds — the heterogeneity DyLU exists to absorb.
This script measures, from `runs/<arm>/worker*.log`, each worker's *actual* step
rate and confirms the spread is real and correctly ordered (fast → slow tracks the
injected delay), and quantifies how big it actually was.

Method: each worker's TTY log prints a timestamped row every `logging_steps`
steps; the median wall-clock time per step over all logged intervals (robust to
warmup/compile and the occasional sync stall) is the worker's step time. Worker k
in `runs/<arm>/worker<k>.log` was launched with delay `DYLU_SPREAD[k]`.

    python analysis/worker_speeds.py

Prints a per-worker table and writes `assets/worker_speeds.png`.
"""

import datetime
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")
RUNS = os.path.join(HERE, "runs")

# Must match experiment.sh's DYLU_SPREAD (seconds of fixed per-step delay).
DYLU_SPREAD = [0.0, 0.24, 0.40, 0.56]
ARMS = ["warm_dylu_off", "warm_dylu_on"]

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_TR = re.compile(r"^\d{4}-\d{2}-\d{2}\s+[\d:]+\s+(\d[\d,]*)\s")


def step_time_ms(path):
    """Median ms/step + total steps from one worker log, or None."""
    rows = []
    for line in open(path, errors="replace"):
        mt, tr = _TS.match(line), _TR.match(line)
        if mt and tr:
            wt = datetime.datetime.strptime(
                mt.group(1), "%Y-%m-%d %H:%M:%S"
            ).timestamp()
            rows.append((wt, int(tr.group(1).replace(",", ""))))
    if len(rows) < 3:
        return None
    rows.sort()
    # per-interval seconds/step, skipping the first interval (compile/warmup)
    per = [
        (rows[i][0] - rows[i - 1][0]) / (rows[i][1] - rows[i - 1][1])
        for i in range(2, len(rows))
        if rows[i][1] > rows[i - 1][1]
    ]
    if not per:
        return None
    return statistics.median(per) * 1000.0, rows[-1][1]


def main():
    results = {}  # arm -> list of (delay, ms/step, steps_per_s, total_steps)
    for arm in ARMS:
        rows = []
        for k, delay in enumerate(DYLU_SPREAD):
            s = step_time_ms(os.path.join(RUNS, arm, f"worker{k}.log"))
            if s:
                ms, total = s
                rows.append((delay, ms, 1000.0 / ms, total))
        if rows:
            results[arm] = rows

    if not results:
        print("No dylu_* worker logs found — run the DyLU arms first.")
        return

    print(
        f"{'arm':<10}{'worker':<8}{'delay(s)':>9}{'ms/step':>9}"
        f"{'steps/s':>9}{'steps':>10}"
    )
    for arm, rows in results.items():
        base = rows[0][1]
        for k, (delay, ms, sps, total) in enumerate(rows):
            print(f"{arm:<10}w{k:<7}{delay:>9.2f}{ms:>9.1f}{sps:>9.2f}{total:>10,}")
        ratio = rows[-1][1] / rows[0][1]
        print(
            f"  -> {arm}: slowest/fastest step time = {ratio:.2f}x "
            f"(monotonic with delay: {all(rows[i][1] >= rows[i-1][1] for i in range(1, len(rows)))})\n"
        )

    # Figure: per-worker step time, grouped by arm — the spread, and that it is
    # consistent across the DyLU off/on A/B (DyLU changes sync cadence, not the
    # injected compute speed).
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    n = len(DYLU_SPREAD)
    x = range(n)
    w = 0.38
    colors = {"warm_dylu_off": "#d9772b", "warm_dylu_on": "#7b3294"}
    for i, (arm, rows) in enumerate(results.items()):
        ms = [r[1] for r in rows]
        ax.bar(
            [xi + (i - 0.5) * w for xi in x],
            ms,
            w,
            label=arm,
            color=colors.get(arm, None),
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"w{k}\n(+{d:.2f}s)" for k, d in enumerate(DYLU_SPREAD)], fontsize=9
    )
    ax.set_ylabel("median wall-clock ms / step")
    ax.set_xlabel("worker (injected fixed delay)")
    ax.set_title(
        "DyLU speed spread — measured per-worker step time (fast → slow)",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = os.path.join(ASSETS, "worker_speeds.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
