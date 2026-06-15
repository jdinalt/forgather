#!/usr/bin/env python3
"""Plot eval loss vs *relative wall-clock time* per arm — the fair comparison for
features that trade per-token quality for throughput.

``plot_experiment.py`` plots loss vs local step (∝ tokens), which is the right
axis for "how good is the model per unit of data". But Streaming DiLoCo's whole
point is overlapping communication with compute: it can be worse per token yet
*faster in wall-clock*, so on a loss-vs-time axis a streaming arm can cross over
and match (or beat) the sync baseline. This script renders that axis.

Source: the captured per-worker TTY logs under ``runs/<arm>/worker0.log`` (rank 0
runs eval), the same logs ``harvest.py`` and ``regen_tb.py`` read. Wall-clock is
the row timestamp; per arm we plot elapsed seconds from that arm's first training
row (so arms are compared on equal "time since start", independent of when they
ran). Run from the project directory:

    python analysis/plot_walltime.py

Writes ``assets/walltime_comparison.png`` (two panels: the headline feature set,
and a streaming-focused view).
"""

import datetime
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")
RUNS = os.path.join(HERE, "runs")

# Reuse harvest/regen_tb log-line grammar so the parse stays consistent.
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_EVAL = re.compile(r"\s(\d[\d,]*)\s+" + _NUM + r"\s+eval-loss:\s*(" + _NUM + r")")

# Superset of arms (a subset is present in any given run). Baseline black; the
# streaming arms in blues (the wall-clock story); the async/DN family in
# greens/reds (the DN-buffer sweep dn4 -> dn8 -> dn16 deepens green).
LABELS = {
    "baseline": "Baseline (sync, H=100)",
    "stream_str2": "Streaming (strided, 2 frag)",
    "stream_seq2": "Streaming (sequential, 2 frag)",
    "stream_str5": "Streaming (strided, 5 frag)",
    "async_nodn": "Async (no DN)",
    "async_dn4": "Async + DN (N=4)",
    "async_dn8": "Async + DN (N=8)",
    "async_dn16": "Async + DN (N=16)",
    "dylu_off": "Async + DN, DyLU off",
    "dylu_on": "Async + DN + DyLU",
}
COLORS = {
    "baseline": "#000000",
    "stream_str2": "#1f6fb2",
    "stream_seq2": "#6baed6",
    "stream_str5": "#08306b",
    "async_nodn": "#d9772b",
    "async_dn4": "#74c476",
    "async_dn8": "#2ca25f",
    "async_dn16": "#006d2c",
    "dylu_off": "#c44e52",
    "dylu_on": "#7b3294",
}

STREAMING_SET = ["baseline", "stream_str2", "stream_seq2", "stream_str5"]


def _walltime(line):
    m = _TS.match(line)
    if not m:
        return None
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()


def load_arm(arm):
    """Return [(elapsed_seconds, eval_loss), ...] from runs/<arm>/worker0.log.

    Elapsed is measured from the first timestamped row in the log (training
    start), so arms are aligned on "time since the run began".
    """
    log = os.path.join(RUNS, arm, "worker0.log")
    if not os.path.isfile(log):
        return []
    t0 = None
    pts = []
    with open(log, errors="replace") as f:
        for line in f:
            wt = _walltime(line)
            if wt is None:
                continue
            if t0 is None:
                t0 = wt
            if "eval-loss:" in line:
                m = _EVAL.search(line)
                if m:
                    pts.append((wt - t0, float(m.group(2))))
    pts.sort()
    return pts


def panel(ax, arms, data, title):
    for s in arms:
        d = data.get(s, [])
        if not d:
            continue
        lw = 2.0 if s == "baseline" else 1.3
        ax.plot(
            [x / 60.0 for x, _ in d],  # minutes
            [v for _, v in d],
            color=COLORS[s],
            lw=lw,
            marker="o",
            ms=3,
            label=LABELS[s],
        )
    ax.set_title(title)
    ax.set_xlabel("relative wall-clock time (minutes since start)")
    ax.set_ylabel("eval loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def main():
    data = {s: load_arm(s) for s in LABELS}
    present = [s for s in LABELS if data.get(s)]
    if not present:
        print("No runs/<arm>/worker0.log found — nothing to plot.")
        return

    print(f"{'experiment':<30}{'eval pts':>10}{'wall (min)':>12}{'final eval':>12}")
    for s in present:
        d = data[s]
        wall = d[-1][0] / 60.0 if d else float("nan")
        fe = d[-1][1] if d else float("nan")
        print(f"{LABELS[s]:<30}{len(d):>10}{wall:>12.1f}{fe:>12.4f}")

    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: the full headline feature set on a wall-clock axis.
    panel(axes[0], present, data, "Eval loss vs wall-clock — all arms")

    # Right: streaming-focused — the comm/compute-overlap crossover. Only the
    # streaming arms + baseline, where the wall-clock story is the point.
    stream_present = [s for s in STREAMING_SET if data.get(s)]
    panel(axes[1], stream_present, data, "Streaming — comm/compute overlap")

    fig.suptitle(
        "DiLoCo feature comparison — wall-clock — small Llama (34.4M), 4 workers, H=100",
        fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(ASSETS, "walltime_comparison.png")
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
