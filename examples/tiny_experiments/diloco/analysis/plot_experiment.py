#!/usr/bin/env python3
"""3-way comparison at the 1x (~500M-token) budget: DDPx2 baseline, single-GPU
baseline, and the 2-worker DiLoCo run.

The DiLoCo curve is read from the committed ``assets/curves.csv`` (its original
run dir is ephemeral); the DDPx2 and single-GPU baselines are parsed from their
TensorBoard runs. All three runs see identical data and use the same constant LR
(annealing=0). Writes the comparison plots and rewrites ``curves.csv`` with all
three series.

Run from examples/tiny_experiments/diloco/.
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project dir
ASSETS = os.path.join(HERE, "assets")
# Single-GPU baseline (1x, 500M, 16062 steps). TrainOutput: 520,462,475 tok / 16062 steps.
BASE1_TB = os.path.join(HERE, "output_models/small/runs/baseline_2026-06-01T16-02-47")
BASE1_TOK_PER_STEP = 520462475 / 16062
# DDPx2 baseline (1x, 500M, 8030 steps). TrainOutput: 520,462,475 tok / 8030 steps.
BASE_DDP2_TB = os.path.join(
    HERE, "output_models/small/runs/baseline_2026-06-02T00-10-55"
)
BASE_DDP2_TOK_PER_STEP = 520462475 / 8030


def tb_scalar(ea, tag):
    return [(e.step, e.value) for e in ea.Scalars(tag)]


# ---- DiLoCo series from committed curves.csv (its run dir is ephemeral) ----
# The baselines are (re)parsed from TensorBoard below, so skip them on load.
existing = defaultdict(list)  # (series, metric) -> [(tokens, value)]
with open(os.path.join(ASSETS, "curves.csv")) as f:
    for row in csv.DictReader(f):
        if row["series"] in ("baseline", "baseline_1gpu"):  # parsed from TB instead
            continue
        existing[(row["series"], row["metric"])].append(
            (float(row["tokens"]), float(row["value"]))
        )

TB_TAGS = [
    ("train-loss", "train_loss"),
    ("eval-loss", "eval_loss"),
    ("tok-per-sec", "tok_per_sec"),
    ("grad-norm", "grad_norm"),
]


def tb_series(run_dir, tok_per_step):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    return {
        metric: [(s * tok_per_step, v) for s, v in tb_scalar(ea, tag)]
        for tag, metric in TB_TAGS
    }


# ---- baselines (TensorBoard): DDPx2 and single-GPU ----
bddp2 = tb_series(BASE_DDP2_TB, BASE_DDP2_TOK_PER_STEP)
b1 = tb_series(BASE1_TB, BASE1_TOK_PER_STEP)

# Series in plot order, with display labels + colors.
COLORS = {"baseline": "#1f6fb2", "baseline_1gpu": "#2ca25f", "diloco": "#d9772b"}
LABELS = {
    "baseline": "DDPx2 baseline",
    "baseline_1gpu": "1-GPU baseline",
    "diloco": "DiLoCo (2 workers)",
}


def curve(series, metric):
    if series == "baseline":
        return bddp2[metric]
    if series == "baseline_1gpu":
        return b1[metric]
    return existing[(series, metric)]


def last(xs):
    return xs[-1][1] if xs else float("nan")


print("==================== SUMMARY (1x, ~500M tokens) ====================")
hdr = f"{'metric':<20}" + "".join(f"{LABELS[s]:>22}" for s in LABELS)
print(hdr)
for label, metric, fn in [
    ("final train loss", "train_loss", last),
    ("final eval loss", "eval_loss", last),
    ("best eval loss", "eval_loss", lambda xs: min(v for _, v in xs)),
]:
    print(f"{label:<20}" + "".join(f"{fn(curve(s, metric)):>22.4f}" for s in LABELS))
print("====================================================================")

# ---- plots ----
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def plot_panel(ax, metric, title, ylabel, scale=1.0, markers=False):
    for s in LABELS:
        data = curve(s, metric)
        if not data:
            continue
        ax.plot(
            [t / 1e6 for t, _ in data],
            [v / scale for _, v in data],
            color=COLORS[s],
            lw=1.3,
            marker=("o" if markers else None),
            ms=3,
            label=LABELS[s],
        )
    ax.set_title(title)
    ax.set_xlabel("tokens (millions)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.25)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
plot_panel(axes[0], "train_loss", "Train loss", "loss")
plot_panel(axes[1], "eval_loss", "Eval loss", "loss", markers=True)
fig.suptitle(
    "DiLoCo vs DDPx2 vs single-GPU — small Llama (34.4M), ~500M tokens",
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "loss_comparison.png"), bbox_inches="tight")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
plot_panel(axes[0], "tok_per_sec", "Throughput", "tok/s (thousands)", scale=1e3)
plot_panel(axes[1], "grad_norm", "Grad norm", "grad norm")
fig.suptitle(
    "DiLoCo vs DDPx2 vs single-GPU — throughput & gradient norm", fontweight="bold"
)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "throughput_gradnorm.png"), bbox_inches="tight")

# ---- rewrite curves.csv with all three series ----
with open(os.path.join(ASSETS, "curves.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["series", "metric", "tokens", "value"])
    for s in LABELS:
        for metric in ("train_loss", "eval_loss", "tok_per_sec", "grad_norm"):
            for t, v in curve(s, metric):
                w.writerow([s, metric, f"{t:.0f}", f"{v:.6f}"])
print("wrote plots + curves.csv to", ASSETS)
