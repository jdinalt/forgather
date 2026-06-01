#!/usr/bin/env python3
"""Extended DiLoCo sweep (2x budget, ~1B tokens) — analysis + plots.

Compares, all on the same small Llama (34.4M) and the same ~1B-token budget:
  - DiLoCo 2-worker at sync_every = 500 / 100 / 20  (the sync-interval study)
  - DiLoCo 1-worker (sync 500)                       (single-worker local-SGD)
  - DDPx2 baseline, single-GPU baseline              (all-reduce / sequential)

DiLoCo curves come from each server's aggregate-stats JSONL; the baselines from
TensorBoard. Writes two plots (sync-interval, 1-worker-vs-1GPU) + a curves CSV.
Run from examples/tiny_experiments/diloco/.
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ASSETS = os.path.join(HERE, "assets")

# DiLoCo runs: server aggregate-stats JSONL (total_tokens is summed across workers).
DILOCO = {
    "h500": "models/sweep_h500/runs/1780336218778531955_hal9000/diloco_server_stats.jsonl",
    "h100": "models/sweep_h100/runs/1780331852973904649_hal9000/diloco_server_stats.jsonl",
    "h20": "models/sweep_h20/runs/1780331855290846739_hal9000/diloco_server_stats.jsonl",
    "1w": "models/sweep_1w/runs/1780332980517220101_hal9000/diloco_server_stats.jsonl",
}
# Baselines: TB run dir + tokens/step (= total_tokens / final_step from TrainOutput).
BASELINES = {
    "ddp2": (
        "output_models/small/runs/baseline_2026-06-01T17-49-43",
        1041735096 / 16061,
    ),
    "1gpu": (
        "output_models/small/runs/baseline_2026-06-01T18-53-18",
        1041850617 / 32124,
    ),
}

LABELS = {
    "h20": "DiLoCo H=20",
    "h100": "DiLoCo H=100",
    "h500": "DiLoCo H=500",
    "1w": "DiLoCo 1-worker",
    "ddp2": "DDPx2 baseline",
    "1gpu": "1-GPU baseline",
}
COLORS = {
    "h20": "#7b3294",
    "h100": "#d9772b",
    "h500": "#c0392b",
    "1w": "#16a085",
    "ddp2": "#1f6fb2",
    "1gpu": "#2ca25f",
}


def diloco_curve(path, metric):
    rows = [json.loads(l) for l in open(os.path.join(REPO, path))]
    return [(r["total_tokens"], r[metric]) for r in rows if r.get(metric) is not None]


def tb_curve(run_dir, tag, tok_per_step):
    ea = EventAccumulator(os.path.join(HERE, run_dir))
    ea.Reload()
    return [(e.step * tok_per_step, e.value) for e in ea.Scalars(tag)]


# Gather all curves: key -> {"train": [...], "eval": [...]}
curves = {}
for k, p in DILOCO.items():
    curves[k] = {
        "train": diloco_curve(p, "train_loss"),
        "eval": diloco_curve(p, "eval_loss"),
    }
for k, (d, tps) in BASELINES.items():
    curves[k] = {
        "train": tb_curve(d, "train-loss", tps),
        "eval": tb_curve(d, "eval-loss", tps),
    }


def final_eval(k):
    return curves[k]["eval"][-1][1]


print("============== SWEEP @2x (~1B tokens) — final eval loss ==============")
for k in ["h20", "h100", "1w", "ddp2", "1gpu", "h500"]:
    print(f"  {LABELS[k]:<18} {final_eval(k):.4f}")
print("=====================================================================")

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def plot_eval(ax, keys, title):
    for k in keys:
        ev = curves[k]["eval"]
        ax.plot(
            [t / 1e6 for t, _ in ev],
            [v for _, v in ev],
            color=COLORS[k],
            lw=1.4,
            marker="o",
            ms=2.5,
            label=LABELS[k],
        )
    ax.set_title(title)
    ax.set_xlabel("tokens (millions)")
    ax.set_ylabel("eval loss")
    ax.legend()
    ax.grid(alpha=0.25)


# Plot A: sync-interval study (DiLoCo H=500/100/20 vs the two baselines).
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
plot_eval(axes[0], ["h500", "h100", "h20", "ddp2", "1gpu"], "Sync interval — full run")
# zoom on the end-game (last ~40% of tokens)
ax = axes[1]
for k in ["h500", "h100", "h20", "ddp2", "1gpu"]:
    ev = [(t, v) for t, v in curves[k]["eval"] if t >= 0.6e9]
    ax.plot(
        [t / 1e6 for t, _ in ev],
        [v for _, v in ev],
        color=COLORS[k],
        lw=1.6,
        marker="o",
        ms=3,
        label=LABELS[k],
    )
ax.set_title("Sync interval — end-game (≥600M tokens)")
ax.set_xlabel("tokens (millions)")
ax.set_ylabel("eval loss")
ax.legend()
ax.grid(alpha=0.25)
fig.suptitle(
    "DiLoCo sync-interval sweep @ 2x budget — small Llama (34.4M)", fontweight="bold"
)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "sweep_sync_interval.png"), bbox_inches="tight")

# Plot B: single-worker local-SGD generalization (1w DiLoCo vs 1-GPU baseline).
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for k in ["1w", "1gpu"]:
    tr = curves[k]["train"]
    ax.plot(
        [t / 1e6 for t, _ in tr],
        [v for _, v in tr],
        color=COLORS[k],
        lw=1.3,
        label=LABELS[k],
    )
ax.set_title("Train loss")
ax.set_xlabel("tokens (millions)")
ax.set_ylabel("loss")
ax.legend()
ax.grid(alpha=0.25)
plot_eval(axes[1], ["1w", "1gpu"], "Eval loss")
fig.suptitle(
    "Single-worker local-SGD vs single-GPU baseline @ 2x (same GPU, data, tokens)",
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "sweep_1worker.png"), bbox_inches="tight")

# curves CSV
with open(os.path.join(ASSETS, "sweep_curves.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "metric", "tokens", "value"])
    for k in curves:
        for metric in ("train", "eval"):
            for t, v in curves[k][metric]:
                w.writerow([k, metric, f"{t:.0f}", f"{v:.6f}"])
print("wrote sweep_sync_interval.png, sweep_1worker.png, sweep_curves.csv to", ASSETS)
