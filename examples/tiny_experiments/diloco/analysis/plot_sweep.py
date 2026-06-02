#!/usr/bin/env python3
"""Extended DiLoCo sweep (2x budget, ~1B tokens) — analysis + plots.

Compares, all on the same small Llama (34.4M) and the same ~1B-token budget,
the same seed init, and the same constant-LR (warmup-stable, annealing=0)
schedule:
  - DiLoCo 2-worker at sync_every = 500 / 100 / 20    (the sync-interval study)
  - DiLoCo 1-worker at sync_every = 500 / 100 / 20     (single-worker local-SGD)
  - DDPx2 baseline, single-GPU baseline                (all-reduce / sequential)
  - PyTorch PostLocalSGD at period = 100 / 20          (pure periodic averaging,
                                                        no outer optimizer)

All workers see identical data (work-unit dispatch for DiLoCo; dispatched,
non-sharded batches for the DDP/PostLocalSGD runs), so the only variable across
a matched-cadence comparison is the synchronization mechanism.

DiLoCo curves come from each server's aggregate-stats JSONL; the DDP-family runs
(baselines + PostLocalSGD) from TensorBoard. Writes three plots
(sync-interval, single-worker, PostLocalSGD-vs-DiLoCo) + a curves CSV.
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
    "1w_h100": "models/sweep_1w_h100/runs/1780350726702501066_hal9000/diloco_server_stats.jsonl",
    "1w_h20": "models/sweep_1w_h20/runs/1780350724335964712_hal9000/diloco_server_stats.jsonl",
}
# TensorBoard runs: (run dir, tokens/step). tokens/step = total_tokens / final_step
# from each run's TrainOutput. DDP-family runs all share the 2x budget (16061 steps).
TB = {
    "ddp2": (
        "output_models/small/runs/baseline_2026-06-01T21-50-57",
        1041850617 / 16061,
    ),
    "1gpu": (
        "output_models/small/runs/baseline_2026-06-01T18-53-18",
        1041850617 / 32124,
    ),
    "pl_h100": (
        "output_models/small/runs/postlocalsgd_2026-06-02T00-10-49",
        1041850617 / 16061,
    ),
    "pl_h20": (
        "output_models/small/runs/postlocalsgd_2026-06-02T00-37-30",
        1041850617 / 16061,
    ),
}

LABELS = {
    "h20": "DiLoCo 2w H=20",
    "h100": "DiLoCo 2w H=100",
    "h500": "DiLoCo 2w H=500",
    "1w": "DiLoCo 1w H=500",
    "1w_h100": "DiLoCo 1w H=100",
    "1w_h20": "DiLoCo 1w H=20",
    "ddp2": "DDPx2 baseline",
    "1gpu": "1-GPU baseline",
    "pl_h100": "PostLocalSGD H=100",
    "pl_h20": "PostLocalSGD H=20",
}
COLORS = {
    "h20": "#7b3294",
    "h100": "#d9772b",
    "h500": "#c0392b",
    "1w": "#c0392b",
    "1w_h100": "#d9772b",
    "1w_h20": "#7b3294",
    "ddp2": "#1f6fb2",
    "1gpu": "#2ca25f",
    "pl_h100": "#e377c2",
    "pl_h20": "#8c564b",
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
for k, (d, tps) in TB.items():
    curves[k] = {
        "train": tb_curve(d, "train-loss", tps),
        "eval": tb_curve(d, "eval-loss", tps),
    }


def final_eval(k):
    return curves[k]["eval"][-1][1]


print("============== SWEEP @2x (~1B tokens) — final eval loss ==============")
order = [
    "1w_h100",
    "1w_h20",
    "h20",
    "h100",
    "1w",
    "1gpu",
    "ddp2",
    "pl_h20",
    "pl_h100",
    "h500",
]
for k in order:
    print(f"  {LABELS[k]:<20} {final_eval(k):.4f}")
print("=====================================================================")

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def plot_eval(ax, keys, title, tok_min=0.0):
    for k in keys:
        ev = [(t, v) for t, v in curves[k]["eval"] if t >= tok_min]
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


# Plot A: sync-interval study (DiLoCo 2w H=500/100/20 vs the two baselines).
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
plot_eval(axes[0], ["h500", "h100", "h20", "ddp2", "1gpu"], "Sync interval — full run")
plot_eval(
    axes[1],
    ["h500", "h100", "h20", "ddp2", "1gpu"],
    "Sync interval — end-game (≥600M tokens)",
    tok_min=0.6e9,
)
fig.suptitle(
    "DiLoCo sync-interval sweep @ 2x budget — small Llama (34.4M)", fontweight="bold"
)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "sweep_sync_interval.png"), bbox_inches="tight")

# Plot B: single-worker local-SGD — sync sensitivity + vs the single-GPU baseline.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
for k in ["1w_h20", "1w_h100", "1w"]:
    tr = curves[k]["train"]
    ax.plot(
        [t / 1e6 for t, _ in tr],
        [v for _, v in tr],
        color=COLORS[k],
        lw=1.3,
        label=LABELS[k],
    )
tr = curves["1gpu"]["train"]
ax.plot(
    [t / 1e6 for t, _ in tr],
    [v for _, v in tr],
    color=COLORS["1gpu"],
    lw=1.3,
    ls="--",
    label=LABELS["1gpu"],
)
ax.set_title("Train loss")
ax.set_xlabel("tokens (millions)")
ax.set_ylabel("loss")
ax.legend()
ax.grid(alpha=0.25)
plot_eval(axes[1], ["1w_h20", "1w_h100", "1w", "1gpu"], "Eval loss")
fig.suptitle(
    "Single-worker local-SGD @ 2x — sync sensitivity vs single-GPU baseline "
    "(same GPU, data, tokens)",
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "sweep_1worker.png"), bbox_inches="tight")

# Plot C: PostLocalSGD (pure averaging) vs DiLoCo (outer Nesterov) at matched
# cadence. Same DDP setup; the only difference is the synchronization mechanism.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
plot_eval(
    axes[0],
    ["h20", "pl_h20", "h100", "pl_h100", "ddp2"],
    "Matched cadence — full run",
)
plot_eval(
    axes[1],
    ["h20", "pl_h20", "h100", "pl_h100", "ddp2"],
    "Matched cadence — end-game (≥600M tokens)",
    tok_min=0.6e9,
)
fig.suptitle(
    "Outer optimizer vs pure averaging @ 2x — DiLoCo H vs PostLocalSGD period",
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "sweep_postlocalsgd.png"), bbox_inches="tight")

# curves CSV
with open(os.path.join(ASSETS, "sweep_curves.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "metric", "tokens", "value"])
    for k in curves:
        for metric in ("train", "eval"):
            for t, v in curves[k][metric]:
                w.writerow([k, metric, f"{t:.0f}", f"{v:.6f}"])
print(
    "wrote sweep_sync_interval.png, sweep_1worker.png, sweep_postlocalsgd.png, "
    "sweep_curves.csv to",
    ASSETS,
)
