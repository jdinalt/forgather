#!/usr/bin/env python3
"""Compare the DDPx2 baseline vs the 2-worker DiLoCo run on the same token
budget. Parses the baseline TensorBoard scalars and the DiLoCo server's
aggregate-stats JSONL, prints a summary, and writes comparison plots + a CSV.

Run from examples/tiny_experiments/diloco/ after both runs complete.
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project dir
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BASE_TB = os.path.join(HERE, "output_models/small/runs/baseline_2026-06-01T07-57-19")
DILOCO_JSONL = os.path.join(
    REPO,
    "models/small_llama/runs/1780300721133383415_hal9000/diloco_server_stats.jsonl",
)
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)

# Tokens-per-step (for converting baseline step -> tokens). From the baseline's
# own TrainOutput: 520,302,316 tokens / 8030 steps.
BASE_TOK_PER_STEP = 520302316 / 8030


def tb_scalar(ea, tag):
    return [(e.step, e.value) for e in ea.Scalars(tag)]


# ---- baseline (TensorBoard) ----
ea = EventAccumulator(BASE_TB)
ea.Reload()
b_train = tb_scalar(ea, "train-loss")
b_eval = tb_scalar(ea, "eval-loss")
b_tps = tb_scalar(ea, "tok-per-sec")
b_gn = tb_scalar(ea, "grad-norm")
b_train_tok = [(s * BASE_TOK_PER_STEP, v) for s, v in b_train]
b_eval_tok = [(s * BASE_TOK_PER_STEP, v) for s, v in b_eval]
b_tps_tok = [(s * BASE_TOK_PER_STEP, v) for s, v in b_tps]
b_gn_tok = [(s * BASE_TOK_PER_STEP, v) for s, v in b_gn]

# ---- DiLoCo (server aggregate-stats JSONL) ----
rows = [json.loads(l) for l in open(DILOCO_JSONL)]
d_train = [
    (r["total_tokens"], r["train_loss"])
    for r in rows
    if r.get("train_loss") is not None
]
d_eval = [
    (r["total_tokens"], r["eval_loss"]) for r in rows if r.get("eval_loss") is not None
]
d_tps = [(r["total_tokens"], r["tok_per_sec"]) for r in rows if r.get("tok_per_sec")]
d_gn = [
    (r["total_tokens"], r["grad_norm"]) for r in rows if r.get("grad_norm") is not None
]


def last(xs):
    return xs[-1][1] if xs else float("nan")


print("================ SUMMARY ================")
print(f"{'metric':<22}{'baseline (DDPx2)':>20}{'DiLoCo (2 workers)':>22}")
print(f"{'final train loss':<22}{last(b_train):>20.4f}{last(d_train):>22.4f}")
print(f"{'final eval loss':<22}{last(b_eval):>20.4f}{last(d_eval):>22.4f}")
print(
    f"{'best eval loss':<22}{min(v for _,v in b_eval):>20.4f}{min(v for _,v in d_eval):>22.4f}"
)
print(
    f"{'total tokens (M)':<22}{b_train_tok[-1][0]/1e6:>20.1f}{d_train[-1][0]/1e6:>22.1f}"
)
print(
    f"{'avg tok/s':<22}{sum(v for _,v in b_tps)/len(b_tps):>20.0f}{sum(v for _,v in d_tps)/len(d_tps):>22.0f}"
)
print("=========================================")

# ---- plots ----
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
BLUE, ORANGE = "#1f6fb2", "#d9772b"

# 1) loss curves (train + eval) vs tokens
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot(
    [t / 1e6 for t, _ in b_train_tok],
    [v for _, v in b_train_tok],
    color=BLUE,
    lw=1.2,
    label="baseline (DDPx2)",
)
ax.plot(
    [t / 1e6 for t, _ in d_train],
    [v for _, v in d_train],
    color=ORANGE,
    lw=1.4,
    label="DiLoCo (2 workers)",
)
ax.set_title("Train loss")
ax.set_xlabel("tokens (millions)")
ax.set_ylabel("loss")
ax.legend()
ax.grid(alpha=0.25)
ax = axes[1]
ax.plot(
    [t / 1e6 for t, _ in b_eval_tok],
    [v for _, v in b_eval_tok],
    color=BLUE,
    marker="o",
    ms=3,
    lw=1.2,
    label="baseline (DDPx2)",
)
ax.plot(
    [t / 1e6 for t, _ in d_eval],
    [v for _, v in d_eval],
    color=ORANGE,
    marker="s",
    ms=3,
    lw=1.4,
    label="DiLoCo (2 workers)",
)
ax.set_title("Eval loss")
ax.set_xlabel("tokens (millions)")
ax.set_ylabel("loss")
ax.legend()
ax.grid(alpha=0.25)
fig.suptitle(
    "DiLoCo vs DDP baseline — small Llama (34.4M), ~500M tokens", fontweight="bold"
)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "loss_comparison.png"), bbox_inches="tight")

# 2) throughput + grad norm vs tokens
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot(
    [t / 1e6 for t, _ in b_tps_tok],
    [v / 1e3 for _, v in b_tps_tok],
    color=BLUE,
    lw=1,
    alpha=0.8,
    label="baseline (DDPx2, 2 GPU)",
)
ax.plot(
    [t / 1e6 for t, _ in d_tps],
    [v / 1e3 for _, v in d_tps],
    color=ORANGE,
    lw=1.2,
    label="DiLoCo (2 workers, 2 GPU)",
)
ax.set_title("Throughput")
ax.set_xlabel("tokens (millions)")
ax.set_ylabel("tok/s (thousands)")
ax.legend()
ax.grid(alpha=0.25)
ax = axes[1]
ax.plot(
    [t / 1e6 for t, _ in b_gn_tok],
    [v for _, v in b_gn_tok],
    color=BLUE,
    lw=0.9,
    alpha=0.7,
    label="baseline (DDPx2)",
)
ax.plot(
    [t / 1e6 for t, _ in d_gn],
    [v for _, v in d_gn],
    color=ORANGE,
    lw=1.2,
    label="DiLoCo (2 workers)",
)
ax.set_title("Grad norm")
ax.set_xlabel("tokens (millions)")
ax.set_ylabel("grad norm")
ax.legend()
ax.grid(alpha=0.25)
fig.suptitle("DiLoCo vs DDP baseline — throughput & gradient norm", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "throughput_gradnorm.png"), bbox_inches="tight")

# 3) parsed curves CSV (reproducibility; small)
series = [
    ("baseline", "train_loss", b_train_tok),
    ("baseline", "eval_loss", b_eval_tok),
    ("baseline", "tok_per_sec", b_tps_tok),
    ("baseline", "grad_norm", b_gn_tok),
    ("diloco", "train_loss", d_train),
    ("diloco", "eval_loss", d_eval),
    ("diloco", "tok_per_sec", d_tps),
    ("diloco", "grad_norm", d_gn),
]
with open(os.path.join(ASSETS, "curves.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["series", "metric", "tokens", "value"])
    for name, metric, data in series:
        for t, v in data:
            w.writerow([name, metric, f"{t:.0f}", f"{v:.6f}"])
print("wrote plots + curves.csv to", ASSETS)
