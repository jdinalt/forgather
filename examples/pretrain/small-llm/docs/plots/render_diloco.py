"""Render the DiLoCo-vs-DDPx4 comparison plot for examples/pretrain/small-llm.

Overlays the 4-worker shared-memory DiLoCo run (mean across workers, with a
min/max band) against the DDPx4 ``default`` baseline (the first 1x-Chinchilla
slice of ``ten_chinchilla``). Same model (medium.yaml 162M), dataset, and
per-rank step schedule; the only variable is 4 workers syncing every H=20
steps via outer SGD vs DDP all-reduce every step.

Run from the project directory:

    python docs/plots/render_diloco.py

Writes docs/plots/diloco_1x_comparison.png and prints a summary table.
"""

from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forgather.ml.analysis import TrainingLog
from forgather.ml.analysis.plotting import smooth_values

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "plots"

BASELINE = "output_models/ten_chinchilla/runs/log_2026-04-02T00-02-58/trainer_logs.json"
X_MAX = 36460  # 1x-Chinchilla slice of the 10x baseline
LINEWIDTH = 1.2


def latest_worker_log(worker):
    runs = sorted(
        glob(str(ROOT / f"output_models/diloco_{worker}/runs/*/trainer_logs.json"))
    )
    return runs[-1] if runs else None


def series(log, kind):
    if kind == "train":
        recs = log.get_training_records()
        return [
            (r["global_step"], r["loss"]) for r in recs if r.get("loss") is not None
        ]
    recs = log.get_eval_records()
    return [
        (r["global_step"], r["eval_loss"])
        for r in recs
        if r.get("eval_loss") is not None
    ]


def clip(points, x_max):
    return [(s, v) for (s, v) in points if s <= x_max]


def worker_mean(worker_points):
    """Mean across workers at each common step (workers eval/log in lockstep)."""
    from collections import defaultdict

    by_step = defaultdict(list)
    for pts in worker_points:
        for s, v in pts:
            by_step[s].append(v)
    steps = sorted(by_step)
    mean = [float(np.mean(by_step[s])) for s in steps]
    lo = [float(np.min(by_step[s])) for s in steps]
    hi = [float(np.max(by_step[s])) for s in steps]
    return steps, mean, lo, hi


def main():
    workers = ["w0", "w1", "w2", "w3"]
    wlogs = {w: latest_worker_log(w) for w in workers}
    missing = [w for w, p in wlogs.items() if p is None]
    if missing:
        raise SystemExit(f"missing worker logs: {missing}")
    dlogs = {w: TrainingLog.from_file(p) for w, p in wlogs.items()}
    base = TrainingLog.from_file(str(ROOT / BASELINE))

    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(16, 6))

    for kind, ax, smooth_w in (("train", ax_t, 50), ("eval", ax_e, 1)):
        # DiLoCo workers -> mean + band
        wpts = [clip(series(dlogs[w], kind), X_MAX) for w in workers]
        steps, mean, lo, hi = worker_mean(wpts)
        if kind == "train":
            mean = smooth_values(mean, smooth_w)
            lo = smooth_values(lo, smooth_w)
            hi = smooth_values(hi, smooth_w)
        ax.fill_between(steps, lo, hi, color="#d62728", alpha=0.18, linewidth=0)
        ax.plot(
            steps,
            mean,
            color="#d62728",
            linewidth=LINEWIDTH,
            label="DiLoCo 4-worker (H=20, shm)",
        )

        # DDPx4 baseline
        bpts = clip(series(base, kind), X_MAX)
        bs = [s for s, _ in bpts]
        bv = [v for _, v in bpts]
        if kind == "train":
            bv = smooth_values(bv, smooth_w)
        ax.plot(
            bs,
            bv,
            color="#1f77b4",
            linewidth=LINEWIDTH,
            label="DDPx4 baseline (default)",
        )

        ax.set_title(
            f"{'Train' if kind == 'train' else 'Eval'} Loss vs step (1x Chinchilla)"
        )
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # focus eval y-axis on the informative tail
    ax_e.set_ylim(2.5, 3.6)
    fig.tight_layout()
    out = OUT / "diloco_1x_comparison.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")

    # summary
    def best_last(log):
        ev = clip(series(log, "eval"), X_MAX)
        return (min(v for _, v in ev), ev[-1]) if ev else (None, None)

    print("\n=== eval-loss summary (1x slice, step <= %d) ===" % X_MAX)
    for w in workers:
        b, l = best_last(dlogs[w])
        print(
            f"  DiLoCo {w}: best={b:.4f} last={l[1]:.4f}@{l[0]}"
            if b
            else f"  DiLoCo {w}: (no eval)"
        )
    # group mean of per-worker best/last
    bests = [best_last(dlogs[w])[0] for w in workers]
    lasts = [best_last(dlogs[w])[1][1] for w in workers]
    print(
        f"  DiLoCo group: best(mean)={np.mean(bests):.4f}  last(mean)={np.mean(lasts):.4f}"
    )
    bb, bl = best_last(base)
    print(f"  DDPx4 baseline: best={bb:.4f} last={bl[1]:.4f}@{bl[0]}")
    print(f"  delta (DiLoCo best mean - baseline best): {np.mean(bests) - bb:+.4f}")


if __name__ == "__main__":
    main()
