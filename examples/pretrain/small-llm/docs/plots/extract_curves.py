"""Extract the experiment loss/LR/grad curves into a committed data file.

The trained runs live under ``output_models/`` (gitignored), so the data
backing the README plots — weeks of GPU time — only exists on the machine that
produced it. This script reads each run's ``trainer_logs.json`` and writes a
compact, source-controlled ``curves.csv`` (long format) holding the curves the
plots need. ``render.py`` and ``render_diloco.py`` then read ``curves.csv``, so
the plots regenerate from a clean checkout with no ``output_models/`` present.

Run from the project directory, on a machine that still has the run logs:

    python docs/plots/extract_curves.py

Re-run it whenever a new run should be folded into the committed data. The
training-derived series (train_loss / learning_rate / grad_norm) are
downsampled to at most MAX_POINTS per run (the plots smooth them heavily, so
this is visually lossless); eval_loss is kept at full resolution (sparse, and
the headline metric).
"""

import csv
from glob import glob
from pathlib import Path

from forgather.ml.analysis import TrainingLog

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "curves.csv"
MAX_POINTS = 1200

# Canonical run -> log path. These are the runs the README plots reference
# (render.py's 10x + 1x sets; the DiLoCo workers are discovered below).
RUNS = {
    "ten_chinchilla": "output_models/ten_chinchilla/runs/log_2026-04-02T00-02-58/trainer_logs.json",
    "long_cooldown": "output_models/long_cooldown/runs/log_2026-04-06T07-14-55/trainer_logs.json",
    "tiny_x_small_lm": "output_models/tiny_x_small_lm/runs/log_2026-04-19T22-50-21/trainer_logs.json",
    "wds": "output_models/wds/runs/log_2026-04-22T04-05-02/trainer_logs.json",
    "final": "output_models/final/runs/log_2026-04-26T11-00-31/trainer_logs.json",
    "bf16": "output_models/bf16/runs/log_2026-04-05T00-51-49/trainer_logs.json",
    "bf16_adafactor": "output_models/bf16_adafactor/runs/log_2026-04-04T20-52-11/trainer_logs.json",
    "high_lr": "output_models/high_lr/runs/log_2026-04-08T14-47-03/trainer_logs.json",
    "canon": "output_models/canon/runs/log_2026-04-05T20-04-51/trainer_logs.json",
    "muon": "output_models/muon/runs/log_2026-04-28T08-22-52/trainer_logs.json",
    "deepone": "output_models/deepone/runs/log_2026-04-24T09-25-27/trainer_logs.json",
}

# DiLoCo workers: their run dirs are timestamped, so discover the latest.
for w in ("w0", "w1", "w2", "w3"):
    hits = sorted(
        glob(str(ROOT / f"output_models/diloco_{w}/runs/*/trainer_logs.json"))
    )
    if hits:
        RUNS[f"diloco_{w}"] = str(Path(hits[-1]).relative_to(ROOT))


def downsample(points, max_points):
    """Keep at most max_points by striding; always keep the last point."""
    if len(points) <= max_points:
        return points
    stride = (len(points) + max_points - 1) // max_points
    out = points[::stride]
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


# The 1x plots use the first 1x-Chinchilla slice of the 10x ``ten_chinchilla``
# run as the ``default`` baseline. Emit that slice as its own series, downsampled
# *within the slice*, so it stays densely sampled there (~1200 pts) rather than
# the ~100 a global downsample of the 400K-step run leaves below 37K.
SLICE_RUN = "ten_chinchilla"
SLICE_NAME = "default"
SLICE_MAX_STEP = 37000


def emit(rows, run, train, evals, step_max=None):
    for metric, key in (
        ("train_loss", "loss"),
        ("learning_rate", "learning_rate"),
        ("grad_norm", "grad_norm"),
    ):
        pts = [
            (r["global_step"], r[key])
            for r in train
            if r.get(key) is not None
            and r.get("global_step") is not None
            and (step_max is None or r["global_step"] <= step_max)
        ]
        for step, val in downsample(pts, MAX_POINTS):
            rows.append((run, metric, step, val))
    n_eval = 0
    for r in evals:
        s = r.get("global_step")
        if (
            r.get("eval_loss") is not None
            and s is not None
            and (step_max is None or s <= step_max)
        ):
            rows.append((run, "eval_loss", s, r["eval_loss"]))
            n_eval += 1
    return n_eval


def emit_worker_mean(rows, name, dir_glob):
    """Per-step mean across N coherent DiLoCo workers (auto-named, so globbed).

    The workers train in lockstep and stay near-identical, so a single mean
    series represents the run; emitted as one ``name`` run.
    """
    from collections import defaultdict

    paths = sorted(p for p in glob(str(ROOT / dir_glob)) if "_master" not in p)
    by_worker = defaultdict(list)
    for p in paths:
        by_worker[Path(p).parents[2].name].append(p)
    logs = [TrainingLog.from_file(sorted(v)[-1]) for v in by_worker.values()]
    if not logs:
        print(f"  {name}: no workers found ({dir_glob})")
        return 0
    for metric, key, kind in (
        ("train_loss", "loss", "train"),
        ("learning_rate", "learning_rate", "train"),
        ("grad_norm", "grad_norm", "train"),
        ("eval_loss", "eval_loss", "eval"),
    ):
        by_step = defaultdict(list)
        for log in logs:
            recs = (
                log.get_training_records()
                if kind == "train"
                else log.get_eval_records()
            )
            for r in recs:
                s, v = r.get("global_step"), r.get(key)
                if s is not None and v is not None:
                    by_step[s].append(v)
        pts = [(s, sum(vs) / len(vs)) for s, vs in sorted(by_step.items())]
        pts = pts if metric == "eval_loss" else downsample(pts, MAX_POINTS)
        for s, v in pts:
            rows.append((name, metric, s, v))
    n_eval = sum(1 for r in rows if r[0] == name and r[1] == "eval_loss")
    print(f"  {name}: {len(logs)} workers averaged, {n_eval} eval")
    return n_eval


def main():
    rows = []  # (run, metric, step, value)
    for run, rel in RUNS.items():
        path = ROOT / rel
        if not path.exists():
            print(f"  skip {run}: {rel} not found")
            continue
        log = TrainingLog.from_file(str(path))
        train = log.get_training_records()
        evals = log.get_eval_records()
        n_eval = emit(rows, run, train, evals)
        print(f"  {run}: {len(train)} train -> downsampled, {n_eval} eval")
        if run == SLICE_RUN:
            n = emit(rows, SLICE_NAME, train, evals, step_max=SLICE_MAX_STEP)
            print(
                f"  {SLICE_NAME}: 1x slice of {SLICE_RUN} (step <= {SLICE_MAX_STEP}), {n} eval"
            )

    # 4-worker DiLoCo 11x run (auto-named workers -> per-step mean).
    emit_worker_mean(
        rows,
        "diloco11x",
        "output_models/diloco_ten_chinchilla_*/runs/*/trainer_logs.json",
    )

    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    with open(OUT, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["run", "metric", "step", "value"])
        for run, metric, step, val in rows:
            wr.writerow([run, metric, step, f"{val:.6g}"])
    print(f"wrote {len(rows)} rows for {len({r[0] for r in rows})} runs -> {OUT}")


if __name__ == "__main__":
    main()
