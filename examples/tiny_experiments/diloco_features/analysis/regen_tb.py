#!/usr/bin/env python3
"""Regenerate per-worker TensorBoard event files from captured worker TTY logs.

Use when the live per-worker TB event files are gone (e.g. the worker output
tree was cleaned) but the harvested worker logs under ``runs/<arm>/worker*.log``
survive. Reconstructs ``train/loss``, ``train/grad_norm`` and ``eval/loss`` at
the workers' full per-log-step resolution — much finer than the server's
per-sync-round aggregate — and stamps each point with the row's real wall-clock
time, so TensorBoard's WALL / RELATIVE time axis works (the streaming
loss-vs-walltime comparison).

Usage:
    python analysis/regen_tb.py [runs_dir]      # default: runs
Writes events to ``runs/<arm>/tb/<workerN>/`` — point TensorBoard at ``runs``.
"""

import datetime
import os
import re
import sys

from torch.utils.tensorboard import SummaryWriter

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
# date time step epoch loss grad lr ...  (mirrors harvest.py's train-row regex)
_TRAIN = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+[\d:]+\s+(\d[\d,]*)\s+"
    + _NUM
    + r"\s+("
    + _NUM
    + r")\s+("
    + _NUM
    + r")\s"
)
_EVAL = re.compile(r"\s(\d[\d,]*)\s+" + _NUM + r"\s+eval-loss:\s*(" + _NUM + r")")


def _walltime(line):
    m = _TS.match(line)
    if not m:
        return None
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()


def regen_one(log_path, out_dir):
    """Emit a TB event file from one worker log; return the point count."""
    writer = SummaryWriter(out_dir)
    n = 0
    with open(log_path, errors="replace") as f:
        for line in f:
            wt = _walltime(line)
            if wt is None:
                continue
            if "eval-loss:" in line:
                m = _EVAL.search(line)
                if m:
                    writer.add_scalar(
                        "eval/loss",
                        float(m.group(2)),
                        int(m.group(1).replace(",", "")),
                        walltime=wt,
                    )
                    n += 1
                continue
            m = _TRAIN.match(line)
            if m:
                step = int(m.group(1).replace(",", ""))
                try:
                    writer.add_scalar(
                        "train/loss", float(m.group(2)), step, walltime=wt
                    )
                    writer.add_scalar(
                        "train/grad_norm", float(m.group(3)), step, walltime=wt
                    )
                    n += 1
                except ValueError:
                    pass
    writer.close()
    return n


def main(runs_dir="runs"):
    for arm in sorted(os.listdir(runs_dir)):
        arm_dir = os.path.join(runs_dir, arm)
        if not os.path.isdir(arm_dir):
            continue
        for wl in sorted(
            f for f in os.listdir(arm_dir) if re.fullmatch(r"worker\d+\.log", f)
        ):
            out = os.path.join(arm_dir, "tb", wl[:-4])
            n = regen_one(os.path.join(arm_dir, wl), out)
            print(f"  {arm}/{wl}: {n} points -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
