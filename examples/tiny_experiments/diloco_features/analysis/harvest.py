#!/usr/bin/env python3
"""Parse the diloco_features comparison sweep into ``assets/curves.csv``.

Each experiment is a 4-worker DiLoCo run at the same token budget and seed,
differing only in the feature under test (streaming / async / DN-buffer / DyLU)
vs the sync baseline. Per-step train loss + grad-norm and periodic eval loss are
in the workers' TTY logs; throughput / sync counts are in the server's stats
JSONL. This reads one rank-0 worker log per experiment (``runs/<exp>/worker0.log``,
captured by ``experiment.sh``) plus the server JSONL, writes the tidy
``assets/curves.csv`` the plot script consumes, and prints a Markdown summary.

Usage (from the project directory, after experiment.sh):
    python analysis/harvest.py

``assets/curves.csv`` is the committed source of truth; the raw run logs under
``runs/`` are ephemeral artifacts (gitignored), mirroring the sibling projects.
"""

import argparse
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")
MODELS = os.path.join(os.path.dirname(HERE), os.pardir, os.pardir, "models")

# series key -> (runs/<dir>/worker0.log, model dir for server JSONL, human label)
# 4-worker comparison. Async runs use phase-jitter (DILOCO_DEBUG_STEP_JITTER) for
# real staleness; DyLU uses a per-worker speed spread. The DN-buffer sweep points
# (N=4/8/16) feed analysis/dn_sweep.py. `baseline_2w` is the preserved 2-worker
# baseline that backs the gRPC-vs-h100 transport check (analysis/verify_baseline.py).
EXPERIMENTS = [
    ("baseline", "baseline", "small_llama_feat_baseline", "Baseline (sync, H=100)"),
    ("streaming", "streaming", "small_llama_feat_streaming", "Streaming (2 fragments)"),
    ("async", "async", "small_llama_feat_async", "Async (no DN)"),
    ("async_dn_b4", "async_dn_b4", "small_llama_feat_async_dn_b4", "Async + DN (N=4)"),
    ("async_dn_b8", "async_dn_b8", "small_llama_feat_async_dn_b8", "Async + DN (N=8)"),
    (
        "async_dn_b16",
        "async_dn_b16",
        "small_llama_feat_async_dn_b16",
        "Async + DN (N=16)",
    ),
    ("dylu", "dylu", "small_llama_feat_dylu", "Async + DN + DyLU"),
    (
        "baseline_2w",
        "baseline_2w",
        "small_llama_feat_baseline",
        "Baseline 2w (transport ref)",
    ),
]

_NUM = r"[-+]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?"


def parse_worker_log(path):
    """Return dicts {step: train_loss}, {step: eval_loss}, {step: grad_norm}."""
    train, eval_, grad = {}, {}, {}
    if not os.path.isfile(path):
        return train, eval_, grad
    with open(path, errors="replace") as f:
        for line in f:
            if "eval-loss:" in line:
                m = re.search(
                    r"\s(\d[\d,]*)\s+[\d.eE+-]+\s+eval-loss:\s*(" + _NUM + ")", line
                )
                if m:
                    step = int(m.group(1).replace(",", ""))
                    eval_[step] = float(m.group(2).replace(",", ""))
                continue
            # train data row: date time step epoch loss grad lr ...
            m = re.match(
                r"^\d{4}-\d{2}-\d{2}\s+[\d:]+\s+(\d[\d,]*)\s+([\d.eE+-]+)\s+"
                r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s",
                line,
            )
            if m:
                step = int(m.group(1).replace(",", ""))
                try:
                    train[step] = float(m.group(3))
                    grad[step] = float(m.group(4))
                except ValueError:
                    pass
    return train, eval_, grad


def parse_server_jsonl(model_dir):
    """Return (avg_tok_s, final_tokens, sync_rounds) from the server stats."""
    pat = os.path.join(MODELS, model_dir, "runs", "*", "diloco_server_stats.jsonl")
    files = sorted(glob.glob(pat))
    if not files:
        return float("nan"), 0, 0
    toks, last = [], {}
    with open(files[-1], errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("tok_per_sec"):
                toks.append(d["tok_per_sec"])
            last = d
    avg = sum(toks) / len(toks) if toks else float("nan")
    return avg, last.get("total_tokens", 0), last.get("sync_round", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-dir",
        default=os.path.join(HERE, "runs"),
        help="dir holding <exp>/worker0.log (default: ./runs)",
    )
    args = ap.parse_args()

    os.makedirs(ASSETS, exist_ok=True)
    rows, summary = [], []
    for exp, runs_subdir, model_dir, label in EXPERIMENTS:
        log = os.path.join(args.runs_dir, runs_subdir, "worker0.log")
        train, eval_, grad = parse_worker_log(log)
        if not train and not eval_:
            print(f"  skip {exp}: no data at {log}")
            continue
        for step, v in sorted(train.items()):
            rows.append([exp, "train_loss", step, f"{v:.6f}"])
        for step, v in sorted(eval_.items()):
            rows.append([exp, "eval_loss", step, f"{v:.6f}"])
        for step, v in sorted(grad.items()):
            rows.append([exp, "grad_norm", step, f"{v:.6f}"])
        avg_tok_s, total_tok, syncs = parse_server_jsonl(model_dir)
        ft = train[max(train)] if train else float("nan")
        fe = eval_[max(eval_)] if eval_ else float("nan")
        be = min(eval_.values()) if eval_ else float("nan")
        summary.append((exp, label, ft, fe, be, avg_tok_s, total_tok, syncs))

    with open(os.path.join(ASSETS, "curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "metric", "step", "value"])
        w.writerows(rows)

    print(f"\nwrote {len(rows)} rows to assets/curves.csv\n")
    print("| Exp | final train | final eval | best eval | avg tok/s | tokens | syncs |")
    print("|---|---|---|---|---|---|---|")
    for exp, label, ft, fe, be, ts, tok, syncs in summary:
        print(
            f"| {label} | {ft:.4f} | {fe:.4f} | {be:.4f} | "
            f"{ts/1e3:.0f}K | {tok/1e6:.0f}M | {syncs} |"
        )


if __name__ == "__main__":
    main()
