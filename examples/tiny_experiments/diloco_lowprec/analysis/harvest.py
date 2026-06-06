#!/usr/bin/env python3
"""Parse the DiLoCo low-precision sweep into ``assets/curves.csv`` + a summary.

Each experiment is a 2-worker DiLoCo run. Per-step train loss and periodic
eval loss are emitted by the workers to their TTY logs; throughput / sync
counts are in the DiLoCo server's stats JSONL. This reads one rank-0 worker
log per experiment (both workers train the same global model) plus the server
JSONL, writes the tidy ``assets/curves.csv`` the plot script consumes, and
prints a Markdown summary table for the README.

Usage (from the project directory):
    python analysis/harvest.py --logs-dir <dir-with-{exp}.worker0.log>

The raw worker logs are run artifacts (ephemeral); ``assets/curves.csv`` is the
committed source of truth, mirroring the sibling ``diloco`` project.
"""

import argparse
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")
SERVER_RUNS = os.path.join(os.path.dirname(HERE), os.pardir, os.pardir, "models")

# experiment -> (model dir name for server JSONL, human label)
EXPERIMENTS = [
    ("b0", "small_llama_lp_b0", "B0 baseline (down fp32)"),
    ("e1", "small_llama_lp_e1", "E1 down bf16"),
    ("e2", "small_llama_lp_e2", "E2 down bf16+SR"),
    ("e3", "small_llama_lp_e3", "E3 up bf16+SR"),
    ("e4", "small_llama_lp_e4", "E4 bf16w, down bf16+SR"),
    ("e5", "small_llama_lp_e5", "E5 bf16w, down bf16"),
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
    pat = os.path.join(SERVER_RUNS, model_dir, "runs", "*", "diloco_server_stats.jsonl")
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
    ap.add_argument("--logs-dir", required=True, help="dir holding <exp>.worker0.log")
    args = ap.parse_args()

    os.makedirs(ASSETS, exist_ok=True)
    rows, summary = [], []
    for exp, model_dir, label in EXPERIMENTS:
        log = os.path.join(args.logs_dir, f"{exp}.worker0.log")
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
