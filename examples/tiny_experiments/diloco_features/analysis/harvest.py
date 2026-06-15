#!/usr/bin/env python3
"""Parse the diloco_features run matrix into ``assets/curves.csv`` + a summary.

Each arm is a from-scratch DiLoCo run trained to the same total-token budget and
seed, differing only in the feature under test (streaming / async / DN / grace /
DyLU) vs the sync baseline. Per-step train loss + grad-norm and periodic eval
loss are in the workers' TTY logs; the actual total_tokens, the grace-batch
histogram and the per-worker staleness are in the live ``/status`` snapshot
``experiment.sh`` captures to ``runs/<arm>/status.json``. This reads one rank-0
worker log per arm (``runs/<arm>/worker0.log``) plus the status snapshot, writes
the tidy ``assets/curves.csv`` the plot scripts consume (with a derived
``perplexity`` series), and prints a Markdown summary.

Usage (from the project directory, after experiment.sh):
    python analysis/harvest.py

``assets/curves.csv`` is the committed source of truth; the raw run logs under
``runs/`` are ephemeral artifacts (gitignored), mirroring the sibling projects.
"""

import argparse
import csv
import glob
import json
import math
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")
MODELS = os.path.join(os.path.dirname(HERE), os.pardir, os.pardir, "models")

# series key -> (runs/<dir>/, model dir for server JSONL fallback, human label).
# The run matrix (experiment.sh), all 4-worker; async_dn4/8/16 are the DN-buffer
# depth sweep. Async arms (async_*) use
# phase-jitter (DILOCO_DEBUG_STEP_JITTER) for real staleness; DyLU arms use a
# per-worker speed spread.
EXPERIMENTS = [
    ("baseline", "baseline", "small_llama_feat_baseline", "Baseline (sync, H=100)"),
    (
        "stream_str2",
        "stream_str2",
        "small_llama_feat_stream_str2",
        "Streaming strided N=2",
    ),
    (
        "stream_seq2",
        "stream_seq2",
        "small_llama_feat_stream_seq2",
        "Streaming sequential N=2",
    ),
    (
        "stream_str5",
        "stream_str5",
        "small_llama_feat_stream_str5",
        "Streaming strided N=5",
    ),
    ("async_nodn", "async_nodn", "small_llama_feat_async_nodn", "Async (no DN)"),
    ("async_dn4", "async_dn4", "small_llama_feat_async_dn4", "Async + DN (N=4)"),
    ("async_dn8", "async_dn8", "small_llama_feat_async_dn8", "Async + DN (N=8)"),
    (
        "async_dn16",
        "async_dn16",
        "small_llama_feat_async_dn16",
        "Async + DN (N=16)",
    ),
    # NB: the from-scratch DyLU arms (dylu_off/dylu_on) were dropped — DyLU is run
    # warm-only (the scratch-vs-warm story is covered by the async arms), and the
    # spread was widened to ~4:1. See the warm_dylu_* arms below.
    # Warm-start arms (started from the 500M DDP checkpoint, not random init).
    # Same flags as their scratch counterparts; the only difference is the
    # server's starting master. Judge warm-async vs warm_baseline (fair, same
    # start) and warm vs scratch (does pretraining close the async gap?).
    (
        "warm_baseline",
        "warm_baseline",
        "small_llama_feat_warm_baseline",
        "Warm Baseline (sync)",
    ),
    (
        "warm_async_dn4",
        "warm_async_dn4",
        "small_llama_feat_warm_async_dn4",
        "Warm Async + DN (N=4)",
    ),
    (
        "warm_async_dn8",
        "warm_async_dn8",
        "small_llama_feat_warm_async_dn8",
        "Warm Async + DN (N=8)",
    ),
    (
        "warm_dylu_off",
        "warm_dylu_off",
        "small_llama_feat_warm_dylu_off",
        "Warm Async + DN, no DyLU",
    ),
    (
        "warm_dylu_on",
        "warm_dylu_on",
        "small_llama_feat_warm_dylu_on",
        "Warm Async + DN + DyLU",
    ),
]

# Async arms for which staleness is harvested as a column. The staleness *gate*
# (analysis/staleness.py) treats the equal-speed jitter arms (async_nodn, async_dn4)
# as round-robin (snapshot mean ~ (k-1)/2; per-submission max ~ k-1) and the
# delay-spread dylu pair as a reducer (dylu_on mean below its dylu_off control).
ASYNC_ARMS = {
    "async_nodn",
    "async_dn4",
    "async_dn8",
    "async_dn16",
    "warm_async_dn4",
    "warm_async_dn8",
    "warm_dylu_off",
    "warm_dylu_on",
}

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
    """Return (avg_tok_s, final_tokens, sync_rounds) from the server stats JSONL.

    Fallback source for total_tokens / throughput when status.json is absent.
    """
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


def parse_status_json(path):
    """Harvest the live /status snapshot captured by experiment.sh.

    Returns a dict with: total_tokens (authoritative, nested under
    aggregate_stats), sync_round, mean_staleness (server sync_round minus each
    worker's last_sync_server_round, averaged over workers that have synced),
    and grace fields (grace_batches, mean_grace_batch_size, grace_histogram,
    all_k_fraction). Missing keys -> None / {} so callers can degrade.
    """
    out = {
        "total_tokens": None,
        "sync_round": None,
        "mean_staleness": None,
        "grace_batches": None,
        "mean_grace_batch_size": None,
        "grace_histogram": {},
        "all_k_fraction": None,
    }
    if not os.path.isfile(path):
        return out
    try:
        with open(path, errors="replace") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return out
    # `forgather diloco status --json` via the orchestrator wraps the snapshot
    # under a top-level "status" dict; the direct path returns it unwrapped (with
    # "status" as the plain "running" string). Descend when wrapped.
    if isinstance(d.get("status"), dict):
        d = d["status"]
    if "error" in d and "aggregate_stats" not in d:
        return out

    agg = d.get("aggregate_stats") or {}
    out["total_tokens"] = agg.get("total_tokens")
    server_round = d.get("sync_round")
    out["sync_round"] = server_round

    workers = d.get("workers") or {}
    if server_round is not None and workers:
        stales = [
            server_round - w.get("last_sync_server_round", 0)
            for w in workers.values()
            if w.get("last_sync_server_round", 0) > 0
        ]
        if stales:
            out["mean_staleness"] = sum(stales) / len(stales)

    hist = {int(k): v for k, v in (d.get("grace_batch_histogram") or {}).items()}
    out["grace_histogram"] = hist
    out["grace_batches"] = d.get("grace_batches")
    out["mean_grace_batch_size"] = d.get("mean_grace_batch_size")
    nw = d.get("num_workers")
    total = sum(hist.values())
    if hist and nw and total:
        out["all_k_fraction"] = hist.get(nw, 0) / total
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-dir",
        default=os.path.join(HERE, "runs"),
        help="dir holding <arm>/worker0.log + <arm>/status.json (default: ./runs)",
    )
    args = ap.parse_args()

    os.makedirs(ASSETS, exist_ok=True)
    rows, summary = [], []
    for exp, runs_subdir, model_dir, label in EXPERIMENTS:
        rundir = os.path.join(args.runs_dir, runs_subdir)
        train, eval_, grad = parse_worker_log(os.path.join(rundir, "worker0.log"))
        if not train and not eval_:
            print(f"  skip {exp}: no data at {rundir}/worker0.log")
            continue
        for step, v in sorted(train.items()):
            rows.append([exp, "train_loss", step, f"{v:.6f}"])
        for step, v in sorted(eval_.items()):
            rows.append([exp, "eval_loss", step, f"{v:.6f}"])
            rows.append([exp, "perplexity", step, f"{math.exp(v):.6f}"])
        for step, v in sorted(grad.items()):
            rows.append([exp, "grad_norm", step, f"{v:.6f}"])

        st = parse_status_json(os.path.join(rundir, "status.json"))
        avg_tok_s, jsonl_tok, syncs = parse_server_jsonl(model_dir)
        total_tok = st["total_tokens"] if st["total_tokens"] else jsonl_tok
        ft = train[max(train)] if train else float("nan")
        fe = eval_[max(eval_)] if eval_ else float("nan")
        be = min(eval_.values()) if eval_ else float("nan")
        ppl = math.exp(be) if be == be else float("nan")  # exp(best eval loss)
        stale = st["mean_staleness"] if exp in ASYNC_ARMS else None
        summary.append(
            (exp, label, ft, fe, be, ppl, avg_tok_s, total_tok, syncs, stale)
        )

    with open(os.path.join(ASSETS, "curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "metric", "step", "value"])
        w.writerows(rows)

    print(f"\nwrote {len(rows)} rows to assets/curves.csv\n")
    print(
        "| Arm | final train | final eval | best eval | best ppl | "
        "avg tok/s | tokens | syncs | mean stale |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for exp, label, ft, fe, be, ppl, ts, tok, syncs, stale in summary:
        stale_s = f"{stale:.2f}" if stale is not None else "—"
        print(
            f"| {label} | {ft:.4f} | {fe:.4f} | {be:.4f} | {ppl:.2f} | "
            f"{ts/1e3:.0f}K | {tok/1e6:.0f}M | {syncs} | {stale_s} |"
        )


if __name__ == "__main__":
    main()
