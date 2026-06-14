#!/usr/bin/env python3
"""Staleness gate — with reducer-vs-should-be-stale semantics.

The async conclusions assume the workers run genuinely out of lock-step, so a
submission's pseudo-gradient is stale by ~workers-1 server rounds by the time it
lands. For each async arm this reads the live /status snapshot
(``runs/<arm>/status.json``) and derives each worker's staleness as

    staleness = server sync_round  -  worker.last_sync_server_round

But not every async arm should be stale. Two classes (see README §3.3):

  * SHOULD-BE-STALE — ``async_nodn`` / ``async_dn4`` / ``dylu_off``: the jitter (or
    the spread, for the DyLU-off control) is *supposed* to produce staleness
    ~ k-1. These are GATED: PASS iff mean ≈ k-1 within tolerance. If they aren't
    stale, the async arms aren't testing what we claim — so this runs BEFORE the
    headline tier.
  * REDUCER — ``dylu_on``: DyLU's whole job is to *cut* staleness. Its success
    criterion is staleness **below its control** (``dylu_off``), NOT ≈ k-1. It is
    reported as reduced/not-reduced vs the control and is never failed for landing
    below k-1.

(Grace is not a study arm — its staleness-reduction is validate-only; see §3.5.)

    python analysis/staleness.py            # all async arms
    python analysis/staleness.py <arm> ...  # specific arm(s)

Reads runs/<arm>/status.json directly (ephemeral). Tolerance is generous (the
induced staleness is stochastic and budget-bounded).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "runs")

SHOULD_BE_STALE = ["async_nodn", "async_dn4", "dylu_off"]
REDUCERS = {"dylu_on": "dylu_off"}  # arm -> its staleness control
TOLERANCE = 1.0  # |mean staleness - (workers-1)| within this => PASS (should-be-stale)


def load_staleness(arm):
    """Return (mean_staleness, num_workers, per_worker [(id, staleness)])."""
    path = os.path.join(RUNS, arm, "status.json")
    if not os.path.isfile(path):
        return None, None, []
    try:
        with open(path, errors="replace") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None, []
    # orchestrator-routed `status --json` wraps the snapshot under "status".
    if isinstance(d.get("status"), dict):
        d = d["status"]
    server_round = d.get("sync_round")
    workers = d.get("workers") or {}
    nw = d.get("num_workers") or len(workers)
    if server_round is None or not workers:
        return None, nw, []
    per = [
        (wid, server_round - w.get("last_sync_server_round", 0))
        for wid, w in sorted(workers.items())
        if w.get("last_sync_server_round", 0) > 0
    ]
    mean = sum(s for _, s in per) / len(per) if per else None
    return mean, nw, per


def main():
    arms = sys.argv[1:]
    if arms:
        # An explicit arg is a reducer iff named in REDUCERS, else gated as stale.
        do_stale = [a for a in arms if a not in REDUCERS]
        do_reduce = [a for a in arms if a in REDUCERS]
    else:
        do_stale, do_reduce = SHOULD_BE_STALE, list(REDUCERS)

    any_data = False

    print("== should-be-stale (gate: mean ~ workers-1) ==")
    print(f"{'arm':<18}{'workers':>9}{'expected':>10}{'mean stale':>12}{'gate':>8}")
    for arm in do_stale:
        mean, nw, per = load_staleness(arm)
        if mean is None:
            print(f"{arm:<18}{'(no status.json / no synced workers)':>49}")
            continue
        any_data = True
        expected = (nw - 1) if nw else float("nan")
        gate = "PASS" if abs(mean - expected) <= TOLERANCE else "FAIL"
        print(f"{arm:<18}{nw:>9}{expected:>10.1f}{mean:>12.2f}{gate:>8}")
        print(f"  per-worker: {'  '.join(f'{w}:{s}' for w, s in per)}")

    if do_reduce:
        print("\n== reducers (success: staleness BELOW control, not ~ k-1) ==")
        print(
            f"{'arm':<14}{'control':<12}{'arm stale':>11}{'ctrl stale':>12}{'verdict':>11}"
        )
        for arm in do_reduce:
            ctrl = REDUCERS.get(arm)
            mean, nw, _ = load_staleness(arm)
            cmean, _, _ = load_staleness(ctrl) if ctrl else (None, None, [])
            if mean is None:
                print(f"{arm:<14}{ctrl or '-':<12}{'(no status.json)':>34}")
                continue
            any_data = True
            if cmean is None:
                print(f"{arm:<14}{ctrl or '-':<12}{mean:>11.2f}{'(no control)':>23}")
                continue
            verdict = "reduced" if mean < cmean else "NOT reduced"
            print(f"{arm:<14}{ctrl:<12}{mean:>11.2f}{cmean:>12.2f}{verdict:>11}")

    if not any_data:
        print("\nNo staleness data — run an async arm first (experiment.sh).")


if __name__ == "__main__":
    main()
