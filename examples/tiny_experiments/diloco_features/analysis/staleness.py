#!/usr/bin/env python3
"""Staleness gate — snapshot-mean ≈ (k-1)/2 for decorrelation; per-submission max ≈ k-1.

The async conclusions assume the jittered workers run genuinely out of lock-step.
We verify it from the captured /status snapshot, where each worker's staleness is

    staleness = server sync_round  −  worker.last_sync_server_round

The round-robin subtlety: for k EQUAL-average-speed workers (the jitter arms), at
any instant the workers sit at staleness {0, 1, …, k-1} — one at each phase of the
k-step sync cycle. So two different numbers, both meaningful:
  * the SNAPSHOT MEAN is (k-1)/2  (= 1.5 for k=4) — the signature of *full*
    decorrelation. A synchronized (un-jittered) run collapses this toward ~0.
  * the PER-SUBMISSION staleness — the async paper's "~k-1" — is how stale a
    gradient is at the instant it is applied = the MAX of the cycle (= 3 for k=4).
So we gate the jitter arms on snapshot mean ≈ (k-1)/2 and report max ≈ k-1.
(More jitter cannot push the snapshot mean above (k-1)/2 at equal average speed —
the workers stay round-robin — so a low mean here means weak jitter, not a low cap.)

Two arm classes (README §3.3):
  * JITTER-GATED — async_nodn / async_dn4: equal-speed jitter ⇒ round-robin.
    PASS iff snapshot mean ≈ (k-1)/2 (if the jitter weren't decorrelating, the
    workers cluster near staleness 0 and the mean collapses).
  * REDUCER — dylu_on vs its dylu_off control: a *delay spread* (not jitter), so
    NOT round-robin (slow workers run staler). Not gated at a fixed value; success
    is dylu_on's mean staleness BELOW dylu_off's (DyLU re-paces to cut it).

    python analysis/staleness.py            # the jitter gate + the dylu reducer
    python analysis/staleness.py <arm> ...  # specific arm(s)

Reads runs/<arm>/status.json directly (ephemeral).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "runs")

JITTER_GATED = ["async_nodn", "async_dn4"]  # equal-speed jitter -> round-robin
REDUCERS = {"dylu_on": "dylu_off"}  # delay-spread; dylu_on should be LESS stale
TOLERANCE = 0.75  # |snapshot mean - (k-1)/2| within this => decorrelated (PASS)


def load_staleness(arm):
    """Return (mean, max, num_workers, per_worker [(id, staleness)])."""
    path = os.path.join(RUNS, arm, "status.json")
    if not os.path.isfile(path):
        return None, None, None, []
    try:
        with open(path, errors="replace") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None, None, []
    if isinstance(d.get("status"), dict):  # orchestrator wraps the snapshot
        d = d["status"]
    server_round = d.get("sync_round")
    workers = d.get("workers") or {}
    nw = d.get("num_workers") or len(workers)
    if server_round is None or not workers:
        return None, None, nw, []
    per = [
        (wid, server_round - w.get("last_sync_server_round", 0))
        for wid, w in sorted(workers.items())
        if w.get("last_sync_server_round", 0) > 0
    ]
    if not per:
        return None, None, nw, []
    vals = [s for _, s in per]
    return sum(vals) / len(vals), max(vals), nw, per


def main():
    arms = sys.argv[1:]
    if arms:
        do_gate = [a for a in arms if a not in REDUCERS]
        do_reduce = [a for a in arms if a in REDUCERS]
    else:
        do_gate, do_reduce = JITTER_GATED, list(REDUCERS)

    any_data = False

    print("== jitter arms (gate: snapshot mean ~ (k-1)/2; per-submission max ~ k-1) ==")
    print(f"{'arm':<14}{'workers':>9}{'mean':>8}{'exp':>7}{'max':>6}{'gate':>8}")
    for arm in do_gate:
        mean, mx, nw, per = load_staleness(arm)
        if mean is None:
            print(f"{arm:<14}{'(no status.json / no synced workers)':>49}")
            continue
        any_data = True
        exp = (nw - 1) / 2 if nw else float("nan")
        gate = "PASS" if abs(mean - exp) <= TOLERANCE else "FAIL"
        print(f"{arm:<14}{nw:>9}{mean:>8.2f}{exp:>7.2f}{mx:>6}{gate:>8}")
        print(f"  per-worker: {'  '.join(f'{w}:{s}' for w, s in per)}")

    if do_reduce:
        print(
            "\n== reducer (success: dylu_on mean staleness BELOW its dylu_off control) =="
        )
        print(
            f"{'arm':<10}{'control':<11}{'arm mean':>9}{'ctrl mean':>11}{'verdict':>13}"
        )
        for arm in do_reduce:
            ctrl = REDUCERS.get(arm)
            mean, _, _, _ = load_staleness(arm)
            cmean = load_staleness(ctrl)[0] if ctrl else None
            if mean is None:
                print(f"{arm:<10}{ctrl or '-':<11}{'(no status.json)':>33}")
                continue
            any_data = True
            if cmean is None:
                print(f"{arm:<10}{ctrl or '-':<11}{mean:>9.2f}{'(no control)':>24}")
                continue
            verdict = "reduced" if mean < cmean else "NOT reduced"
            print(f"{arm:<10}{ctrl:<11}{mean:>9.2f}{cmean:>11.2f}{verdict:>13}")

    if not any_data:
        print("\nNo staleness data — run an async arm first (experiment.sh).")


if __name__ == "__main__":
    main()
