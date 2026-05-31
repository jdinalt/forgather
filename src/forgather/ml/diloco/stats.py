"""Unified server-side training statistics for DiLoCo.

The DiLoCo server has no training loop of its own: the picture of "how is this
run going" lives in the workers. This module aggregates the per-worker metrics
each worker reports (on the heartbeat, sourced from the DiLoCo callback) into a
single server-level view — total tokens/FLOs/steps over the server's lifetime,
aggregate throughput/utilization/memory, and smoothed train/eval loss.

Design (see .claude_notes/DILOCO_UNIFIED_STATS_DESIGN.md):

- **Lifetime counters** (``total_tokens``, ``total_flos``, ``total_steps``) are
  accumulated from per-worker *deltas* keyed by ``worker_id``: each worker
  reports its own cumulative count, we add the increment since that worker's
  last report. Reusing a worker_id on resume (the way ``diloco worker
  --resume-workers`` does) therefore continues the count instead of
  double-adding the resumed history. These persist in the server checkpoint.
- **Live gauges** (``tok_per_sec``, ``mfu``, ``peak_memory``, ``grad_norm``) are
  computed on demand from the latest snapshot of each *currently-reporting*
  worker. They are not persisted — they repopulate as workers heartbeat after a
  restart, and a worker the server evicts is dropped from the gauge via
  :meth:`drop_worker`.
- **Loss** is a token-weighted EMA: ``S = decay*S + w*loss``, ``Z = decay*Z + w``,
  ``loss = S/Z``. Recent, high-token reports dominate; old ones decay. ``S`` and
  ``Z`` persist so smoothing survives a checkpoint resume. ``train_loss`` uses a
  stronger decay; ``eval_loss`` uses a weaker one (evals don't track precisely
  between syncs — per-worker detail stays in each worker's own TensorBoard).

The aggregator consumes a *normalized* snapshot schema (below), not raw trainer
log keys — the DiLoCo callback maps the trainer's ``logs`` dict onto it, so this
module and the server stay decoupled from trainer-side key names.

Normalized per-worker snapshot (all keys optional; missing → that metric is
just not updated by this report):

    tokens_total : int    cumulative input tokens seen (num_input_tokens_seen)
    flos_total   : float  cumulative total_flos
    step_total   : int    cumulative optimizer steps (global_step)
    tokens_window: int    tokens since this worker's last log (loss weight)
    loss         : float  latest train loss
    grad_norm    : float  latest grad norm
    tok_per_sec  : float  latest throughput
    mfu          : float  latest model-FLOPs-utilization fraction
    peak_mem     : float  latest peak memory (bytes; summed across devices)
    eval_loss    : float  latest eval loss (only on an eval report)
    eval_step    : int    step at which eval_loss was computed
"""

import threading
from typing import Any, Dict, Optional

# Keys the aggregator treats as monotonic per-worker cumulative counters,
# mapped to their lifetime-accumulator attribute.
_DELTA_FIELDS = {
    "tokens_total": "total_tokens",
    "flos_total": "total_flos",
    "step_total": "total_steps",
}

# Keys carried forward as the worker's latest live-gauge values.
_GAUGE_FIELDS = ("tok_per_sec", "mfu", "peak_mem", "grad_norm", "tokens_window")


class StatsAggregator:
    """Aggregate per-worker training metrics into a server-level view.

    Thread-safe: heartbeats arrive concurrently across workers while
    :meth:`snapshot` is read from the status handler on another thread.
    """

    def __init__(self, train_loss_decay: float = 0.9, eval_loss_decay: float = 0.5):
        # Lifetime accumulators (persisted).
        self.total_tokens: int = 0
        self.total_flos: float = 0.0
        self.total_steps: int = 0

        # Per-worker last-seen cumulative counters, for delta accumulation
        # (persisted — must survive a restart so a resumed worker_id doesn't
        # re-add its history).
        self._last_seen: Dict[str, Dict[str, float]] = {}

        # Per-worker latest gauge snapshot (NOT persisted — repopulates as
        # workers heartbeat; pruned on eviction).
        self._live: Dict[str, Dict[str, float]] = {}

        # Token-weighted EMA accumulators for loss (persisted).
        self._train_decay = float(train_loss_decay)
        self._eval_decay = float(eval_loss_decay)
        self._train_s = 0.0
        self._train_z = 0.0
        self._eval_s = 0.0
        self._eval_z = 0.0
        self._eval_step: Optional[int] = None

        self._lock = threading.Lock()

    # -- ingest ---------------------------------------------------------------

    def update(self, worker_id: str, snap: Dict[str, Any]) -> None:
        """Fold one worker's normalized snapshot into the aggregate."""
        if not worker_id or not snap:
            return
        with self._lock:
            # Lifetime delta accumulators.
            seen = self._last_seen.setdefault(worker_id, {})
            for key, attr in _DELTA_FIELDS.items():
                if key not in snap or snap[key] is None:
                    continue
                cur = snap[key]
                prev = seen.get(key)
                # First report from this worker → add its full cumulative.
                # Otherwise add the (non-negative) increment; clamp so a
                # worker that reset its counter can't subtract.
                delta = cur if prev is None else max(0, cur - prev)
                seen[key] = cur
                if attr == "total_flos":
                    self.total_flos += float(delta)
                elif attr == "total_tokens":
                    self.total_tokens += int(delta)
                else:  # total_steps
                    self.total_steps += int(delta)

            # Latest live-gauge values for this worker.
            live = self._live.setdefault(worker_id, {})
            for key in _GAUGE_FIELDS:
                if key in snap and snap[key] is not None:
                    live[key] = snap[key]

            # Token-weighted train-loss EMA.
            loss = snap.get("loss")
            if loss is not None:
                w = self._weight(snap)
                self._train_s = self._train_decay * self._train_s + w * float(loss)
                self._train_z = self._train_decay * self._train_z + w

            # Token-weighted (weak) eval-loss EMA.
            eval_loss = snap.get("eval_loss")
            if eval_loss is not None:
                w = self._weight(snap)
                self._eval_s = self._eval_decay * self._eval_s + w * float(eval_loss)
                self._eval_z = self._eval_decay * self._eval_z + w
                step = snap.get("eval_step")
                if step is not None:
                    self._eval_step = int(step)

    @staticmethod
    def _weight(snap: Dict[str, Any]) -> float:
        """Loss weight for one report: its window tokens, or 1 when unknown,
        so a report without a token count still contributes to the EMA."""
        w = snap.get("tokens_window")
        try:
            w = float(w)
        except (TypeError, ValueError):
            w = 0.0
        return w if w > 0 else 1.0

    def drop_worker(self, worker_id: str) -> None:
        """Remove a worker from the live gauges (e.g. on eviction).

        Keeps its ``_last_seen`` entry: a resumed worker reusing this id must
        continue its delta accounting rather than re-add from zero.
        """
        with self._lock:
            self._live.pop(worker_id, None)

    # -- read -----------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Current aggregate view. Live gauges sum/average over reporting
        workers; loss values are the smoothed EMAs (None until first report)."""
        with self._lock:
            tok_per_sec = sum(v.get("tok_per_sec", 0.0) for v in self._live.values())
            mfu = sum(v.get("mfu", 0.0) for v in self._live.values())
            peak_memory = sum(v.get("peak_mem", 0.0) for v in self._live.values())
            grad_norm = self._weighted_grad_norm()
            return {
                "total_tokens": self.total_tokens,
                "total_flos": self.total_flos,
                "total_steps": self.total_steps,
                "tok_per_sec": tok_per_sec,
                "mfu": mfu,
                "peak_memory": peak_memory,
                "grad_norm": grad_norm,
                "train_loss": (
                    (self._train_s / self._train_z) if self._train_z > 0 else None
                ),
                "eval_loss": (
                    (self._eval_s / self._eval_z) if self._eval_z > 0 else None
                ),
                "eval_step": self._eval_step,
                "num_reporting": len(self._live),
            }

    def _weighted_grad_norm(self) -> Optional[float]:
        """Token-weighted mean of the latest per-worker grad norms (caller
        holds the lock). Falls back to an equal-weight mean when no worker
        reported a window-token count."""
        num = 0.0
        den = 0.0
        for v in self._live.values():
            gn = v.get("grad_norm")
            if gn is None:
                continue
            w = v.get("tokens_window")
            w = float(w) if isinstance(w, (int, float)) and w > 0 else 1.0
            num += w * float(gn)
            den += w
        return (num / den) if den > 0 else None

    # -- persistence ----------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Serializable lifetime state (live gauges are intentionally omitted —
        they repopulate from heartbeats)."""
        with self._lock:
            return {
                "total_tokens": self.total_tokens,
                "total_flos": self.total_flos,
                "total_steps": self.total_steps,
                "last_seen": {k: dict(v) for k, v in self._last_seen.items()},
                "train_s": self._train_s,
                "train_z": self._train_z,
                "eval_s": self._eval_s,
                "eval_z": self._eval_z,
                "eval_step": self._eval_step,
            }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore lifetime state from a checkpoint. Tolerant of missing keys
        (pre-feature checkpoints) — anything absent keeps its initial value."""
        if not state:
            return
        with self._lock:
            self.total_tokens = int(state.get("total_tokens", 0))
            self.total_flos = float(state.get("total_flos", 0.0))
            self.total_steps = int(state.get("total_steps", 0))
            self._last_seen = {
                k: dict(v) for k, v in (state.get("last_seen") or {}).items()
            }
            self._train_s = float(state.get("train_s", 0.0))
            self._train_z = float(state.get("train_z", 0.0))
            self._eval_s = float(state.get("eval_s", 0.0))
            self._eval_z = float(state.get("eval_z", 0.0))
            es = state.get("eval_step")
            self._eval_step = int(es) if es is not None else None
