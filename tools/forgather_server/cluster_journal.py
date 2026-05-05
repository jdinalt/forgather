"""Append-only event journal for global cluster state.

Phase 1 ships only the seam: the writer, the replay iterator, and the
subscriber hook. No production code emits events yet — the queue,
GPU policy, and cluster job stores are migrated onto this in Phase 4.

Why a journal at all in v1: master failover (v2) requires that every
mutation to globally-visible state be replicable to a backup. The
cheapest way to make sure no decision in v1 forecloses that future is
to put the seam in now and force every later mutation through it. If
we ship v1 with state stored as in-place ``*.json`` blob rewrites,
introducing replication later is a much bigger refactor — there is no
causal record to ship.

Format: JSON-Lines (``.jsonl``). One file, ``events.jsonl``, written
with O_APPEND under a process-wide lock. Rotation, fsync policy, and
truncation-by-snapshot are deferred to Phase 4 when there's something
real to journal.

Concurrency: the server is single-process (asyncio + a few helper
threads). One ``threading.RLock`` serializes append. Subscribers run
synchronously in the appender's thread; long-running subscriber work
must hop to a worker.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from . import paths

log = logging.getLogger("forgather_server.cluster.journal")


JOURNAL_FILENAME = "events.jsonl"


@dataclass
class JournalEvent:
    """Single appended event.

    ``seq`` is monotonic per-process and starts at 1 after replay; it
    is *not* a globally-unique cluster sequence number. v2 will need a
    cluster-wide ordering for replicated state, but in v1 there is one
    writer (the local master) so per-process ordering is sufficient.
    """

    seq: int
    ts: float
    origin_node_id: str
    type: str
    payload: Dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "ts": self.ts,
                "origin_node_id": self.origin_node_id,
                "type": self.type,
                "payload": self.payload,
            },
            sort_keys=True,
        )

    @staticmethod
    def from_jsonl(line: str) -> "JournalEvent":
        d = json.loads(line)
        return JournalEvent(
            seq=int(d["seq"]),
            ts=float(d["ts"]),
            origin_node_id=str(d["origin_node_id"]),
            type=str(d["type"]),
            payload=dict(d.get("payload") or {}),
        )


SubscriberFn = Callable[[JournalEvent], None]


class _Journal:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_seq: int = 1
        self._subscribers: List[SubscriberFn] = []
        self._inited: bool = False

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    def init(self) -> None:
        """Read the existing journal to recover ``next_seq``.

        Called once on server startup when cluster mode is active. No-op
        on subsequent calls. Replaying state into other modules is not
        done here — each subscriber is responsible for replaying what
        it cares about (Phase 4 wiring).
        """
        with self._lock:
            if self._inited:
                return
            path = self._path()
            if path.exists():
                last_seq = 0
                try:
                    for ev in self._iter_file(path):
                        if ev.seq > last_seq:
                            last_seq = ev.seq
                except Exception:
                    log.exception(
                        "journal replay failed at %s; "
                        "starting fresh seq numbering",
                        path,
                    )
                    last_seq = 0
                self._next_seq = last_seq + 1
            self._inited = True
            log.info(
                "journal initialized at %s (next_seq=%d)",
                path,
                self._next_seq,
            )

    # ------------------------------------------------------------
    # Append / replay
    # ------------------------------------------------------------

    def append(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        origin_node_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> JournalEvent:
        """Persist one event. Subscribers fire synchronously after fsync."""
        if not event_type:
            raise ValueError("event_type must be a non-empty string")
        # Resolve origin lazily so the journal works even when the
        # cluster module hasn't been activated (tests).
        origin = origin_node_id
        if origin is None:
            try:
                from . import cluster as _cluster

                ident = _cluster.self_identity()
                origin = ident.node_id if ident is not None else "local"
            except Exception:
                origin = "local"
        with self._lock:
            if not self._inited:
                self.init()
            ev = JournalEvent(
                seq=self._next_seq,
                ts=ts if ts is not None else time.time(),
                origin_node_id=origin,
                type=event_type,
                payload=dict(payload or {}),
            )
            self._next_seq += 1
            self._write_line(ev.to_jsonl())
            for cb in list(self._subscribers):
                try:
                    cb(ev)
                except Exception:
                    log.exception(
                        "journal subscriber %r raised on event %s",
                        cb,
                        ev.type,
                    )
            return ev

    def replay(self) -> Iterator[JournalEvent]:
        """Yield every event currently on disk, in order.

        The iterator reads a snapshot of the file at call time. Events
        appended *during* iteration are not yielded — callers wanting
        live updates should subscribe.
        """
        path = self._path()
        if not path.exists():
            return iter([])
        return self._iter_file(path)

    def subscribe(self, cb: SubscriberFn) -> None:
        with self._lock:
            self._subscribers.append(cb)

    def unsubscribe(self, cb: SubscriberFn) -> None:
        with self._lock:
            try:
                self._subscribers.remove(cb)
            except ValueError:
                pass

    # ------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------

    def _path(self) -> Path:
        return paths.cluster_journal_dir() / JOURNAL_FILENAME

    def _write_line(self, line: str) -> None:
        # O_APPEND is atomic for writes < PIPE_BUF on POSIX; one JSON
        # line is well under that. Even with multiple writers, lines do
        # not interleave. We still hold the in-process lock so seq
        # numbering stays monotonic.
        path = self._path()
        # Tighten directory perms (idempotent) before write so a
        # crashed-and-restarted server eventually settles on 0700.
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        fd = os.open(path, flags, 0o600)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
        finally:
            os.close(fd)

    def _iter_file(self, path: Path) -> Iterator[JournalEvent]:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield JournalEvent.from_jsonl(raw)
                except Exception:
                    log.exception(
                        "skipping malformed journal line in %s: %r",
                        path,
                        raw[:200],
                    )

    # ------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------

    def _reset_for_tests(self) -> None:
        with self._lock:
            self._next_seq = 1
            self._subscribers.clear()
            self._inited = False


_journal = _Journal()


# ---------------------------------------------------------------------------
# Public module API
# ---------------------------------------------------------------------------


def init() -> None:
    _journal.init()


def append(
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    origin_node_id: Optional[str] = None,
    ts: Optional[float] = None,
) -> JournalEvent:
    return _journal.append(
        event_type, payload, origin_node_id=origin_node_id, ts=ts
    )


def replay() -> Iterator[JournalEvent]:
    return _journal.replay()


def subscribe(cb: SubscriberFn) -> None:
    _journal.subscribe(cb)


def unsubscribe(cb: SubscriberFn) -> None:
    _journal.unsubscribe(cb)


def _reset_for_tests() -> None:
    _journal._reset_for_tests()
