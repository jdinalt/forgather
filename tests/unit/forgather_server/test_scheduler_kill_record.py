"""Unit tests for ``scheduler._kill_record`` force-kill / abort semantics.

The motivating incident: a stuck/orphaned training process can OUTLIVE its
JobRecord — the record is marked terminal (done/failed/aborted) and dropped
from the UI while the PID lingers, holding a GPU. Soft ``kill`` (SIGTERM) keeps
the terminal guard (a terminal record means "already gone, nothing to do"), but
``force-kill`` (SIGKILL, "do whatever it takes") must still reap a live process
behind a terminal record.
"""

from __future__ import annotations

import signal
from unittest.mock import patch

from forgather_server import scheduler
from forgather_server.job_records import JobRecord


def _rec(status: str, pid: int = 4321, **extra) -> JobRecord:
    return JobRecord(
        queue_id="q-test",
        job_type="training",
        status=status,
        pid=pid,
        **extra,
    )


def test_force_kill_reaps_live_process_behind_terminal_record():
    """SIGKILL on a terminal record still signals a lingering live PID."""
    rec = _rec("failed", pid=4321)
    with (
        patch.object(scheduler.job_records, "get_record", return_value=rec),
        patch.object(scheduler, "_pid_is_alive", return_value=True),
        patch.object(scheduler, "_wait_for_pid_exit", return_value=True) as wait,
        patch.object(scheduler.launcher, "kill_process_group") as kpg,
    ):
        ok = scheduler.force_kill_record("q-test")

    assert ok is True
    kpg.assert_called_once_with(4321, signal.SIGKILL)
    wait.assert_called_once()


def test_force_kill_terminal_record_with_dead_pid_is_noop():
    """No live PID behind the terminal record => nothing to reap."""
    rec = _rec("aborted", pid=4321)
    with (
        patch.object(scheduler.job_records, "get_record", return_value=rec),
        patch.object(scheduler, "_pid_is_alive", return_value=False),
        patch.object(scheduler.launcher, "kill_process_group") as kpg,
    ):
        ok = scheduler.force_kill_record("q-test")

    assert ok is False
    kpg.assert_not_called()


def test_soft_kill_terminal_record_is_noop_even_if_pid_alive():
    """SIGTERM keeps the terminal guard: a soft kill never reaps an orphan."""
    rec = _rec("done", pid=4321)
    with (
        patch.object(scheduler.job_records, "get_record", return_value=rec),
        patch.object(scheduler, "_pid_is_alive", return_value=True),
        patch.object(scheduler.launcher, "kill_process_group") as kpg,
    ):
        ok = scheduler.abort_record("q-test")

    assert ok is False
    kpg.assert_not_called()


def test_force_kill_unknown_record_returns_false():
    with patch.object(scheduler.job_records, "get_record", return_value=None):
        assert scheduler.force_kill_record("missing") is False
