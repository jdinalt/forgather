"""Tests for helpers in routes/jobs.py — notably _pid_alive."""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest
from forgather_server.routes.jobs import _endpoint_is_live, _pid_alive

from forgather import trainer_control


class TestPidAlive:
    def test_none_returns_false(self):
        assert _pid_alive(None) is False

    def test_current_pid_is_alive(self):
        # The current process is definitely alive.
        assert _pid_alive(os.getpid()) is True

    def test_very_high_pid_not_alive(self):
        # PID 999999999 is almost certainly not a real process.
        result = _pid_alive(999999999)
        assert result is False

    def test_psutil_import_error_returns_true(self):
        """If psutil is not importable, _pid_alive optimistically returns True."""
        with patch.dict(sys.modules, {"psutil": None}):
            # Force the import to fail by temporarily removing psutil.
            # The function does `import psutil` inside a try block, which
            # would raise ImportError when the module is None in sys.modules —
            # but the actual CPython behaviour is that `None` in sys.modules
            # means "this module is known to NOT exist" and raises ImportError.
            result = _pid_alive(os.getpid())
            # With psutil unavailable the function returns True (optimistic).
            assert result is True

    def test_no_such_process_returns_false(self):
        """psutil.NoSuchProcess on is_running() should yield False.

        Build realistic exception classes so the ``except`` tuple in
        _pid_alive actually catches them.
        """

        # Define the exception types first so they can be raised and caught.
        class NoSuchProcess(Exception):
            pass

        class AccessDenied(Exception):
            pass

        mock_psutil = MagicMock()
        mock_psutil.pid_exists.return_value = True
        mock_psutil.NoSuchProcess = NoSuchProcess
        mock_psutil.AccessDenied = AccessDenied
        mock_proc = MagicMock()
        mock_proc.is_running.side_effect = NoSuchProcess("no such process")
        mock_psutil.Process.return_value = mock_proc

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = _pid_alive(9999)
        assert result is False


def _ep(pid, started_at):
    return trainer_control.JobInfo(
        job_id="job_test",
        host="127.0.0.1",
        port=8900,
        pid=pid,
        started_at=started_at,
    )


class TestEndpointIsLive:
    """PID-reuse-aware liveness check for externally-discovered endpoints.

    Repro for the original bug: a stale ``endpoint.json`` from before a
    host reboot named a pid the kernel later reassigned to an unrelated
    daemon. The bare ``_pid_alive`` check returned True, locking the
    ghost into the Jobs view as a phantom "running" job that the UI's
    delete handler refused to evict.
    """

    def test_none_pid_returns_false(self):
        assert _endpoint_is_live(_ep(None, time.time())) is False

    def test_current_pid_with_matching_started_at_is_live(self):
        # Current process is alive and was created before "now" — should
        # pass the create_time vs started_at guard.
        assert _endpoint_is_live(_ep(os.getpid(), time.time())) is True

    def test_very_high_pid_not_live(self):
        # Nothing at this pid → not live, regardless of started_at.
        assert _endpoint_is_live(_ep(999999999, time.time())) is False

    def test_recycled_pid_is_not_live(self):
        """Live process younger than the endpoint → pid was recycled."""

        class NoSuchProcess(Exception):
            pass

        class AccessDenied(Exception):
            pass

        mock_psutil = MagicMock()
        mock_psutil.STATUS_ZOMBIE = "zombie"
        mock_psutil.NoSuchProcess = NoSuchProcess
        mock_psutil.AccessDenied = AccessDenied

        # Endpoint says the trainer started at t=1000.
        # Live process at the same pid was created at t=5000 — long after.
        # That mismatch means the kernel recycled the pid.
        mock_proc = MagicMock()
        mock_proc.is_running.return_value = True
        mock_proc.status.return_value = "running"
        mock_proc.create_time.return_value = 5000.0
        mock_psutil.Process.return_value = mock_proc

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert _endpoint_is_live(_ep(457, started_at=1000.0)) is False

    def test_create_time_within_slack_is_live(self):
        """create_time slightly after started_at is fine (fork timing)."""

        class NoSuchProcess(Exception):
            pass

        class AccessDenied(Exception):
            pass

        mock_psutil = MagicMock()
        mock_psutil.STATUS_ZOMBIE = "zombie"
        mock_psutil.NoSuchProcess = NoSuchProcess
        mock_psutil.AccessDenied = AccessDenied

        mock_proc = MagicMock()
        mock_proc.is_running.return_value = True
        mock_proc.status.return_value = "running"
        # 5 seconds after started_at — within the 10s slack window.
        mock_proc.create_time.return_value = 1005.0
        mock_psutil.Process.return_value = mock_proc

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert _endpoint_is_live(_ep(457, started_at=1000.0)) is True

    def test_zombie_is_not_live(self):
        class NoSuchProcess(Exception):
            pass

        class AccessDenied(Exception):
            pass

        mock_psutil = MagicMock()
        mock_psutil.STATUS_ZOMBIE = "zombie"
        mock_psutil.NoSuchProcess = NoSuchProcess
        mock_psutil.AccessDenied = AccessDenied

        mock_proc = MagicMock()
        mock_proc.is_running.return_value = True
        mock_proc.status.return_value = "zombie"
        mock_proc.create_time.return_value = 1000.0
        mock_psutil.Process.return_value = mock_proc

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert _endpoint_is_live(_ep(457, started_at=1000.0)) is False

    def test_no_such_process_returns_false(self):
        class NoSuchProcess(Exception):
            pass

        class AccessDenied(Exception):
            pass

        mock_psutil = MagicMock()
        mock_psutil.NoSuchProcess = NoSuchProcess
        mock_psutil.AccessDenied = AccessDenied
        mock_psutil.Process.side_effect = NoSuchProcess("gone")

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert _endpoint_is_live(_ep(457, started_at=1000.0)) is False
