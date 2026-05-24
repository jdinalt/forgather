"""Tests for helpers in routes/jobs.py — notably _pid_alive and the
``_endpoint_is_live`` wrapper that delegates to
``trainer_control.is_endpoint_pid_alive``. Deep coverage of the
PID-reuse-aware liveness check lives in
``tests/unit/forgather/test_trainer_control.py``; this file only
verifies the wrapper plumbs ``JobInfo`` fields through correctly.
"""

import os
import sys
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


class TestEndpointIsLiveWrapper:
    """The wrapper just passes ``(ep.pid, ep.started_at)`` to the
    canonical helper. Smoke-test that plumbing here; behaviour tests
    for the helper itself live in test_trainer_control.py.
    """

    def test_delegates_to_canonical_helper(self):
        with patch.object(
            trainer_control, "is_endpoint_pid_alive", return_value=True
        ) as mock_fn:
            ep = _ep(pid=457, started_at=1000.0)
            assert _endpoint_is_live(ep) is True
        mock_fn.assert_called_once_with(457, 1000.0)

    def test_passes_through_false(self):
        with patch.object(trainer_control, "is_endpoint_pid_alive", return_value=False):
            assert _endpoint_is_live(_ep(pid=457, started_at=1000.0)) is False

    def test_none_pid_passes_through(self):
        # Wrapper must forward None pid; canonical helper handles it.
        with patch.object(
            trainer_control, "is_endpoint_pid_alive", return_value=False
        ) as mock_fn:
            _endpoint_is_live(_ep(pid=None, started_at=1000.0))
        mock_fn.assert_called_once_with(None, 1000.0)
