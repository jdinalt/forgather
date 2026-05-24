"""Tests for the shared ``is_endpoint_pid_alive`` helper.

This is the single source of truth for "is the trainer at this pid
still the one that wrote this endpoint.json?" — used by the Jobs API,
the scheduler's startup re-attach, and the ``forgather control``
list/cleanup CLI paths. The PID-reuse guard is the only thing
distinguishing it from a bare liveness check; the boundary cases
matter, so this file pins them.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from forgather.trainer_control import (
    PID_REUSE_SLACK_SECONDS,
    is_endpoint_pid_alive,
)


def _mock_psutil(
    *,
    is_running=True,
    status="running",
    create_time=None,
    process_raises=None,
):
    """Build a ``psutil`` mock with realistic exception classes.

    The helper catches ``psutil.NoSuchProcess`` / ``psutil.AccessDenied``
    by class — MagicMock auto-attributes won't be caught, so they have
    to be real subclasses of ``Exception``.
    """

    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    mock_psutil = MagicMock()
    mock_psutil.STATUS_ZOMBIE = "zombie"
    mock_psutil.NoSuchProcess = NoSuchProcess
    mock_psutil.AccessDenied = AccessDenied

    if process_raises is not None:
        mock_psutil.Process.side_effect = process_raises
    else:
        proc = MagicMock()
        proc.is_running.return_value = is_running
        proc.status.return_value = status
        if create_time is not None:
            proc.create_time.return_value = create_time
        mock_psutil.Process.return_value = proc

    return mock_psutil, NoSuchProcess


class TestIsEndpointPidAlive:
    def test_none_pid_returns_false(self):
        assert is_endpoint_pid_alive(None, 1000.0) is False
        assert is_endpoint_pid_alive(None, None) is False

    def test_current_pid_alive_when_started_at_omitted(self):
        # No started_at → bare liveness only. Current process is alive.
        assert is_endpoint_pid_alive(os.getpid(), None) is True

    def test_high_pid_is_not_alive(self):
        # Nothing at pid 999999999.
        assert is_endpoint_pid_alive(999999999, 1000.0) is False
        assert is_endpoint_pid_alive(999999999, None) is False

    def test_recycled_pid_is_not_live(self):
        """Live process created well after the endpoint → kernel recycled."""
        mock_psutil, _ = _mock_psutil(create_time=5000.0)
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert is_endpoint_pid_alive(457, started_at=1000.0) is False

    def test_within_slack_is_live(self):
        # 5s after started_at — well inside the slack window.
        mock_psutil, _ = _mock_psutil(create_time=1005.0)
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert is_endpoint_pid_alive(457, started_at=1000.0) is True

    def test_boundary_at_slack_is_live(self):
        # Exactly at started_at + SLACK is still live (the check is
        # strict ``>``, so the boundary value is *not* "recycled").
        mock_psutil, _ = _mock_psutil(
            create_time=1000.0 + PID_REUSE_SLACK_SECONDS,
        )
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert is_endpoint_pid_alive(457, started_at=1000.0) is True

    def test_just_past_boundary_is_recycled(self):
        # One tick past the boundary → recycled. Pins the contract so
        # a future tightening of the constant can't silently break
        # actively-running trainers on slow-forking hosts.
        mock_psutil, _ = _mock_psutil(
            create_time=1000.0 + PID_REUSE_SLACK_SECONDS + 0.001,
        )
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert is_endpoint_pid_alive(457, started_at=1000.0) is False

    def test_started_at_none_skips_reuse_guard(self):
        # When started_at is unknown we can't run the reuse guard, so
        # we fall back to bare liveness — even a freshly-created
        # process at the same pid is "live".
        mock_psutil, _ = _mock_psutil(create_time=99999999.0)
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert is_endpoint_pid_alive(457, started_at=None) is True

    def test_zombie_is_not_live(self):
        mock_psutil, _ = _mock_psutil(status="zombie", create_time=1000.0)
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert is_endpoint_pid_alive(457, started_at=1000.0) is False

    def test_is_running_false_is_not_live(self):
        mock_psutil, _ = _mock_psutil(is_running=False, create_time=1000.0)
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert is_endpoint_pid_alive(457, started_at=1000.0) is False

    def test_process_raises_no_such_process(self):
        # Race: pid disappears between our enumeration and the check.
        class NoSuchProcess(Exception):
            pass

        mock_psutil = MagicMock()
        mock_psutil.NoSuchProcess = NoSuchProcess
        mock_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        mock_psutil.Process.side_effect = NoSuchProcess("gone")

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert is_endpoint_pid_alive(457, started_at=1000.0) is False

    def test_create_time_race_raises_no_such_process(self):
        # is_running()/status() pass, then create_time() raises because
        # the process exited in between. Should be caught → False.
        class NoSuchProcess(Exception):
            pass

        class AccessDenied(Exception):
            pass

        mock_psutil = MagicMock()
        mock_psutil.STATUS_ZOMBIE = "zombie"
        mock_psutil.NoSuchProcess = NoSuchProcess
        mock_psutil.AccessDenied = AccessDenied

        proc = MagicMock()
        proc.is_running.return_value = True
        proc.status.return_value = "running"
        proc.create_time.side_effect = NoSuchProcess("died between calls")
        mock_psutil.Process.return_value = proc

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            assert is_endpoint_pid_alive(457, started_at=1000.0) is False


class TestNoPsutilFallback:
    """Without psutil, the reuse guard is unavailable. Confirm we still
    detect dead pids correctly via ``os.kill(pid, 0)`` and don't crash.
    """

    def test_no_psutil_alive_for_current_pid(self):
        with patch.dict(sys.modules, {"psutil": None}):
            # Bare existence: current pid is alive.
            assert is_endpoint_pid_alive(os.getpid(), 1000.0) is True

    def test_no_psutil_dead_for_high_pid(self):
        with patch.dict(sys.modules, {"psutil": None}):
            assert is_endpoint_pid_alive(999999999, 1000.0) is False

    def test_no_psutil_none_pid_still_returns_false(self):
        with patch.dict(sys.modules, {"psutil": None}):
            assert is_endpoint_pid_alive(None, 1000.0) is False
