"""Tests for helpers in routes/jobs.py — notably _pid_alive."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from forgather_server.routes.jobs import _pid_alive


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
