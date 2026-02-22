#!/usr/bin/env python3
"""
Unit tests for resumable logging functionality.

Tests the following components:
- ResumableSummaryWriter: Lazy, checkpoint-aware TensorBoard SummaryWriter wrapper
- JsonLogger: Stateful JSON logger with resume and truncation support
- _parse_json_log: Robust JSON log file parsing with corruption handling
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from forgather.ml.trainer.callbacks.json_logger import JsonLogger, _parse_json_log
from forgather.ml.trainer.callbacks.resumable_summary_writer import (
    ResumableSummaryWriter,
)
from forgather.ml.trainer.trainer_types import TrainerControl

# ---------------------------------------------------------------------------
# _parse_json_log tests
# ---------------------------------------------------------------------------


class TestParseJsonLog:
    """Tests for the _parse_json_log helper that handles corrupted JSON files."""

    def test_valid_complete_json(self):
        """Parses a well-formed JSON array."""
        content = json.dumps(
            [
                {"global_step": 1, "loss": 1.0},
                {"global_step": 2, "loss": 0.8},
            ]
        )
        result = _parse_json_log(content)
        assert len(result) == 2
        assert result[0]["global_step"] == 1
        assert result[1]["global_step"] == 2

    def test_empty_string(self):
        """Returns empty list for empty string."""
        assert _parse_json_log("") == []

    def test_whitespace_only(self):
        """Returns empty list for whitespace-only string."""
        assert _parse_json_log("   \n\t  ") == []

    def test_empty_array(self):
        """Parses an empty JSON array."""
        assert _parse_json_log("[]") == []

    def test_missing_closing_bracket(self):
        """Handles file not properly closed (missing ])."""
        content = '[\n{"global_step": 1, "loss": 1.0},\n{"global_step": 2, "loss": 0.8}'
        result = _parse_json_log(content)
        assert len(result) == 2
        assert result[1]["global_step"] == 2

    def test_missing_closing_bracket_with_trailing_comma(self):
        """Handles missing ] and a trailing comma after last record."""
        content = (
            '[\n{"global_step": 1, "loss": 1.0},\n{"global_step": 2, "loss": 0.8},'
        )
        result = _parse_json_log(content)
        assert len(result) == 2

    def test_trailing_comma_before_closing_bracket(self):
        """Handles trailing comma inside an otherwise valid array."""
        content = (
            '[\n{"global_step": 1, "loss": 1.0},\n{"global_step": 2, "loss": 0.8},\n]'
        )
        result = _parse_json_log(content)
        assert len(result) == 2

    def test_partially_written_last_record(self):
        """Handles a truncated last record (crash mid-write)."""
        content = '[\n{"global_step": 1, "loss": 1.0},\n{"global_step": 2, "lo'
        result = _parse_json_log(content)
        # Should recover the first complete record
        assert len(result) == 1
        assert result[0]["global_step"] == 1

    def test_single_record_no_closing(self):
        """Handles a single record with no closing bracket."""
        content = '[\n{"global_step": 1, "loss": 1.0}'
        result = _parse_json_log(content)
        assert len(result) == 1
        assert result[0]["global_step"] == 1

    def test_completely_corrupted(self):
        """Returns empty list for completely unparseable content."""
        result = _parse_json_log("this is not json at all {{{")
        assert result == []

    def test_just_opening_bracket(self):
        """Returns empty list for a file with only the opening bracket."""
        result = _parse_json_log("[\n")
        assert result == []

    def test_real_world_format(self):
        """Parses the exact format written by JsonLogger."""
        # Simulate what JsonLogger writes: "[\n" + records + "\n]"
        records = [
            {"timestamp": 1000.0, "global_step": 10, "epoch": 0.1, "loss": 2.5},
            {"timestamp": 1100.0, "global_step": 20, "epoch": 0.2, "loss": 2.0},
            {"timestamp": 1200.0, "global_step": 30, "epoch": 0.3, "loss": 1.5},
        ]
        content = "[\n"
        for i, r in enumerate(records):
            prefix = ",\n" if i > 0 else ""
            content += prefix + json.dumps(r)
        content += "\n]"

        result = _parse_json_log(content)
        assert len(result) == 3
        assert result[0]["global_step"] == 10
        assert result[2]["loss"] == 1.5

    def test_real_world_format_unclosed(self):
        """Parses the format written by JsonLogger when process was killed
        before close() was called (missing trailing ']')."""
        records = [
            {"timestamp": 1000.0, "global_step": 10, "epoch": 0.1, "loss": 2.5},
            {"timestamp": 1100.0, "global_step": 20, "epoch": 0.2, "loss": 2.0},
        ]
        content = "[\n"
        for i, r in enumerate(records):
            prefix = ",\n" if i > 0 else ""
            content += prefix + json.dumps(r)
        # No closing bracket -- process was killed

        result = _parse_json_log(content)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# ResumableSummaryWriter tests
# ---------------------------------------------------------------------------


class TestResumableSummaryWriterInit:
    """Tests for ResumableSummaryWriter initialization."""

    def test_init_stores_log_dir(self):
        rsw = ResumableSummaryWriter(log_dir="/tmp/test_logs")
        assert rsw._new_log_dir == "/tmp/test_logs"
        assert rsw._active_log_dir == "/tmp/test_logs"
        assert rsw._writer is None
        assert rsw._resumed is False

    def test_writer_is_not_created_on_init(self):
        """Writer construction is deferred until first use."""
        rsw = ResumableSummaryWriter(log_dir="/tmp/nonexistent")
        assert rsw._writer is None


class TestResumableSummaryWriterStateDict:
    """Tests for ResumableSummaryWriter state_dict / load_state_dict."""

    def test_state_dict_returns_active_dir(self):
        rsw = ResumableSummaryWriter(log_dir="/tmp/run_001")
        sd = rsw.state_dict()
        assert sd == {"log_dir": "/tmp/run_001"}

    def test_load_state_dict_restores_existing_dir(self):
        """load_state_dict sets active dir to original when it exists."""
        with tempfile.TemporaryDirectory() as original_dir:
            rsw = ResumableSummaryWriter(log_dir="/tmp/new_run")
            rsw.load_state_dict({"log_dir": original_dir})
            assert rsw._active_log_dir == original_dir
            assert rsw._resumed is True

    def test_load_state_dict_falls_back_on_missing_dir(self):
        """load_state_dict falls back to new dir if original is gone."""
        rsw = ResumableSummaryWriter(log_dir="/tmp/new_run")
        rsw.load_state_dict({"log_dir": "/tmp/nonexistent_dir_12345"})
        assert rsw._active_log_dir == "/tmp/new_run"
        assert rsw._resumed is False

    def test_load_state_dict_handles_none(self):
        """load_state_dict handles None log_dir gracefully."""
        rsw = ResumableSummaryWriter(log_dir="/tmp/new_run")
        rsw.load_state_dict({"log_dir": None})
        assert rsw._active_log_dir == "/tmp/new_run"
        assert rsw._resumed is False

    def test_state_dict_roundtrip(self):
        """state_dict -> load_state_dict roundtrip preserves directory."""
        with tempfile.TemporaryDirectory() as log_dir:
            rsw1 = ResumableSummaryWriter(log_dir=log_dir)
            sd = rsw1.state_dict()

            rsw2 = ResumableSummaryWriter(log_dir="/tmp/new_run")
            rsw2.load_state_dict(sd)
            assert rsw2._active_log_dir == log_dir
            assert rsw2._resumed is True


class TestResumableSummaryWriterOnTrainBegin:
    """Tests for ResumableSummaryWriter.on_train_begin callback."""

    def _make_state(self, global_step=0):
        state = MagicMock()
        state.global_step = global_step
        return state

    def test_sets_purge_step_on_resume(self):
        """on_train_begin sets purge_step when resuming."""
        with tempfile.TemporaryDirectory() as original_dir:
            rsw = ResumableSummaryWriter(log_dir="/tmp/new")
            rsw.load_state_dict({"log_dir": original_dir})
            state = self._make_state(global_step=500)
            rsw.on_train_begin(MagicMock(), state, MagicMock())
            assert rsw._purge_step == 500

    def test_no_purge_step_on_fresh_start(self):
        """on_train_begin does not set purge_step for fresh runs."""
        rsw = ResumableSummaryWriter(log_dir="/tmp/new")
        state = self._make_state(global_step=0)
        rsw.on_train_begin(MagicMock(), state, MagicMock())
        assert rsw._purge_step is None

    def test_no_purge_step_when_not_resumed(self):
        """on_train_begin does not set purge_step when not resumed."""
        rsw = ResumableSummaryWriter(log_dir="/tmp/new")
        state = self._make_state(global_step=100)
        rsw.on_train_begin(MagicMock(), state, MagicMock())
        assert rsw._purge_step is None


class TestResumableSummaryWriterLazyConstruction:
    """Tests for lazy SummaryWriter construction and method proxying."""

    def test_ensure_writer_creates_writer(self):
        """_ensure_writer creates the underlying SummaryWriter."""
        with tempfile.TemporaryDirectory() as log_dir:
            rsw = ResumableSummaryWriter(log_dir=log_dir)
            writer = rsw._ensure_writer()
            assert writer is not None
            assert rsw._writer is writer
            rsw.close()

    def test_ensure_writer_returns_same_instance(self):
        """_ensure_writer returns the same writer on subsequent calls."""
        with tempfile.TemporaryDirectory() as log_dir:
            rsw = ResumableSummaryWriter(log_dir=log_dir)
            w1 = rsw._ensure_writer()
            w2 = rsw._ensure_writer()
            assert w1 is w2
            rsw.close()

    def test_ensure_writer_creates_directory(self):
        """_ensure_writer creates the log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as base:
            log_dir = os.path.join(base, "subdir", "logs")
            rsw = ResumableSummaryWriter(log_dir=log_dir)
            rsw._ensure_writer()
            assert os.path.isdir(log_dir)
            rsw.close()

    def test_ensure_writer_passes_purge_step(self):
        """_ensure_writer passes purge_step to SummaryWriter constructor."""
        with tempfile.TemporaryDirectory() as original_dir:
            rsw = ResumableSummaryWriter(log_dir="/tmp/new")
            rsw.load_state_dict({"log_dir": original_dir})
            state = MagicMock()
            state.global_step = 100
            rsw.on_train_begin(MagicMock(), state, MagicMock())

            with patch(
                "forgather.ml.trainer.callbacks.resumable_summary_writer.SummaryWriter"
            ) as MockSW:
                rsw._ensure_writer()
                MockSW.assert_called_once_with(original_dir, purge_step=100)

    def test_add_scalar_proxies(self):
        """add_scalar is forwarded to the underlying writer."""
        with tempfile.TemporaryDirectory() as log_dir:
            rsw = ResumableSummaryWriter(log_dir=log_dir)
            # Use a mock writer to verify proxy
            mock_writer = MagicMock()
            rsw._writer = mock_writer

            rsw.add_scalar("loss", 0.5, global_step=10)
            mock_writer.add_scalar.assert_called_once_with("loss", 0.5, global_step=10)

    def test_add_text_proxies(self):
        """add_text is forwarded to the underlying writer."""
        rsw = ResumableSummaryWriter(log_dir="/tmp/test")
        mock_writer = MagicMock()
        rsw._writer = mock_writer

        rsw.add_text("tag", "content")
        mock_writer.add_text.assert_called_once_with("tag", "content")

    def test_flush_noop_when_no_writer(self):
        """flush is a no-op when writer has not been created."""
        rsw = ResumableSummaryWriter(log_dir="/tmp/test")
        rsw.flush()  # Should not raise

    def test_flush_forwards_when_writer_exists(self):
        rsw = ResumableSummaryWriter(log_dir="/tmp/test")
        mock_writer = MagicMock()
        rsw._writer = mock_writer
        rsw.flush()
        mock_writer.flush.assert_called_once()

    def test_close_clears_writer(self):
        rsw = ResumableSummaryWriter(log_dir="/tmp/test")
        mock_writer = MagicMock()
        rsw._writer = mock_writer
        rsw.close()
        assert rsw._writer is None
        mock_writer.close.assert_called_once()

    def test_close_noop_when_no_writer(self):
        """close is safe when no writer exists."""
        rsw = ResumableSummaryWriter(log_dir="/tmp/test")
        rsw.close()  # Should not raise


# ---------------------------------------------------------------------------
# JsonLogger Stateful tests
# ---------------------------------------------------------------------------


class TestJsonLoggerStateful:
    """Tests for JsonLogger's Stateful protocol implementation."""

    def test_state_dict_initial(self):
        """state_dict returns initial values before any logging."""
        logger = JsonLogger()
        sd = logger.state_dict()
        assert sd == {"log_path": None, "last_step": -1}

    def test_state_dict_after_logging(self):
        """state_dict returns updated values after logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonLogger()
            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            state.global_step = 42
            state.epoch = 1.0
            control = TrainerControl()

            logger.on_train_begin(args, state, control)
            logger._write_log(state, {"loss": 0.5})

            sd = logger.state_dict()
            assert sd["log_path"] == os.path.join(tmpdir, "trainer_logs.json")
            assert sd["last_step"] == 42

            logger.close()

    def test_load_state_dict(self):
        """load_state_dict stores resume state."""
        logger = JsonLogger()
        logger.load_state_dict(
            {
                "log_path": "/some/path/trainer_logs.json",
                "last_step": 100,
            }
        )
        assert logger._original_log_path == "/some/path/trainer_logs.json"
        assert logger._resume_step == 100


class TestJsonLoggerResume:
    """Tests for JsonLogger resume from checkpoint."""

    def _write_log_file(self, path, records, close_properly=True):
        """Write a trainer_logs.json file with the given records."""
        with open(path, "w") as f:
            f.write("[\n")
            for i, record in enumerate(records):
                prefix = ",\n" if i > 0 else ""
                f.write(prefix + json.dumps(record))
            if close_properly:
                f.write("\n]")

    def test_resume_truncates_to_checkpoint_step(self):
        """On resume, entries after checkpoint step are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "trainer_logs.json")
            records = [
                {"timestamp": 1000, "global_step": 10, "epoch": 0.1, "loss": 2.5},
                {"timestamp": 1100, "global_step": 20, "epoch": 0.2, "loss": 2.0},
                {"timestamp": 1200, "global_step": 30, "epoch": 0.3, "loss": 1.5},
                {"timestamp": 1300, "global_step": 40, "epoch": 0.4, "loss": 1.0},
            ]
            self._write_log_file(log_path, records)

            # Create logger and simulate checkpoint resume at step 20
            logger = JsonLogger()
            logger.load_state_dict({"log_path": log_path, "last_step": 20})

            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            state.global_step = 20
            state.epoch = 0.2
            control = TrainerControl()

            logger.on_train_begin(args, state, control)

            # Write a new entry
            state.global_step = 21
            state.epoch = 0.21
            logger._write_log(state, {"loss": 1.9})
            logger.close()

            # Verify: should have steps 10, 20, 21
            with open(log_path) as f:
                data = json.load(f)
            assert len(data) == 3
            assert [r["global_step"] for r in data] == [10, 20, 21]

    def test_resume_with_unclosed_file(self):
        """Resume handles a log file that was not properly closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "trainer_logs.json")
            records = [
                {"timestamp": 1000, "global_step": 10, "epoch": 0.1, "loss": 2.5},
                {"timestamp": 1100, "global_step": 20, "epoch": 0.2, "loss": 2.0},
            ]
            self._write_log_file(log_path, records, close_properly=False)

            logger = JsonLogger()
            logger.load_state_dict({"log_path": log_path, "last_step": 10})

            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            state.global_step = 10
            state.epoch = 0.1
            control = TrainerControl()

            logger.on_train_begin(args, state, control)
            logger.close()

            with open(log_path) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["global_step"] == 10

    def test_resume_with_missing_file_falls_back(self):
        """If original log file is missing, start fresh in new dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonLogger()
            logger.load_state_dict(
                {
                    "log_path": "/nonexistent/trainer_logs.json",
                    "last_step": 50,
                }
            )

            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            state.global_step = 50
            state.epoch = 1.0
            control = TrainerControl()

            logger.on_train_begin(args, state, control)

            # Should have started fresh
            assert logger.log_path == os.path.join(tmpdir, "trainer_logs.json")
            assert logger.log_file is not None

            logger._write_log(state, {"loss": 0.5})
            logger.close()

            with open(logger.log_path) as f:
                data = json.load(f)
            assert len(data) == 1

    def test_resume_truncate_to_step_zero(self):
        """Resuming at step 0 removes all entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "trainer_logs.json")
            records = [
                {"timestamp": 1000, "global_step": 10, "epoch": 0.1, "loss": 2.5},
            ]
            self._write_log_file(log_path, records)

            logger = JsonLogger()
            logger.load_state_dict({"log_path": log_path, "last_step": 0})

            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            state.global_step = 0
            state.epoch = 0.0
            control = TrainerControl()

            logger.on_train_begin(args, state, control)
            logger.close()

            with open(log_path) as f:
                data = json.load(f)
            assert len(data) == 0

    def test_resume_with_corrupted_file_recovers_gracefully(self):
        """If log file is completely corrupted, parser returns empty list and
        logger continues with an empty file (no entries matched the step filter)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "trainer_logs.json")
            with open(log_path, "w") as f:
                f.write("this is completely invalid")

            logger = JsonLogger()
            logger.load_state_dict({"log_path": log_path, "last_step": 10})

            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            state.global_step = 10
            state.epoch = 0.1
            control = TrainerControl()

            logger.on_train_begin(args, state, control)

            # Logger should be usable (0 records kept from corrupted file)
            assert logger.log_file is not None

            logger._write_log(state, {"loss": 1.0})
            logger.close()

            with open(log_path) as f:
                data = json.load(f)
            assert len(data) == 1

    def test_resume_with_read_error_backs_up(self):
        """If reading the log file raises an exception, back it up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "trainer_logs.json")
            with open(log_path, "w") as f:
                f.write('[{"global_step": 1}]')

            logger = JsonLogger()
            logger.load_state_dict({"log_path": log_path, "last_step": 10})

            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            state.global_step = 10
            state.epoch = 0.1
            control = TrainerControl()

            # Force an exception during truncation
            with patch(
                "builtins.open",
                side_effect=[
                    OSError("simulated read error"),  # First open (read) fails
                    open(log_path + ".bak", "w"),  # Backup rename
                    open(log_path, "w"),  # Fresh start
                ],
            ):
                # We need a different approach -- mock at a higher level
                pass

            # Simpler approach: make _parse_json_log raise
            with patch(
                "forgather.ml.trainer.callbacks.json_logger._parse_json_log",
                side_effect=RuntimeError("parse failed"),
            ):
                logger.on_train_begin(args, state, control)

            assert os.path.exists(log_path + ".bak")
            assert logger.log_file is not None
            logger.close()

    def test_resume_multiple_times(self):
        """Multiple resume cycles work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "trainer_logs.json")

            # First training run
            logger1 = JsonLogger()
            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            control = TrainerControl()

            logger1.on_train_begin(args, state, control)
            for step in [10, 20, 30]:
                state.global_step = step
                state.epoch = step / 100
                logger1._write_log(state, {"loss": 3.0 - step * 0.05})
            sd1 = logger1.state_dict()
            logger1.close()

            # Second run (resume from step 20)
            logger2 = JsonLogger()
            logger2.load_state_dict({"log_path": sd1["log_path"], "last_step": 20})
            logger2.on_train_begin(args, state, control)
            for step in [21, 22]:
                state.global_step = step
                state.epoch = step / 100
                logger2._write_log(state, {"loss": 2.0 - step * 0.01})
            sd2 = logger2.state_dict()
            logger2.close()

            with open(log_path) as f:
                data = json.load(f)
            assert [r["global_step"] for r in data] == [10, 20, 21, 22]

            # Third run (resume from step 21)
            logger3 = JsonLogger()
            logger3.load_state_dict({"log_path": sd2["log_path"], "last_step": 21})
            logger3.on_train_begin(args, state, control)
            state.global_step = 22
            state.epoch = 0.22
            logger3._write_log(state, {"loss": 1.5})
            logger3.close()

            with open(log_path) as f:
                data = json.load(f)
            assert [r["global_step"] for r in data] == [10, 20, 21, 22]

    def test_fresh_start_no_state(self):
        """Without load_state_dict, logger starts fresh as before."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonLogger()
            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            state.global_step = 0
            state.epoch = 0.0
            control = TrainerControl()

            logger.on_train_begin(args, state, control)
            logger._write_log(state, {"loss": 1.0})
            logger.close()

            log_path = os.path.join(tmpdir, "trainer_logs.json")
            with open(log_path) as f:
                data = json.load(f)
            assert len(data) == 1

    def test_non_zero_rank_skips(self):
        """Non-zero rank processes do not open any files on resume."""
        logger = JsonLogger()
        logger.load_state_dict(
            {
                "log_path": "/some/path/trainer_logs.json",
                "last_step": 50,
            }
        )

        args = MagicMock()
        args.logging_dir = "/some/dir"
        state = MagicMock()
        state.is_world_process_zero = False
        control = TrainerControl()

        logger.on_train_begin(args, state, control)
        assert logger.log_file is None


# ---------------------------------------------------------------------------
# Integration-style: ResumableSummaryWriter + JsonLogger full lifecycle
# ---------------------------------------------------------------------------


class TestResumableLoggingLifecycle:
    """End-to-end tests simulating the full train -> checkpoint -> resume cycle."""

    def test_json_logger_state_dict_roundtrip_lifecycle(self):
        """Full lifecycle: train, save state, resume, continue logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # --- First training session ---
            logger1 = JsonLogger()
            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            control = TrainerControl()

            logger1.on_train_begin(args, state, control)

            for step in range(1, 6):
                state.global_step = step * 10
                state.epoch = step * 0.1
                logger1.on_log(args, state, control, logs={"loss": 3.0 - step * 0.3})

            # "Checkpoint" at step 30
            state_dict = logger1.state_dict()
            assert state_dict["last_step"] == 50  # Last step logged
            logger1.close()

            # --- Resume session ---
            logger2 = JsonLogger()
            logger2.load_state_dict(
                {
                    "log_path": state_dict["log_path"],
                    "last_step": 30,  # Checkpoint was at step 30
                }
            )

            args2 = MagicMock()
            args2.logging_dir = os.path.join(tmpdir, "new_run")
            state.global_step = 30
            state.epoch = 0.3
            logger2.on_train_begin(args2, state, control)

            # Log should have been truncated to step 30
            # Continue logging
            for step in [31, 32, 33]:
                state.global_step = step
                state.epoch = step / 100
                logger2.on_log(args2, state, control, logs={"loss": 1.0})

            logger2.on_train_end(args2, state, control)

            # Verify final file content
            with open(state_dict["log_path"]) as f:
                data = json.load(f)

            steps = [r["global_step"] for r in data]
            assert steps == [10, 20, 30, 31, 32, 33]

    def test_resumable_summary_writer_lifecycle(self):
        """Full lifecycle for ResumableSummaryWriter with mocked SummaryWriter."""
        with tempfile.TemporaryDirectory() as original_dir:
            # --- First session ---
            rsw = ResumableSummaryWriter(log_dir=original_dir)
            state = MagicMock()
            state.global_step = 0
            rsw.on_train_begin(MagicMock(), state, MagicMock())

            # Simulate logging (triggers lazy construction)
            with patch(
                "forgather.ml.trainer.callbacks.resumable_summary_writer.SummaryWriter"
            ) as MockSW:
                mock_instance = MagicMock()
                MockSW.return_value = mock_instance
                rsw.add_scalar("loss", 1.0, global_step=10)
                MockSW.assert_called_once_with(original_dir)
                mock_instance.add_scalar.assert_called_once()

            # Save state
            sd = rsw.state_dict()
            assert sd["log_dir"] == original_dir
            rsw.close()

            # --- Resume session ---
            rsw2 = ResumableSummaryWriter(log_dir="/tmp/new_run")
            rsw2.load_state_dict(sd)
            assert rsw2._active_log_dir == original_dir

            state.global_step = 100
            rsw2.on_train_begin(MagicMock(), state, MagicMock())
            assert rsw2._purge_step == 100

            with patch(
                "forgather.ml.trainer.callbacks.resumable_summary_writer.SummaryWriter"
            ) as MockSW:
                mock_instance = MagicMock()
                MockSW.return_value = mock_instance
                rsw2.add_scalar("loss", 0.5, global_step=110)
                MockSW.assert_called_once_with(original_dir, purge_step=100)
            rsw2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
