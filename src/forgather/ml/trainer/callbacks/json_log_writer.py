"""Reusable JSON array log file writer with checkpoint support."""

import datetime
import json
import logging
import os

from torch.distributed.checkpoint.stateful import Stateful

from .json_logger import _parse_json_log

logger = logging.getLogger(__name__)


class JsonLogWriter(Stateful):
    """JSON array log file writer with checkpoint resume support.

    Writes records as a JSON array to a file. Implements the ``Stateful``
    protocol so the log file path and last written step are saved with
    checkpoints. On resume, the file is truncated to the checkpoint step
    and appending continues.

    This is a building block for callbacks that need structured log files
    with checkpoint coordination. It mirrors the file management logic of
    ``JsonLogger`` but is decoupled from the callback interface.
    """

    def __init__(self, filename: str):
        """
        Parameters
        ----------
        filename : str
            Relative filename (e.g., ``"parameter_norms.json"``). Will be
            created inside the trainer's ``logging_dir``.
        """
        self.filename = filename
        self.log_file = None
        self.log_path: str | None = None
        self._prefix = ""
        self._last_step = -1

        # Set by load_state_dict when resuming from checkpoint
        self._original_log_path: str | None = None
        self._resume_step: int | None = None

    def __del__(self):
        self.close()

    # -- Public API -----------------------------------------------------------

    def open(self, logging_dir: str) -> None:
        """Open the log file for writing.

        If resuming from a checkpoint, reopens the original file and
        truncates entries after the checkpoint step. Otherwise creates
        a new file.
        """
        if self._original_log_path and os.path.isfile(self._original_log_path):
            self.log_path = self._original_log_path
            self._truncate_and_reopen()
        else:
            if self._original_log_path:
                logger.warning(
                    "%s: original log file not found (%s), " "starting fresh in %s",
                    self.filename,
                    self._original_log_path,
                    logging_dir,
                )
            os.makedirs(logging_dir, exist_ok=True)
            self.log_path = os.path.join(logging_dir, self.filename)
            self.log_file = open(self.log_path, "x")
            self.log_file.write("[\n")

    def write_record(self, global_step: int, epoch: float, data: dict) -> None:
        """Write a JSON record with timestamp, step, epoch, and data."""
        if self.log_file is None:
            return
        record = dict(
            timestamp=datetime.datetime.now(datetime.UTC).timestamp(),
            global_step=global_step,
            epoch=epoch,
        )
        record.update(data)
        self.log_file.write(self._prefix + json.dumps(record))
        self._prefix = ",\n"
        self._last_step = global_step
        self.log_file.flush()

    def close(self) -> None:
        """Close the log file, writing the closing bracket."""
        if self.log_file is not None:
            self.log_file.write("\n]")
            self.log_file.close()
            self.log_file = None

    @property
    def is_open(self) -> bool:
        return self.log_file is not None

    # -- Stateful protocol ----------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "log_path": self.log_path,
            "last_step": self._last_step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._original_log_path = state_dict.get("log_path")
        self._resume_step = state_dict.get("last_step", -1)
        logger.debug(
            "%s: loaded state (path=%s, step=%s)",
            self.filename,
            self._original_log_path,
            self._resume_step,
        )

    # -- Internal -------------------------------------------------------------

    def _truncate_and_reopen(self):
        """Reopen the original log file and truncate entries after
        the checkpoint step."""
        log_path = self.log_path
        assert log_path is not None
        resume_step = self._resume_step if self._resume_step is not None else -1

        try:
            with open(log_path, "r") as f:
                content = f.read()

            records = _parse_json_log(content)
            kept = [r for r in records if r.get("global_step", 0) <= resume_step]

            logger.info(
                "%s: resuming %s, kept %d/%d records (up to step %d)",
                self.filename,
                log_path,
                len(kept),
                len(records),
                resume_step,
            )

            self.log_file = open(log_path, "w")
            self.log_file.write("[\n")
            self._prefix = ""
            for record in kept:
                self.log_file.write(self._prefix + json.dumps(record))
                self._prefix = ",\n"
            self.log_file.flush()

        except Exception as e:
            logger.warning(
                "%s: failed to parse/truncate %s: %s. "
                "Backing up and starting fresh.",
                self.filename,
                log_path,
                e,
            )
            backup = log_path + ".bak"
            try:
                os.rename(log_path, backup)
                logger.info("%s: backed up corrupted file to %s", self.filename, backup)
            except OSError:
                pass
            self.log_file = open(log_path, "w")
            self.log_file.write("[\n")
            self._prefix = ""
