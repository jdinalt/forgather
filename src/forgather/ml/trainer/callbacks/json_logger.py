import datetime
import json
import logging
import os

from torch.distributed.checkpoint.stateful import Stateful

from ..trainer_types import (
    MinimalTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

logger = logging.getLogger(__name__)


class JsonLogger(TrainerCallback, Stateful):
    """
    A JSON logger callback that writes training metrics to a JSON file.

    Writes a JSON record (with UTC timestamp, global_step, epoch, and all
    reported metrics) each time ``on_log`` or ``on_evaluate`` is called.

    Implements the ``Stateful`` protocol so that the log file path and last
    written step are saved with checkpoints.  When training resumes from a
    checkpoint, the logger reopens the original file, truncates any entries
    recorded after the checkpoint step, and continues appending.
    """

    def __init__(self, **kwargs):
        """
        The contents of kwargs will be recorded when training starts
        """
        super().__init__()
        self.log_file = None
        self.log_path = None
        self.kwargs = kwargs
        self.prefix = ""
        self._last_step = -1

        # Set by load_state_dict when resuming from checkpoint
        self._original_log_path: str | None = None
        self._resume_step: int | None = None

    def __del__(self):
        self.close()

    def close(self):
        if self.log_file is not None:
            self.log_file.write("\n]")
            self.log_file.close()
            self.log_file = None

    # -- Stateful protocol --------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "log_path": self.log_path,
            "last_step": self._last_step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._original_log_path = state_dict.get("log_path")
        self._resume_step = state_dict.get("last_step", -1)
        logger.debug(
            "JsonLogger: loaded state (path=%s, step=%s)",
            self._original_log_path,
            self._resume_step,
        )

    # -- Callback hooks -----------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or args.logging_dir is None:
            return

        if self._original_log_path and os.path.isfile(self._original_log_path):
            self.log_path = self._original_log_path
            self._truncate_and_reopen()
        else:
            if self._original_log_path:
                logger.warning(
                    "JsonLogger: original log file not found (%s), "
                    "starting fresh in %s",
                    self._original_log_path,
                    args.logging_dir,
                )
            os.makedirs(args.logging_dir, exist_ok=True)
            self.log_path = os.path.join(args.logging_dir, "trainer_logs.json")
            self.log_file = open(self.log_path, "x")
            self.log_file.write("[\n")

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})
        if self.log_file is None:
            return
        self._write_log(state, metrics)

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        if self.log_file is None:
            return
        self._write_log(state, logs)

    def on_train_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.close()

    # -- Internal -----------------------------------------------------------

    def _write_log(self, state, data: dict):
        assert self.log_file is not None
        new_fields = dict(
            timestamp=datetime.datetime.now(datetime.UTC).timestamp(),
            global_step=state.global_step,
            epoch=state.epoch,
        )
        self.log_file.write(self.prefix + json.dumps(new_fields | data))
        self.prefix = ",\n"
        self._last_step = state.global_step

    def _truncate_and_reopen(self):
        """Reopen the original JSON log file and truncate entries after
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
                "JsonLogger: resuming %s, kept %d/%d records (up to step %d)",
                log_path,
                len(kept),
                len(records),
                resume_step,
            )

            self.log_file = open(log_path, "w")
            self.log_file.write("[\n")
            self.prefix = ""
            for record in kept:
                self.log_file.write(self.prefix + json.dumps(record))
                self.prefix = ",\n"
            self.log_file.flush()

        except Exception as e:
            logger.warning(
                "JsonLogger: failed to parse/truncate %s: %s. "
                "Backing up and starting fresh.",
                log_path,
                e,
            )
            backup = log_path + ".bak"
            try:
                os.rename(log_path, backup)
                logger.info("JsonLogger: backed up corrupted file to %s", backup)
            except OSError:
                pass
            self.log_file = open(log_path, "w")
            self.log_file.write("[\n")
            self.prefix = ""


def _parse_json_log(content: str) -> list:
    """Parse a trainer_logs.json file, handling common corruption modes.

    Handles:
    - Complete valid JSON array
    - Missing closing bracket (unclean shutdown)
    - Trailing comma before closing bracket
    - Partially written last record
    """
    content = content.strip()
    if not content:
        return []

    # Try direct parse first
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try adding closing bracket (file was not closed properly)
    if not content.endswith("]"):
        trimmed = content.rstrip(",\n\r\t ")
        try:
            result = json.loads(trimmed + "\n]")
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Last record may be partially written; find last complete record
        last_brace = trimmed.rfind("}")
        if last_brace > 0:
            attempt = trimmed[: last_brace + 1].rstrip(",\n\r\t ") + "\n]"
            try:
                result = json.loads(attempt)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

    # Handle trailing comma inside otherwise valid array
    if content.endswith("]"):
        # Remove trailing comma before ]
        inner = content[:-1].rstrip(",\n\r\t ") + "\n]"
        try:
            result = json.loads(inner)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning("JsonLogger: could not parse JSON log, returning empty list")
    return []
