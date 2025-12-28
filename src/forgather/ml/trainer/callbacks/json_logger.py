import datetime
import json
import os

from forgather.ml.trainer.logging import format_train_info

from ..trainer_types import (
    MinimalTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)


class JsonLogger(TrainerCallback):
    """
    A very simple JSON  logger callback

    It just writes a JSON record to a file, adding a UTC timestamp, each time on_log or on_evaluate are called.
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

    def __del__(self):
        self.close()

    def close(self):
        if self.log_file is not None:
            self.log_file.write("\n]")
            self.log_file.close()
            self.log_file = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or args.logging_dir is None:
            return
        os.makedirs(args.logging_dir, exist_ok=True)
        self.log_path = os.path.join(args.logging_dir, "trainer_logs.json")
        self.log_file = open(self.log_path, "x")
        self.log_file.write("[\n")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.log_file is None:
            return
        self._write_log(state, metrics)

    def on_log(self, args, state, control, logs, **kwargs):
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

    def _write_log(self, state, data: dict):
        assert self.log_file is not None
        new_fields = dict(
            timestamp=datetime.datetime.utcnow().timestamp(),
            global_step=state.global_step,
            epoch=state.epoch,
        )
        self.log_file.write(self.prefix + json.dumps(new_fields | data))
        self.prefix = ",\n"
