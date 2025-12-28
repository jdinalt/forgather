# A light-weight replacement for the HF Trainer class
import logging
import sys
from io import TextIOBase
from typing import Literal, Optional

from tqdm.auto import tqdm

from forgather.ml.trainer.logging import (
    format_eval_log,
    format_mapping,
    format_timestamp,
    format_train_info,
    format_train_log,
    get_env_type,
)

from ..trainer_types import MinimalTrainingArguments, TrainerControl, TrainerState

OutputStream = TextIOBase | Literal["stderr", "stdout"]


class ProgressCallback:
    """
    A TQDM progress-bar callback class based upon:
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py

    This will fallback to writing to stdout, if
    """

    def __init__(
        self,
        use_tqdm: Optional[bool] = None,
        output_stream: Optional[OutputStream] = None,
    ):
        """
        Args:
            use_tqdm: If True, use TQDM; else if False, use "logging; else auto select
            output_stream: The output stream to use, if using "logging"
        """
        super().__init__()
        self.train_progress_bar = None
        self.eval_progress_bar = None
        if use_tqdm is None:
            self.use_tqdm = get_env_type() != "file"
        else:
            self.use_tqdm = use_tqdm

        if not self.use_tqdm:
            self.logger = logging.getLogger("progress_logger")
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

            console_handler = logging.StreamHandler(
                self._get_output_stream(output_stream)
            )
            log_format = logging.Formatter(
                fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(log_format)
            self.logger.addHandler(console_handler)

    @staticmethod
    def _get_output_stream(output_stream: OutputStream) -> TextIOBase:
        if output_stream is None:
            return sys.stdout
        elif isinstance(output_stream, TextIOBase):
            return output_stream
        else:
            assert isinstance(output_stream, str)
            if output_stream == "stderr":
                return sys.stderr
            elif output_stream == "stdout":
                return sys.stdout
            else:
                raise ValueError("Must be one of 'stderr' or 'stdout'")

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.last_step = state.global_step
        if self.use_tqdm:
            self.train_progress_bar = tqdm(
                initial=state.global_step, total=state.max_steps, dynamic_ncols=True
            )

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.use_tqdm:
            self.train_progress_bar.close()
            self.train_progress_bar = None

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.use_tqdm:
            self.train_progress_bar.update(state.global_step - self.last_step)
        self.last_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.use_tqdm:
            if self.eval_progress_bar is None:
                max_eval_steps = getattr(state, "max_eval_steps", -1)
                total = max(len(eval_dataloader), max_eval_steps, 1)
                self.eval_progress_bar = tqdm(
                    initial=1,
                    total=total,
                    leave=self.train_progress_bar is None,
                    dynamic_ncols=True,
                )
            else:
                self.eval_progress_bar.update(1)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.use_tqdm:
            if self.eval_progress_bar is not None:
                self.eval_progress_bar.write(
                    format_timestamp() + format_eval_log(state, metrics)
                )
                self.eval_progress_bar.close()
                self.eval_progress_bar = None
        else:
            self.logger.info(format_eval_log(state, metrics))

    def on_log(self, args, state, control, logs, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.use_tqdm:
            if self.train_progress_bar is not None:
                self.train_progress_bar.write(
                    format_timestamp() + format_train_log(state, logs)
                )
        else:
            self.logger.info(format_train_log(state, logs))


class InfoCallback:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = logging.getLogger("info_logger")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.logger.propagate = False

        console_handler = logging.StreamHandler(sys.stdout)
        log_format = logging.Formatter(fmt="[%(levelname)s|%(name)s] %(message)s")
        console_handler.setFormatter(log_format)
        self.logger.addHandler(console_handler)

    def on_train_begin(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        info, extra_info = format_train_info(args, state, control, **kwargs)
        self.logger.info("\n" + format_mapping(info))
        self.logger.debug("\n" + format_mapping(extra_info))
