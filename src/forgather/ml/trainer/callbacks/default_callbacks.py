# A light-weight replacement for the HF Trainer class
import logging
import sys
import time
from io import TextIOBase
from typing import Literal, Optional, cast

from tqdm.auto import tqdm

from forgather.ml.trainer.logging import (
    format_eval_log,
    format_final_metrics,
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

    Controls which metrics are displayed in console logs during training.
    All metrics are still logged to JsonLogger regardless of display settings.

    Speed metrics (tokens_per_second, mfu) are computed from the wall-clock delta
    between consecutive log steps and the delta in token/FLOP counts, giving
    throughput over the logged interval rather than a cumulative average.
    """

    def __init__(
        self,
        use_tqdm: Optional[bool] = None,
        output_stream: Optional[OutputStream] = None,
        show_loss: bool = True,
        show_grad_norm: bool = True,
        show_learning_rate: bool = True,
        show_tokens: bool = False,
        show_epoch: bool = True,
        show_tokens_per_second: bool = False,
        peak_hardware_flops: Optional[float] = None,
        show_peak_memory: bool = False,
    ):
        """
        Args:
            use_tqdm: If True, use TQDM; else if False, use logging; else auto select
            output_stream: The output stream to use, if using logging
            show_loss: Display loss in console logs (default: True)
            show_grad_norm: Display gradient norm in console logs (default: True)
            show_learning_rate: Display learning rate in console logs (default: True)
            show_tokens: Display token count for the log interval (default: False)
            show_epoch: Display epoch in console logs (default: True)
            show_tokens_per_second: Display tokens/sec computed from the log interval
                wall-clock time and token delta (default: False)
            peak_hardware_flops: Aggregate peak BF16 FLOP/s across all GPUs used in
                training, used to compute MFU (Model FLOPs Utilization). Must be the
                total across all ranks since total_flos accounts for tokens processed
                across all ranks. If provided, MFU is displayed when
                show_tokens_per_second is True.
                Example values (dense BF16, FP32 accumulate):
                  Single RTX 4090:  165.2e12
                  Single RTX 3090:   71.2e12
                  4x RTX 4090:      660.8e12
                  A100 SXM:         312e12
                  H100 SXM:         989e12
                (default: None, MFU not computed)
            show_peak_memory: Display peak CUDA memory allocated on the logging rank
                since the last log step, formatted as GiB. Peak stats are reset after
                each read, so the value reflects the interval high-water mark rather
                than a cumulative maximum. Requires CUDA. (default: False)
        """
        super().__init__()
        self.train_progress_bar = None
        self.eval_progress_bar = None
        self.show_loss = show_loss
        self.show_grad_norm = show_grad_norm
        self.show_learning_rate = show_learning_rate
        self.show_tokens = show_tokens
        self.show_epoch = show_epoch
        self.show_tokens_per_second = show_tokens_per_second
        self.peak_hardware_flops = peak_hardware_flops
        self.show_peak_memory = show_peak_memory

        # Tracking for per-interval speed metrics
        self._last_log_time: Optional[float] = None
        self._last_total_flos: float = 0.0

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
    def _get_output_stream(output_stream: Optional[OutputStream]) -> TextIOBase:
        if output_stream is None:
            # sys.stdout satisfies the TextIOBase interface at runtime
            return cast(TextIOBase, sys.stdout)  # type: ignore[return-value]
        elif isinstance(output_stream, TextIOBase):
            return output_stream
        else:
            assert isinstance(output_stream, str)
            if output_stream == "stderr":
                # sys.stderr satisfies the TextIOBase interface at runtime
                return cast(TextIOBase, sys.stderr)  # type: ignore[return-value]
            elif output_stream == "stdout":
                # sys.stdout satisfies the TextIOBase interface at runtime
                return cast(TextIOBase, sys.stdout)  # type: ignore[return-value]
            else:
                raise ValueError("Must be one of 'stderr' or 'stdout'")

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.last_step = state.global_step
        # Initialize speed metric tracking; use state values to handle checkpoint resume
        self._last_log_time = time.monotonic()
        self._last_total_flos = state.total_flos
        if self.use_tqdm:
            self.train_progress_bar = tqdm(
                initial=state.global_step,
                smoothing=0.03,
                total=state.max_steps,
                dynamic_ncols=True,
            )

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.use_tqdm:
            if self.train_progress_bar is not None:
                self.train_progress_bar.close()
            self.train_progress_bar = None

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.use_tqdm:
            if self.train_progress_bar is not None:
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

        # Final training metrics get their own formatted summary
        if "train_runtime" in logs:
            summary = format_final_metrics(logs)
            if self.use_tqdm:
                if self.train_progress_bar is not None:
                    self.train_progress_bar.write(format_timestamp() + summary)
                else:
                    tqdm.write(format_timestamp() + summary)
            else:
                self.logger.info(summary)
            return

        now = time.monotonic()

        # Filter logs based on display options
        display_logs = {}
        if self.show_epoch and "epoch" in logs:
            display_logs["epoch"] = logs["epoch"]
        if self.show_loss and "loss" in logs:
            display_logs["loss"] = logs["loss"]
        if self.show_grad_norm and "grad_norm" in logs:
            display_logs["grad_norm"] = logs["grad_norm"]
        if self.show_grad_norm and "max_grad_norm" in logs:
            display_logs["max_grad_norm"] = logs["max_grad_norm"]
        if self.show_learning_rate and "learning_rate" in logs:
            display_logs["learning_rate"] = logs["learning_rate"]
        if self.show_tokens and "tokens" in logs:
            display_logs["tokens"] = logs["tokens"]

        # Compute per-interval speed metrics from delta since last log step
        if (
            self.show_tokens_per_second
            and self._last_log_time is not None
            and "tokens" in logs
        ):
            elapsed = now - self._last_log_time
            if elapsed > 0:
                delta_tokens = logs["tokens"]
                tps = delta_tokens / elapsed
                display_logs["tok/s"] = round(tps)

                # MFU requires knowing the hardware's peak FLOP/s
                if self.peak_hardware_flops is not None and "total_flos" in logs:
                    delta_flos = logs["total_flos"] - self._last_total_flos
                    if delta_flos > 0:
                        achieved_flops = delta_flos / elapsed
                        mfu = achieved_flops / self.peak_hardware_flops
                        display_logs["mfu"] = f"{mfu:.1%}"

        # Peak CUDA memory on the logging rank since last log step.
        # The value is captured and reset once in _log_step before on_log is dispatched.
        if self.show_peak_memory:
            peak_mem = logs.get("peak_mem_allocated")
            if peak_mem is not None:
                gib = 1024**3
                display_logs["peak_mem"] = f"{peak_mem / gib:.3f} GiB"

        # Update tracking for next interval
        self._last_log_time = now
        self._last_total_flos = logs.get("total_flos", self._last_total_flos)

        if self.use_tqdm:
            if self.train_progress_bar is not None:
                # Update steps, if max steps changes
                if self.train_progress_bar.total != state.max_steps:
                    self.train_progress_bar.total = state.max_steps
                    self.train_progress_bar.refresh()
                self.train_progress_bar.write(
                    format_timestamp() + format_train_log(state, display_logs)
                )
        else:
            self.logger.info(format_train_log(state, display_logs))


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
