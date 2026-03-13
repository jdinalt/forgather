# A light-weight replacement for the HF Trainer class
import logging
import sys
import time
from io import TextIOBase
from typing import Literal, Optional, cast

from tqdm.auto import tqdm

from forgather.ml.trainer.logging import (
    ColumnSpec,
    FinalMetricSpec,
    default_final_metrics,
    default_step_columns,
    format_eval_log,
    format_final_metrics,
    format_mapping,
    format_timestamp,
    format_train_header,
    format_train_info,
    format_train_log,
    get_env_type,
)

from ..trainer_types import MinimalTrainingArguments, TrainerControl, TrainerState

OutputStream = TextIOBase | Literal["stderr", "stdout"]


def _normalize_columns(columns: dict | list) -> list[ColumnSpec]:
    """Convert a dict or list of column specs into a list of ColumnSpec.

    Dict form (preferred for YAML)::

        {"loss": {"label": "loss", "width": 8, "fmt": ".5f"},
         "grad_norm": None}   # None erases the column

    List form::

        [ColumnSpec("loss", ...), {"key": "loss", "label": "loss", ...}]
    """
    if isinstance(columns, dict):
        result: list[ColumnSpec] = []
        for key, spec in columns.items():
            if spec is None:
                continue
            if isinstance(spec, ColumnSpec):
                result.append(spec)
            elif isinstance(spec, dict):
                spec = dict(spec)  # copy to avoid mutating the original
                spec["key"] = key
                result.append(ColumnSpec(**spec))
            else:
                raise TypeError(
                    f"step_columns[{key!r}]: expected dict or None, got {type(spec)}"
                )
        return result
    else:
        return [c if isinstance(c, ColumnSpec) else ColumnSpec(**c) for c in columns]


class ProgressCallback:
    """
    A TQDM progress-bar callback class based upon:
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py

    Controls which metrics are displayed in console logs during training
    via configurable column specifications.  All metrics are still logged
    to JsonLogger regardless of display settings.

    Token throughput (tok/s) is computed from the wall-clock delta between
    consecutive log steps, capturing real end-to-end throughput including
    optimizer, data loading, and other overhead.

    FLOPs and MFU are computed from accumulated pure training step time
    (on_step_begin to on_step_end), excluding evaluation and other non-
    forward/backward time, giving a measure of hardware utilization during
    the compute-bound portion of training.
    """

    def __init__(
        self,
        use_tqdm: Optional[bool] = None,
        output_stream: Optional[OutputStream] = None,
        step_columns: Optional[dict | list] = None,
        final_metrics: Optional[list] = None,
        peak_hardware_flops: Optional[float] = None,
        header_interval: int = 20,
    ):
        """
        Args:
            use_tqdm: If True, use TQDM; else if False, use logging; else auto select
            output_stream: The output stream to use, if using logging
            step_columns: Column specifications controlling which metrics are
                displayed and how they are formatted.  Accepts either:

                * A **dict** mapping metric keys to column spec dicts
                  (label, width, fmt).  This is the recommended form for
                  YAML configs because child templates can override or
                  erase columns via the standard merge pattern.  Set a
                  key to ``null`` / ``~`` to remove that column.
                * A **list** of ``ColumnSpec`` objects or dicts (must
                  include a ``key`` field).

                When ``None``, uses ``default_step_columns()``.
                Column order follows insertion order.  Only columns whose
                key appears in the current log entry are shown.
            final_metrics: Metric specifications for the end-of-training
                summary.  Each entry is a ``FinalMetricSpec`` or a dict with
                FinalMetricSpec fields.  When ``None``, uses
                ``default_final_metrics()``.
            peak_hardware_flops: Aggregate peak BF16 FLOP/s across all GPUs used in
                training, used to compute MFU (Model FLOPs Utilization). Must be the
                total across all ranks since total_flos accounts for tokens processed
                across all ranks. If provided, MFU is computed and available for
                display when a ``mfu`` column is in step_columns.
                Example values (dense BF16, FP32 accumulate):
                  Single RTX 4090:  165.2e12
                  Single RTX 3090:   71.2e12
                  4x RTX 4090:      660.8e12
                  A100 SXM:         312e12
                  H100 SXM:         989e12
                (default: None, MFU not computed)
            header_interval: Print a column header row every this many log steps, and
                also whenever the set of active columns changes. (default: 20)
        """
        super().__init__()
        self.train_progress_bar = None
        self.eval_progress_bar = None
        self.peak_hardware_flops = peak_hardware_flops
        self.header_interval = header_interval

        # Normalize step_columns: accept dict (recommended for YAML),
        # list, or None (use defaults).
        if step_columns is not None:
            self.step_columns: list[ColumnSpec] = _normalize_columns(step_columns)
        else:
            self.step_columns = default_step_columns()

        if final_metrics is not None:
            self.final_metrics: list[FinalMetricSpec] | None = [
                m if isinstance(m, FinalMetricSpec) else FinalMetricSpec(**m)
                for m in final_metrics
            ]
        else:
            self.final_metrics = None

        self._column_keys: frozenset[str] = frozenset(c.key for c in self.step_columns)

        # Tracking for per-interval speed metrics.
        # _last_log_time records the wall-clock time at each log step, used for
        # tok/s which should reflect real end-to-end throughput.
        # _step_start_time is set at on_step_begin and cleared at on_step_end.
        # _accumulated_train_time sums pure training step durations between log calls,
        # used only for FLOPs/MFU (excludes evaluation, optimizer, data loading time).
        self._last_log_time: Optional[float] = None
        self._step_start_time: Optional[float] = None
        self._accumulated_train_time: float = 0.0
        self._last_total_flos: float = 0.0

        # Column header tracking: print header every header_interval rows and
        # whenever the active column set changes.
        self._log_row_count: int = 0
        self._last_active_keys: frozenset[str] = frozenset()

        # Remember actual eval steps from previous run for accurate progress bar
        self._last_eval_steps: Optional[int] = None

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
        self._last_log_time = None
        self._last_total_flos = state.total_flos
        self._accumulated_train_time = 0.0
        self._step_start_time = None
        self._log_row_count = 0
        self._last_active_keys = frozenset()
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

    def on_step_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._step_start_time = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        if self._step_start_time is not None:
            self._accumulated_train_time += time.monotonic() - self._step_start_time
            self._step_start_time = None

        if self.use_tqdm:
            self.train_progress_bar.update(state.global_step - self.last_step)
        self.last_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.use_tqdm:
            if self.eval_progress_bar is None:
                if self._last_eval_steps is not None:
                    total = self._last_eval_steps
                else:
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
                # Remember actual step count for next eval's progress bar
                self._last_eval_steps = self.eval_progress_bar.n
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
            summary = format_final_metrics(logs, self.final_metrics)
            if self.use_tqdm:
                if self.train_progress_bar is not None:
                    self.train_progress_bar.write(format_timestamp() + summary)
                else:
                    tqdm.write(format_timestamp() + summary)
            else:
                self.logger.info(summary)
            return

        # Build a merged metrics dict: raw log entries + computed metrics.
        all_metrics: dict = dict(logs)

        # Compute per-interval speed metrics.
        # tok/s uses wall-clock time between log steps for real end-to-end throughput
        # (includes optimizer, data loading, and all other overhead).
        # FLOPs/MFU uses accumulated pure training step time (on_step_begin to
        # on_step_end) to measure hardware utilization during forward/backward only.
        now = time.monotonic()
        if "tokens" in logs:
            if self._last_log_time is not None:
                wall_elapsed = now - self._last_log_time
                if wall_elapsed > 0:
                    all_metrics["tok_per_sec"] = round(logs["tokens"] / wall_elapsed)

            # MFU requires knowing the hardware's peak FLOP/s.
            # Uses accumulated train time (forward/backward only) for accurate
            # hardware utilization measurement.
            train_elapsed = self._accumulated_train_time
            if self.peak_hardware_flops is not None and "total_flos" in logs:
                if train_elapsed > 0:
                    delta_flos = logs["total_flos"] - self._last_total_flos
                    if delta_flos > 0:
                        achieved_flops = delta_flos / train_elapsed
                        all_metrics["mfu"] = achieved_flops / self.peak_hardware_flops

        # Peak CUDA memory: expose raw bytes for the column formatter.
        peak_mem = logs.get("peak_mem_allocated")
        if peak_mem is not None:
            all_metrics["peak_mem"] = peak_mem

        # Reset interval tracking for the next log period
        self._last_log_time = now
        self._accumulated_train_time = 0.0
        self._last_total_flos = logs.get("total_flos", self._last_total_flos)

        # Filter to only keys present in step_columns
        display_metrics = {
            k: v for k, v in all_metrics.items() if k in self._column_keys
        }

        # Print a column header when the interval fires or the active column set changes
        active_keys = frozenset(display_metrics)
        if (
            self._log_row_count % self.header_interval == 0
            or active_keys != self._last_active_keys
        ):
            header_line = format_train_header(self.step_columns, display_metrics)
            if self.use_tqdm:
                if self.train_progress_bar is not None:
                    self.train_progress_bar.write(format_timestamp() + header_line)
            else:
                self.logger.info(header_line)
            self._last_active_keys = active_keys
        self._log_row_count += 1

        if self.use_tqdm:
            if self.train_progress_bar is not None:
                # Update steps, if max steps changes
                if self.train_progress_bar.total != state.max_steps:
                    self.train_progress_bar.total = state.max_steps
                    self.train_progress_bar.refresh()
                self.train_progress_bar.write(
                    format_timestamp()
                    + format_train_log(state, self.step_columns, display_metrics)
                )
        else:
            self.logger.info(
                format_train_log(state, self.step_columns, display_metrics)
            )


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
