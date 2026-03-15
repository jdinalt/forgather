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
    _merge_spec_dicts,
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

from ..trainer_types import (
    MinimalTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

OutputStream = TextIOBase | Literal["stderr", "stdout"]


class DefaultMetrics(TrainerCallback):
    """Compute derived performance metrics and inject them into logs.

    Runs via ``on_log_step`` (before ``on_log``), so computed values are
    available to all downstream loggers (ProgressCallback, TBLogger, etc.).

    Computed metrics:
        tok_per_sec   -- tokens processed per wall-clock second between log steps.
        mfu           -- Model FLOPs Utilization (requires *peak_hardware_flops*).
        peak_mem      -- peak CUDA memory allocated (bytes), aliased from
                         ``peak_mem_allocated`` for display formatting.
    """

    def __init__(
        self,
        peak_hardware_flops: Optional[float] = None,
    ):
        """
        Args:
            peak_hardware_flops: Aggregate peak BF16 FLOP/s across all GPUs used
                in training, used to compute MFU.  Must be the total across all
                ranks since total_flos accounts for tokens processed across all
                ranks.  When ``None``, MFU is not computed.
                Example values (dense BF16, FP32 accumulate):
                  Single RTX 4090:  165.2e12
                  Single RTX 3090:   71.2e12
                  4x RTX 4090:      660.8e12
                  A100 SXM:         312e12
                  H100 SXM:         989e12
        """
        super().__init__()
        self.peak_hardware_flops = peak_hardware_flops

        # Wall-clock time at each log step, for tok/s end-to-end throughput.
        self._last_log_time: Optional[float] = None
        # Pure training step timing (on_step_begin to on_step_end), for FLOPs/MFU.
        self._step_start_time: Optional[float] = None
        self._accumulated_train_time: float = 0.0
        self._last_total_flos: float = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._last_log_time = None
        self._last_total_flos = state.total_flos
        self._accumulated_train_time = 0.0
        self._step_start_time = None

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

    def on_log_step(self, state, logs, **kwargs):
        if not state.is_world_process_zero:
            return

        now = time.monotonic()

        if "tokens" in logs:
            # tok/s: wall-clock throughput between log steps
            if self._last_log_time is not None:
                wall_elapsed = now - self._last_log_time
                if wall_elapsed > 0:
                    logs["tok_per_sec"] = round(logs["tokens"] / wall_elapsed)

            # MFU: hardware utilization during forward/backward only
            train_elapsed = self._accumulated_train_time
            if self.peak_hardware_flops is not None and "total_flos" in logs:
                if train_elapsed > 0:
                    delta_flos = logs["total_flos"] - self._last_total_flos
                    if delta_flos > 0:
                        achieved_flops = delta_flos / train_elapsed
                        logs["mfu"] = achieved_flops / self.peak_hardware_flops

        # Peak CUDA memory: alias for display formatting
        peak_mem = logs.get("peak_mem_allocated")
        if peak_mem is not None:
            logs["peak_mem"] = peak_mem

        # Reset interval tracking for the next log period
        self._last_log_time = now
        self._accumulated_train_time = 0.0
        self._last_total_flos = logs.get("total_flos", self._last_total_flos)


def _normalize_columns(columns: dict) -> list[ColumnSpec]:
    """Convert a dict of column specs into a list of ColumnSpec.

    Example::

        {"loss": {"label": "loss", "width": 8, "fmt": ".5f"},
         "grad_norm": {"label": "grad", "width": 8}}
    """
    result: list[ColumnSpec] = []
    for key, spec in columns.items():
        if isinstance(spec, ColumnSpec):
            result.append(spec)
        elif isinstance(spec, dict):
            spec = dict(spec)  # copy to avoid mutating the original
            spec["key"] = key
            result.append(ColumnSpec(**spec))
        else:
            raise TypeError(f"step_columns[{key!r}]: expected dict, got {type(spec)}")
    return result


def _normalize_final_metrics(metrics: dict) -> list[FinalMetricSpec]:
    """Convert a dict of final metric specs into a list of FinalMetricSpec."""
    result: list[FinalMetricSpec] = []
    for key, spec in metrics.items():
        if isinstance(spec, FinalMetricSpec):
            result.append(spec)
        elif isinstance(spec, dict):
            spec = dict(spec)
            spec["key"] = key
            result.append(FinalMetricSpec(**spec))
        else:
            raise TypeError(f"final_metrics[{key!r}]: expected dict, got {type(spec)}")
    return result


class ProgressCallback(TrainerCallback):
    """
    A TQDM progress-bar callback class based upon:
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py

    Controls which metrics are displayed in console logs during training
    via configurable column specifications.  All metrics are still logged
    to JsonLogger regardless of display settings.

    Derived performance metrics (tok/s, MFU, peak_mem) are computed by
    ``DefaultMetrics`` via ``on_log_step`` and are available in the logs
    dict by the time ``on_log`` fires.
    """

    def __init__(
        self,
        use_tqdm: Optional[bool] = None,
        output_stream: Optional[OutputStream] = None,
        step_columns: Optional[dict] = None,
        final_metrics: Optional[dict] = None,
        header_interval: int = 20,
    ):
        """
        Args:
            use_tqdm: If True, use TQDM; else if False, use logging; else auto select
            output_stream: The output stream to use, if using logging
            step_columns: A dict of column spec overrides, merged with
                ``default_step_columns()``.  Each key maps a metric name
                to a dict of ColumnSpec fields (label, width, fmt).
                Set a key to ``None`` to erase that column from the
                defaults.  Individual fields can be overridden without
                re-specifying the entire spec.  When ``None``, uses
                defaults unmodified.  Column order follows insertion
                order of the merged result.  Only columns whose key
                appears in the current log entry are shown.
            final_metrics: A dict of final metric spec overrides, merged
                with ``default_final_metrics()``.  Each key maps a metric
                name to a dict of FinalMetricSpec fields (label, fmt,
                suffix).  Set a key to ``None`` to erase that metric.
                When ``None``, uses defaults unmodified.
            header_interval: Print a column header row every this many log steps, and
                also whenever the set of active columns changes. (default: 20)
        """
        super().__init__()
        self.train_progress_bar = None
        self.eval_progress_bar = None
        self.header_interval = header_interval

        # Merge step_columns overrides with defaults, then convert to ColumnSpec list.
        merged_columns = _merge_spec_dicts(default_step_columns(), step_columns)
        self.step_columns: list[ColumnSpec] = _normalize_columns(merged_columns)

        # Merge final_metrics overrides with defaults, then convert to FinalMetricSpec list.
        merged_final = _merge_spec_dicts(default_final_metrics(), final_metrics)
        self.final_metrics: list[FinalMetricSpec] = _normalize_final_metrics(
            merged_final
        )

        self._column_keys: frozenset[str] = frozenset(c.key for c in self.step_columns)

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

        # Computed metrics (tok_per_sec, mfu, peak_mem, etc.) are already
        # in logs, injected by DefaultMetrics via on_log_step.

        # Filter to only keys present in step_columns
        display_metrics = {k: v for k, v in logs.items() if k in self._column_keys}

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


class InfoCallback(TrainerCallback):
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
