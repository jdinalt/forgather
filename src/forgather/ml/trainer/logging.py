import datetime
import sys
from pprint import pformat
from typing import Any, Callable, Literal

from forgather.ml.trainer.trainer_types import TrainerState
from forgather.ml.utils import alt_repr

Mapping = dict[str, Any]


def _fmt_si(v: int) -> str:
    """Format an integer with SI suffix (K, M, G)."""
    v = int(v)
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.3g}G"
    elif v >= 1_000_000:
        return f"{v / 1_000_000:.3g}M"
    elif v >= 1_000:
        return f"{v / 1_000:.3g}K"
    return str(v)


# Per-metric display config for columnar step logs.
# Each entry: (column_header, column_width, value_formatter)
# column_width must accommodate both the header label and the widest formatted value.
_STEP_METRICS: dict[str, tuple[str, int, Callable]] = {
    "epoch": ("epoch", 8, lambda v: f"{v:.4g}"),
    "loss": ("loss", 8, lambda v: f"{v:.5f}"),
    "grad_norm": ("grad", 8, lambda v: f"{v:.4f}"),
    "max_grad_norm": ("maxg", 8, lambda v: f"{v:.4f}"),
    "grad_norm_std": ("gn_std", 8, lambda v: f"{v:.4f}"),
    "learning_rate": ("lr", 10, lambda v: f"{v:.2e}"),
    "tokens": ("tokens", 10, lambda v: f"{int(v):,}"),
    "total_tokens": ("total_tok", 10, _fmt_si),
    "total_flos": ("flos", 10, lambda v: f"{v:.3e}"),
    "tok/s": ("tok/s", 10, lambda v: f"{int(v):,}"),
    "mfu": ("mfu", 6, lambda v: str(v)),
    "peak_mem": ("peak_mem", 11, lambda v: str(v)),
}

# Canonical column display order for train log rows.
_COLUMN_ORDER: list[str] = [
    "epoch",
    "loss",
    "grad_norm",
    "max_grad_norm",
    "grad_norm_std",
    "learning_rate",
    "tokens",
    "total_tokens",
    "tok/s",
    "mfu",
    "peak_mem",
]

# Width of the step count prefix column.
_STEP_COL_WIDTH = 10

# Ordered display labels and formatters for final training metrics.
# Keys present in the metrics dict but not listed here are shown at the end.
_FINAL_METRICS: dict[str, tuple[str, Callable]] = {
    "train_runtime": ("Runtime", lambda v: f"{v:.2f} s"),
    "step": ("Total steps", lambda v: f"{int(v):,}"),
    "train_samples": ("Total samples", lambda v: f"{int(v):,}"),
    "effective_batch_size": ("Effective batch size", lambda v: f"{int(v):,}"),
    "train_samples_per_second": ("Samples/sec", lambda v: f"{v:.3f}"),
    "train_steps_per_second": ("Steps/sec", lambda v: f"{v:.3f}"),
    "epoch": ("Epoch", lambda v: f"{v:.6g}"),
    "total_tokens": ("Total tokens", lambda v: f"{int(v):,}"),
    "tokens_per_second": ("Tokens/sec", lambda v: f"{v:,.0f}"),
    "total_flops": ("Total FLOPs", lambda v: f"{v:.3e}"),
    "flops_per_second": ("FLOPs/sec", lambda v: f"{v:.3e}"),
}


def _fmt_step_value(key: str, value: Any) -> str:
    if key in _STEP_METRICS:
        return _STEP_METRICS[key][2](value)
    elif isinstance(value, float):
        return f"{value:.4g}"
    elif isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _fmt_step_label(key: str) -> str:
    return _STEP_METRICS[key][0] if key in _STEP_METRICS else key


def _col_spec(key: str) -> tuple[str, int]:
    """Return (header_label, column_width) for a metric key."""
    if key in _STEP_METRICS:
        label, width, _ = _STEP_METRICS[key]
        return label, width
    return key, max(len(key) + 2, 8)


def format_train_info(
    args,
    state,
    control,
    model,
    processing_class,
    optimizer,
    lr_scheduler,
    train_dataloader,
    eval_dataloader,
    trainer=None,
    **kwargs,
):
    """
    Given objects passed to TrainerCallback, generate nice representations for logging

    This returns two dictionaries, info and extra_info, for basic and verbose logging.
    """
    if hasattr(state, "num_processes"):
        # Use trainer's method if available to correctly account for pipeline/model parallelism
        if trainer is not None and hasattr(trainer, "_calculate_effective_batch_size"):
            total_train_batch_size = trainer._calculate_effective_batch_size()
        else:
            # Fallback: assume data parallelism (may be incorrect for pipeline parallel)
            total_train_batch_size = state.num_processes * state.train_batch_size

        total_train_samples = total_train_batch_size * state.max_steps
        total_examples = state.epoch_train_steps * total_train_batch_size
        total_train_batch_size = f"{total_train_batch_size:,}"
        total_train_samples = f"{total_train_samples:,}"
        total_examples = f"{total_examples:,}"
    else:
        # TODO: The HF Trainer does not pass these values. Is there a way to compute this
        # from the available information?
        total_train_batch_size = "Unavailable"
        total_train_samples = "Unavailable"
        total_examples = "Unavailable"

    total_parameters = sum(t.numel() for t in model.parameters())
    trainable_parameters = sum(
        t.numel() if t.requires_grad else 0 for t in model.parameters()
    )
    num_params = lambda x: f"{x/1000000:.1f}M"

    info = {
        "total_examples": f"{total_examples}",
        "total_train_samples": f"{total_examples}",
        "per_device_train_batch_size": f"{args.per_device_train_batch_size:,}",
        "actual_per_device_batch_size": f"{state.train_batch_size:,}",
        "total_train_batch_size": f"{total_train_batch_size}",
        "max_steps": f"{state.max_steps:,}",
        "total_parameters": f"{num_params(total_parameters)}",
        "trainable_parameters": f"{num_params(trainable_parameters)}",
        "max_steps": f"{state.max_steps:,}",
    }

    extra_info = {
        "args": pformat(args),
        "state": pformat(state),
        "processing_class": pformat(processing_class),
        "optimizer": alt_repr(optimizer),
        "lr_schedulerr": alt_repr(lr_scheduler),
        "train_dataloader": alt_repr(train_dataloader),
        "eval_dataloader": alt_repr(eval_dataloader),
        "model": str(model),
    }
    return info, extra_info


def format_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{timestamp:<22}"


def format_log_header(state: TrainerState):
    s = f"{state.global_step:>10,d}  {round(state.epoch, 2):<5.3}"
    return s


def format_train_header(mapping: Mapping) -> str:
    """
    Format a column header row for the active metrics in mapping.

    Columns appear in canonical order (_COLUMN_ORDER), followed by any
    unrecognised keys. Each label is right-aligned within its column width,
    matching the alignment used by format_train_log.
    """
    parts = [f"{'step':>{_STEP_COL_WIDTH}}"]
    for key in _COLUMN_ORDER:
        if key in mapping:
            label, width = _col_spec(key)
            parts.append(f"{label:>{width}}")
    for key in mapping:
        if key not in _COLUMN_ORDER:
            label, width = _col_spec(key)
            parts.append(f"{label:>{width}}")
    return "  ".join(parts)


def format_train_log(state: TrainerState, mapping: Mapping) -> str:
    """
    Format a single columnar training step data row.

    Values are right-aligned in fixed-width columns matching the header
    produced by format_train_header. No key labels are included; the
    header row provides those. Unknown keys are appended after the
    canonical columns.
    """
    parts = [f"{state.global_step:>{_STEP_COL_WIDTH},d}"]
    for key in _COLUMN_ORDER:
        if key in mapping:
            _, width = _col_spec(key)
            parts.append(f"{_fmt_step_value(key, mapping[key]):>{width}}")
    for key, value in mapping.items():
        if key not in _COLUMN_ORDER:
            _, width = _col_spec(key)
            parts.append(f"{_fmt_step_value(key, value):>{width}}")
    return "  ".join(parts)


def format_final_metrics(metrics: Mapping) -> str:
    """
    Format end-of-training metrics as a multi-line human-readable summary.

    Known metrics are shown first in a fixed order with descriptive labels.
    Any remaining keys are appended at the end.
    """
    col_width = 28
    lines = ["Training complete:"]
    shown: set[str] = set()

    for key, (label, fmt) in _FINAL_METRICS.items():
        if key in metrics:
            lines.append(f"  {label + ':':{col_width}} {fmt(metrics[key])}")
            shown.add(key)

    # Any keys not in the known list
    for key, value in metrics.items():
        if key in shown:
            continue
        if isinstance(value, float):
            formatted = f"{value:.4g}"
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = str(value)
        lines.append(f"  {key + ':':{col_width}} {formatted}")

    return "\n".join(lines)


def format_eval_log(state, mapping: Mapping):
    header = format_log_header(state)
    if "eval_loss" in mapping:
        return f"{header}  eval-loss: {round(mapping['eval_loss'], 5)}"
    else:
        return header + format_mapping(mapping)


def format_mapping(mapping: Mapping):
    """
    Format a mapping for pretty-printing

    This is intended for formatting the mappings returned by format_train_info() as strings
    for console logging, but may be useful for formatting other datatypes as well.
    """
    s = ""
    for key, value in mapping.items():
        if isinstance(value, int):
            value = f"{value:,}"
        elif isinstance(value, float):
            value = f"{value:.4}"
        elif not isinstance(value, str):
            value = pformat(value)
        if len(value) > 80:
            s += f"{key}:\n{value}\n\n"
        else:
            s += f"{key}: {value}\n"
    return s


EvnType = Literal["file", "tty", "notebook"]


def get_env_type() -> EvnType:
    """
    Determine if output environment is a notebook, a TTY, or file/pipe
    """
    # Check if we are even in an IPython environment
    ipython = sys.modules.get("IPython")
    if ipython:
        try:
            shell = ipython.get_ipython()
            # Check for the Kernel config as TQDM does
            if shell and "IPKernelApp" in shell.config:
                return "notebook"
        except (AttributeError, NameError):
            pass

    # Check if we are outputting to a real terminal
    if sys.stdout.isatty():
        return "tty"

    # Default to file/redirection
    return "file"
