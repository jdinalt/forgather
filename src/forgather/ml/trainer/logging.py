import datetime
import sys
from dataclasses import dataclass
from pprint import pformat
from typing import Any, Callable, Literal

from forgather.ml.trainer.trainer_types import TrainerState
from forgather.ml.utils import alt_repr, count_parameters, fmt_si

Mapping = dict[str, Any]

Reduction = str | Callable[[list], Any] | None


# ---------------------------------------------------------------------------
# Column / metric specification dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ColumnSpec:
    """Specification for a single column in the step-log table.

    Attributes
    ----------
    key : str
        Metric key in the logs dict (e.g. ``"loss"``, ``"tok_per_sec"``).
    label : str
        Column header text.  Defaults to *key* when empty.
    width : int
        Fixed column width in characters.
    fmt : str or callable
        Formatting control -- one of:

        * A Python format-spec string (``".5f"``, ``".2e"``, ``",d"``).
          Applied via ``format(value, spec)``.  Integer presentation
          types (``d``, ``o``, ``x``, ...) auto-convert the value to
          ``int`` first.
        * A named formatter alias (``"si"``, ``"gib"``).
        * A ``Callable[[Any], str]``.
        * ``""`` for a type-based fallback (float -> ``.4g``,
          int -> comma-separated, else ``str``).
    reduce : str or callable or None
        How to render list-valued metrics (e.g. a per-rank list of
        peak memory values).  One of:

        * ``None`` (default) -- scalar values pass through; list
          values fall back to an implicit ``max`` reduction.
        * ``"max"``, ``"min"``, ``"mean"``, ``"sum"`` -- reduce a
          list to a scalar before formatting.
        * ``"all"`` -- format each element individually and join
          with ``"/"`` (per-rank display; may overflow *width*).
        * A ``Callable[[list], Any]`` -- custom reduction; the
          result is then formatted via *fmt*.

        Ignored for scalar values unless the reduction is a callable.
    """

    key: str
    label: str = ""
    width: int = 10
    fmt: str | Callable[[Any], str] = ""
    reduce: Reduction = None

    def __post_init__(self):
        if not self.label:
            self.label = self.key


@dataclass(slots=True)
class FinalMetricSpec:
    """Specification for a single row in the end-of-training summary.

    Attributes
    ----------
    key : str
        Metric key in the final metrics dict.
    label : str
        Human-readable label.  Defaults to *key* when empty.
    fmt : str or callable
        Same semantics as :class:`ColumnSpec.fmt`.
    suffix : str
        String appended after the formatted value (e.g. ``" s"``).
    """

    key: str
    label: str = ""
    fmt: str | Callable[[Any], str] = ""
    suffix: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.key


# ---------------------------------------------------------------------------
# Spec dict merging
# ---------------------------------------------------------------------------


def _merge_spec_dicts(defaults: dict, overrides: dict | None) -> dict:
    """Merge *overrides* into *defaults*, returning a new dict.

    - New keys are added.
    - Existing keys with dict values are shallow-merged (override fields win).
    - Keys set to ``None`` are erased.
    """
    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in defaults.items()}
    if overrides is None:
        return merged
    for key, value in overrides.items():
        if value is None:
            merged.pop(key, None)
        elif (
            key in merged and isinstance(merged[key], dict) and isinstance(value, dict)
        ):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Public formatters
# ---------------------------------------------------------------------------


# Re-exported from forgather.ml.utils for backward compatibility.
# fmt_si is imported at the top of this module.


def fmt_gib(v: float | int) -> str:
    """Format a byte count as GiB."""
    return f"{v / (1024 ** 3):.3f} GiB"


# Named formatter registry -- looked up by format_value when *fmt* is a
# string that matches a key here.
_NAMED_FORMATTERS: dict[str, Callable[[Any], str]] = {
    "si": fmt_si,
    "gib": fmt_gib,
}


# Named reducers for list-valued metrics.  See ColumnSpec.reduce.
_NAMED_REDUCERS: dict[str, Callable[[list], Any]] = {
    "max": max,
    "min": min,
    "mean": lambda xs: sum(xs) / len(xs) if xs else 0,
    "sum": sum,
}


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

# Integer presentation types that require an int operand.
_INT_TYPES = frozenset("doxXbn")


def _format_scalar(value: Any, fmt: str | Callable[[Any], str]) -> str:
    """Format a single scalar metric value.

    Resolution order:

    1. *fmt* is callable -- call it directly.
    2. *fmt* matches a named formatter -- delegate to it.
    3. *fmt* is a non-empty string -- treat as a Python format spec.
       Integer presentation types (``d``, ``o``, …) auto-convert *value*
       to ``int``.
    4. *fmt* is empty -- fall back to type-based defaults.
    """
    if callable(fmt):
        return fmt(value)
    if fmt in _NAMED_FORMATTERS:
        return _NAMED_FORMATTERS[fmt](value)
    if not fmt:
        if isinstance(value, float):
            return f"{value:.4g}"
        elif isinstance(value, int):
            return f"{value:,}"
        return str(value)
    if fmt[-1] in _INT_TYPES:
        value = int(value)
    if value is None:
        return "None"
    return format(value, fmt)


def format_value(
    value: Any,
    fmt: str | Callable[[Any], str],
    reduce: Reduction = None,
) -> str:
    """Format a single metric value, with optional list reduction.

    For scalar *value* the *fmt* path is unchanged from legacy behavior.

    For list *value* the *reduce* argument controls rendering:

    * ``None`` -- implicit ``max`` reduction (suits peak-style metrics).
    * ``"max" / "min" / "mean" / "sum"`` -- named scalar reduction,
      then format via *fmt*.
    * ``"all"`` -- format each element with *fmt* and join with ``"/"``.
    * Callable -- apply to the list; result is then formatted via *fmt*.

    A callable *reduce* applied to a scalar value is still honored.
    """
    if isinstance(value, list):
        if reduce is None or reduce == "max":
            value = max(value) if value else 0
        elif reduce == "all":
            return "/".join(_format_scalar(v, fmt) for v in value)
        elif callable(reduce):
            value = reduce(value)
        elif reduce in _NAMED_REDUCERS:
            value = _NAMED_REDUCERS[reduce](value)
        else:
            raise ValueError(f"Unknown reduction {reduce!r}")
    elif callable(reduce):
        value = reduce(value)
    return _format_scalar(value, fmt)


# ---------------------------------------------------------------------------
# Default column / metric specs
# ---------------------------------------------------------------------------

# Width of the step count prefix column.
_STEP_COL_WIDTH = 10


def default_step_columns() -> dict[str, dict]:
    """Return the default step-log columns as a dict.

    Each key is a metric name; each value is a dict of ColumnSpec fields
    (label, width, fmt).  Columns whose key is absent from the metrics
    dict are silently skipped at display time.
    """
    return {
        "epoch": {"label": "epoch", "width": 10, "fmt": ".4g"},
        "loss": {"label": "loss", "width": 8, "fmt": ".5f"},
        "grad_norm": {"label": "grad", "width": 8, "fmt": ".4f"},
        "learning_rate": {"label": "lr", "width": 10, "fmt": ".2e"},
        "tokens": {"label": "tokens", "width": 10, "fmt": ",d"},
        "total_tokens": {"label": "total_tok", "width": 10, "fmt": "si"},
        "tok_per_sec": {"label": "tok/s", "width": 10, "fmt": ",d"},
        "mfu": {"label": "mfu", "width": 6, "fmt": ".1%"},
        "peak_mem": {"label": "peak_mem", "width": 11, "fmt": "gib", "reduce": "max"},
    }


def default_final_metrics() -> dict[str, dict]:
    """Return the default end-of-training summary metrics as a dict.

    Each key is a metric name; each value is a dict of FinalMetricSpec
    fields (label, fmt, suffix).
    """
    return {
        "train_runtime": {"label": "Runtime", "fmt": ".2f", "suffix": " s"},
        "step": {"label": "Total steps", "fmt": ",d"},
        "train_samples": {"label": "Total samples", "fmt": ",d"},
        "effective_batch_size": {"label": "Effective batch size", "fmt": ",d"},
        "train_samples_per_second": {"label": "Samples/sec", "fmt": ".3f"},
        "train_steps_per_second": {"label": "Steps/sec", "fmt": ".3f"},
        "epoch": {"label": "Epoch", "fmt": ".6g"},
        "total_tokens": {"label": "Total tokens", "fmt": ",d"},
        "tokens_per_second": {"label": "Tokens/sec", "fmt": ",.0f"},
        "total_flops": {"label": "Total FLOPs", "fmt": ".3e"},
        "flops_per_second": {"label": "FLOPs/sec", "fmt": ".3e"},
    }


# ---------------------------------------------------------------------------
# Train info (unchanged)
# ---------------------------------------------------------------------------


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

    stats = count_parameters(model)

    info = {
        "total_examples": f"{total_examples}",
        "total_train_samples": f"{total_examples}",
        "per_device_train_batch_size": f"{args.per_device_train_batch_size:,}",
        "actual_per_device_batch_size": f"{state.train_batch_size:,}",
        "total_train_batch_size": f"{total_train_batch_size}",
        "max_steps": f"{state.max_steps:,}",
        "total_parameters": fmt_si(stats.total),
        "trainable_parameters": fmt_si(stats.trainable),
        "embedding_parameters": fmt_si(stats.embedding),
        "non_embedding_parameters": fmt_si(stats.non_embedding),
        "tied_embeddings": str(stats.tied_embeddings),
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


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{timestamp:<22}"


def format_log_header(state: TrainerState):
    s = f"{state.global_step:>10,d}  {round(state.epoch, 2):<5.3}"
    return s


def format_train_header(columns: list[ColumnSpec], metrics: Mapping) -> str:
    """Format a column header row for the active metrics.

    Only columns whose key appears in *metrics* are included.
    Column order follows the *columns* list.
    """
    parts = [f"{'step':>{_STEP_COL_WIDTH}}"]
    for col in columns:
        if col.key in metrics:
            parts.append(f"{col.label:>{col.width}}")
    return "  ".join(parts)


def format_train_log(
    state: TrainerState, columns: list[ColumnSpec], metrics: Mapping
) -> str:
    """Format a single columnar training step data row.

    Values are right-aligned in fixed-width columns matching the header
    produced by :func:`format_train_header`.  Only columns whose key
    appears in *metrics* are shown.
    """
    parts = [f"{state.global_step:>{_STEP_COL_WIDTH},d}"]
    for col in columns:
        if col.key in metrics:
            formatted = format_value(metrics[col.key], col.fmt, col.reduce)
            parts.append(f"{formatted:>{col.width}}")
    return "  ".join(parts)


def format_final_metrics(
    metrics: Mapping, specs: list[FinalMetricSpec] | None = None
) -> str:
    """Format end-of-training metrics as a multi-line human-readable summary.

    Known metrics (from *specs*, or :func:`default_final_metrics` when
    *specs* is ``None``) are shown first in spec order.  Any remaining
    keys are appended at the end.
    """
    if specs is None:
        specs = [
            FinalMetricSpec(key=k, **v) for k, v in default_final_metrics().items()
        ]

    col_width = 28
    lines = ["Training complete:"]
    shown: set[str] = set()

    for spec in specs:
        if spec.key in metrics:
            formatted = format_value(metrics[spec.key], spec.fmt) + spec.suffix
            lines.append(f"  {spec.label + ':':{col_width}} {formatted}")
            shown.add(spec.key)

    for key, value in metrics.items():
        if key in shown:
            continue
        lines.append(f"  {key + ':':{col_width}} {format_value(value, '')}")

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
