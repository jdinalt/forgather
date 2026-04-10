import math
from dataclasses import dataclass
from typing import Callable, Optional

from forgather.ml.trainer.logging import (
    _merge_spec_dicts,
    format_mapping,
    format_train_info,
)

from ..trainer_types import TrainerCallback


def get_perplexity(value, metrics):
    try:
        pp = math.exp(value)
    except OverflowError:
        pp = float("inf")
    return pp


# ---------------------------------------------------------------------------
# TBScalar specification
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TBScalarSpec:
    """Specification for a single TensorBoard scalar to log.

    Attributes:
        tag:       TensorBoard scalar tag (from dict key).
        source:    Metric key to read from logs.  Defaults to *tag* with
                   hyphens replaced by underscores.
        transform: Optional ``callable(value, metrics) -> new_value``.
    """

    tag: str
    source: str = ""
    transform: Callable | None = None

    def __post_init__(self):
        if not self.source:
            self.source = self.tag.replace("-", "_")


def default_tb_scalars() -> dict[str, dict]:
    """Return the default TensorBoard scalars as a dict.

    Each key is a TensorBoard tag; each value is a dict of TBScalarSpec
    fields (source, transform).  Keys whose source metric is absent from
    the logs dict are silently skipped at log time.
    """
    return {
        "train-loss": {"source": "loss"},
        "learning-rate": {},
        "grad-norm": {},
        "eval-loss": {},
        "eval-perplexity": {"source": "eval_loss", "transform": get_perplexity},
    }


def _normalize_tb_scalars(scalars: dict) -> list[TBScalarSpec]:
    """Convert a dict of TB scalar specs into a list of TBScalarSpec."""
    result: list[TBScalarSpec] = []
    for key, spec in scalars.items():
        if isinstance(spec, TBScalarSpec):
            result.append(spec)
        elif isinstance(spec, dict):
            spec = dict(spec)
            spec["tag"] = key
            result.append(TBScalarSpec(**spec))
        else:
            raise TypeError(f"tb_scalars[{key!r}]: expected dict, got {type(spec)}")
    return result


# ---------------------------------------------------------------------------
# TBLogger callback
# ---------------------------------------------------------------------------


class TBLogger(TrainerCallback):
    """A Trainer callback that logs scalars to TensorBoard.

    Scalars are configured as a dict mapping TensorBoard tags to spec
    dicts with optional ``source`` and ``transform`` fields.  The dict
    is merged with ``default_tb_scalars()`` so only deltas need to be
    specified.  Set a key to ``None`` to erase a default scalar.
    """

    def __init__(
        self,
        summary_writer,
        scalars: Optional[dict] = None,
        experiment_info: Optional[dict] = None,
    ):
        super().__init__()
        merged = _merge_spec_dicts(default_tb_scalars(), scalars)
        self.scalars: list[TBScalarSpec] = _normalize_tb_scalars(merged)
        self.summary_writer = summary_writer
        self.last_step = -1
        if experiment_info is not None:
            self.experiment_info = self.mapping_as_markdown(experiment_info)
        else:
            self.experiment_info = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.experiment_info is not None:
            self.summary_writer.add_text(
                "experiment", self.experiment_info, global_step=state.global_step
            )

        info, extra_info = format_train_info(args, state, control, **kwargs)
        self.summary_writer.add_text(
            "training_info",
            self.mapping_as_markdown(info | extra_info),
            global_step=state.global_step,
        )

    @staticmethod
    def mapping_as_markdown(mapping):
        """
        Format dictionary as markdown

        Tensorboard expects text to be in markdown format...
        """
        s = "```\n"
        s += format_mapping(mapping)
        s += "```"
        return s

    def _log_metrics(self, global_step, metrics):
        for spec in self.scalars:
            value = metrics.get(spec.source)
            if value is None:
                continue
            if spec.transform is not None:
                value = spec.transform(value, metrics)
                if value is None:
                    continue
            self.summary_writer.add_scalar(spec.tag, value, global_step=global_step)

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})
        if not state.is_world_process_zero:
            return
        global_step = state.global_step
        if self.last_step == global_step:
            return
        self.last_step = global_step

        self._log_metrics(global_step, metrics)

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        if not state.is_world_process_zero:
            return

        self._log_metrics(state.global_step, logs)
        self.summary_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero and len(state.log_history):
            return
        self.summary_writer.add_text(
            "train_results",
            self.mapping_as_markdown(
                state.log_history[-1],
            ),
            global_step=state.global_step,
        )
