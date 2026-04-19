"""Parse and load training logs."""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _try_recover_truncated_json(
    text: str, log_path: Path, original_error: json.JSONDecodeError
) -> List[Dict[str, Any]]:
    """Attempt to recover records from a truncated JSON array.

    Finds the last complete JSON object in the array and re-parses.
    Raises ValueError if recovery fails.
    """
    # Find the last complete object (closing brace)
    last_brace = text.rfind("}")
    if last_brace <= 0:
        raise ValueError(f"Invalid JSON in log file: {original_error}")

    try:
        records = json.loads(text[: last_brace + 1] + "]")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in log file: {original_error}")

    if not isinstance(records, list):
        raise ValueError(f"Invalid JSON in log file: {original_error}")

    logger.warning(
        "Recovered %d records from truncated log file: %s", len(records), log_path
    )
    return records


@dataclass
class TrainingLog:
    """Container for a parsed Forgather training log.

    Holds all JSON records emitted by Forgather's JSON logger
    (``trainer_logs.json``) together with metadata inferred from the
    file-system path.  Typically created via :meth:`from_file` or
    :meth:`from_run_dir` rather than constructed directly.

    Parameters
    ----------
    log_path : Path
        Absolute path to the ``trainer_logs.json`` file.
    records : list of dict
        Raw JSON records as loaded from the log file.  Each record is a
        dictionary that may contain keys such as ``global_step``, ``loss``,
        ``eval_loss``, ``learning_rate``, ``grad_norm``, ``epoch``,
        ``timestamp``, and ``train_runtime``.
    run_name : str, optional
        Human-readable name of the training run, usually the timestamped
        directory name under ``runs/``.  Inferred from *log_path* when not
        provided.
    model_name : str, optional
        Name of the model, usually the directory immediately above ``runs/``
        in the output path.  Inferred from *log_path* when not provided.
    label : str, optional
        Explicit display label used when plotting.  When set, this takes
        priority over *model_name* and *run_name*.

    Examples
    --------
    >>> from forgather.ml.analysis import TrainingLog
    >>> log = TrainingLog.from_file("output_models/my_model/runs/run_001/trainer_logs.json")
    >>> train_records = log.get_training_records()
    >>> losses = log.get_metric_values("loss", train_records)
    """

    log_path: Path
    records: List[Dict[str, Any]]
    run_name: Optional[str] = None
    model_name: Optional[str] = None
    label: Optional[str] = None

    def __post_init__(self):
        """Extract run name and model name from path if not provided."""
        parts = self.log_path.parts
        if "runs" in parts:
            runs_idx = parts.index("runs")
            if self.run_name is None and runs_idx + 1 < len(parts):
                self.run_name = parts[runs_idx + 1]
            if self.model_name is None and runs_idx > 0:
                self.model_name = parts[runs_idx - 1]

    def get_label(self, index: int = 0) -> str:
        """Return a human-readable label for this log.

        Selection priority: explicit :attr:`label` > :attr:`model_name` >
        :attr:`run_name` > ``'Run N'`` (where *N* is *index* + 1).

        Parameters
        ----------
        index : int, optional
            Zero-based position of this log in a collection, used as a
            fallback label suffix.  Default is 0.

        Returns
        -------
        str
            Display label suitable for plot legends and summary output.
        """
        if self.label:
            return self.label
        if self.model_name:
            return self.model_name
        if self.run_name:
            return self.run_name
        return f"Run {index + 1}"

    @classmethod
    def from_file(cls, log_path: str | Path) -> "TrainingLog":
        """Load a training log from a ``trainer_logs.json`` file.

        Handles truncated files (e.g. from a crash or a still-running job) by
        attempting to recover all complete JSON records before the truncation
        point.

        Parameters
        ----------
        log_path : str or Path
            Path to a ``trainer_logs.json`` file produced by Forgather's JSON
            logger.

        Returns
        -------
        TrainingLog
            Populated instance with all recoverable records.

        Raises
        ------
        FileNotFoundError
            If *log_path* does not exist on disk.
        ValueError
            If the file is not a valid JSON array and recovery fails.

        Examples
        --------
        >>> log = TrainingLog.from_file("output_models/my_model/runs/run_001/trainer_logs.json")
        >>> print(f"Loaded {len(log.records)} records")
        """
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        with open(log_path, "r") as f:
            text = f.read()
        try:
            records = json.loads(text)
        except json.JSONDecodeError as e:
            # Attempt to recover truncated JSON (e.g., crash, still-running job)
            records = _try_recover_truncated_json(text, log_path, e)

        if not isinstance(records, list):
            raise ValueError("Log file must contain a JSON array")

        return cls(log_path=log_path, records=records)

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "TrainingLog":
        """Load a training log from a run directory.

        Convenience wrapper around :meth:`from_file` that automatically appends
        ``trainer_logs.json`` to the supplied directory path.

        Parameters
        ----------
        run_dir : str or Path
            Path to a run directory (e.g.
            ``output_models/my_model/runs/run_001/``) that contains a
            ``trainer_logs.json`` file.

        Returns
        -------
        TrainingLog
            Populated instance with all recoverable records.

        Raises
        ------
        FileNotFoundError
            If ``trainer_logs.json`` is not found inside *run_dir*.
        ValueError
            If the log file is not a valid JSON array and recovery fails.
        """
        run_dir = Path(run_dir)
        log_path = run_dir / "trainer_logs.json"
        return cls.from_file(log_path)

    def get_training_records(self) -> List[Dict[str, Any]]:
        """Return records that contain training-step metrics.

        Training records are identified by the presence of a ``loss`` key and
        the absence of an ``eval_loss`` key.  They typically also carry
        ``grad_norm``, ``learning_rate``, ``global_step``, ``epoch``, and
        ``timestamp``.

        Returns
        -------
        list of dict
            Subset of :attr:`records` corresponding to training steps.
        """
        return [r for r in self.records if "loss" in r and "eval_loss" not in r]

    def get_eval_records(self) -> List[Dict[str, Any]]:
        """Return records that contain evaluation metrics.

        Evaluation records are identified by the presence of an ``eval_loss``
        key.  They typically also carry ``global_step`` and ``epoch``.

        Returns
        -------
        list of dict
            Subset of :attr:`records` corresponding to evaluation checkpoints.
        """
        return [r for r in self.records if "eval_loss" in r]

    def get_final_record(self) -> Optional[Dict[str, Any]]:
        """Return the final summary record emitted at the end of training.

        The final record is identified by the presence of a ``train_runtime``
        key and may also contain ``train_samples``,
        ``train_samples_per_second``, ``train_steps_per_second``, and
        ``effective_batch_size``.

        Returns
        -------
        dict or None
            The last record containing ``train_runtime``, or ``None`` if no
            such record exists (e.g. training was interrupted).
        """
        for r in reversed(self.records):
            if "train_runtime" in r:
                return r
        return None

    def get_metric_values(
        self, metric: str, records: Optional[List[Dict[str, Any]]] = None
    ) -> List[float]:
        """Extract the values for a named metric from a set of records.

        Records that do not contain *metric* are silently skipped, so the
        returned list may be shorter than *records*.

        Parameters
        ----------
        metric : str
            Key to extract (e.g. ``'loss'``, ``'learning_rate'``,
            ``'grad_norm'``, ``'eval_loss'``, ``'global_step'``).
        records : list of dict, optional
            Records to search.  When ``None``, all :attr:`records` are used.

        Returns
        -------
        list of float
            Ordered values for *metric* drawn from the matching records.
        """
        if records is None:
            records = self.records
        return [r[metric] for r in records if metric in r]

    def get_steps(self, records: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """Extract ``global_step`` values from records.

        Parameters
        ----------
        records : list of dict, optional
            Records to search.  When ``None``, all :attr:`records` are used.

        Returns
        -------
        list of int
            Ordered global step numbers.
        """
        return self.get_metric_values("global_step", records)

    def get_epochs(self, records: Optional[List[Dict[str, Any]]] = None) -> List[float]:
        """Extract ``epoch`` values from records.

        Parameters
        ----------
        records : list of dict, optional
            Records to search.  When ``None``, all :attr:`records` are used.

        Returns
        -------
        list of float
            Ordered fractional epoch numbers.
        """
        return self.get_metric_values("epoch", records)

    def get_timestamps(
        self, records: Optional[List[Dict[str, Any]]] = None
    ) -> List[float]:
        """Extract ``timestamp`` values from records.

        Timestamps are Unix epoch seconds recorded when each log entry was
        written.  They can be used to build a wall-clock x-axis for plots.

        Parameters
        ----------
        records : list of dict, optional
            Records to search.  When ``None``, all :attr:`records` are used.

        Returns
        -------
        list of float
            Ordered Unix timestamps (seconds since epoch).
        """
        return self.get_metric_values("timestamp", records)

    def find_best_step(
        self, metric: str, mode: str = "min"
    ) -> Optional[tuple[int, float]]:
        """Find the training step at which a metric reaches its best value.

        Parameters
        ----------
        metric : str
            Metric key to search for (e.g. ``'loss'``, ``'eval_loss'``).
        mode : {'min', 'max'}, optional
            Whether to look for the minimum (``'min'``) or maximum (``'max'``)
            value.  Default is ``'min'``.

        Returns
        -------
        tuple of (int, float) or None
            ``(global_step, value)`` at the best record, or ``None`` if no
            record contains *metric*.
        """
        records = [r for r in self.records if metric in r]
        if not records:
            return None

        if mode == "min":
            best_record = min(records, key=lambda r: r[metric])
        else:
            best_record = max(records, key=lambda r: r[metric])

        return best_record["global_step"], best_record[metric]


def find_log_files(
    project_dir: str | Path, model_name: Optional[str] = None
) -> List[Path]:
    """Find all ``trainer_logs.json`` files in a project's output directory.

    Searches under ``<project_dir>/output_models/`` using the standard
    Forgather run directory structure (``<model>/runs/<run>/trainer_logs.json``).

    Parameters
    ----------
    project_dir : str or Path
        Root directory of the Forgather project.
    model_name : str, optional
        When provided, only log files under ``output_models/<model_name>/``
        are returned.

    Returns
    -------
    list of Path
        Absolute paths to matching ``trainer_logs.json`` files, sorted by
        modification time (most recent first).
    """
    project_dir = Path(project_dir)
    output_models_dir = project_dir / "output_models"

    if not output_models_dir.exists():
        return []

    log_files = []
    search_pattern = (
        f"{model_name}/runs/*/trainer_logs.json"
        if model_name
        else "*/runs/*/trainer_logs.json"
    )

    for log_file in output_models_dir.glob(search_pattern):
        log_files.append(log_file)

    return sorted(log_files, key=lambda p: p.stat().st_mtime, reverse=True)


def find_diagnostic_logs(project_dir: str | Path, filename: str) -> List[Path]:
    """Find diagnostic log files in a project's output directory.

    Searches for files matching *filename* (e.g. ``"parameter_norms.json"``,
    ``"gradient_norms.json"``) in the standard Forgather run directory
    structure under ``<project_dir>/output_models/``.

    Parameters
    ----------
    project_dir : str or Path
        Root directory of the Forgather project.
    filename : str
        Exact filename to search for within run directories.

    Returns
    -------
    list of Path
        Absolute paths to matching files, sorted by modification time (most
        recent first).
    """
    project_dir = Path(project_dir)
    output_models_dir = project_dir / "output_models"

    if not output_models_dir.exists():
        return []

    log_files = list(output_models_dir.glob(f"*/runs/*/{filename}"))

    return sorted(log_files, key=lambda p: p.stat().st_mtime, reverse=True)
