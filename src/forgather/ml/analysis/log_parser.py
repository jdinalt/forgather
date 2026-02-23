"""Parse and load training logs."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainingLog:
    """Represents a parsed training log."""

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
        """Get a human-readable label for this log.

        Priority: explicit label > model_name > run_name > 'Run N'
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
        """Load training log from JSON file.

        Args:
            log_path: Path to trainer_logs.json file

        Returns:
            TrainingLog object

        Raises:
            FileNotFoundError: If log file doesn't exist
            ValueError: If log file is invalid JSON
        """
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        try:
            with open(log_path, "r") as f:
                records = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in log file: {e}")

        if not isinstance(records, list):
            raise ValueError("Log file must contain a JSON array")

        return cls(log_path=log_path, records=records)

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "TrainingLog":
        """Load training log from a run directory.

        Args:
            run_dir: Path to run directory containing trainer_logs.json

        Returns:
            TrainingLog object
        """
        run_dir = Path(run_dir)
        log_path = run_dir / "trainer_logs.json"
        return cls.from_file(log_path)

    def get_training_records(self) -> List[Dict[str, Any]]:
        """Get records with training metrics (loss, grad_norm, lr)."""
        return [r for r in self.records if "loss" in r and "eval_loss" not in r]

    def get_eval_records(self) -> List[Dict[str, Any]]:
        """Get records with evaluation metrics (eval_loss)."""
        return [r for r in self.records if "eval_loss" in r]

    def get_final_record(self) -> Optional[Dict[str, Any]]:
        """Get the final summary record (contains train_runtime)."""
        for r in reversed(self.records):
            if "train_runtime" in r:
                return r
        return None

    def get_metric_values(
        self, metric: str, records: Optional[List[Dict[str, Any]]] = None
    ) -> List[float]:
        """Extract values for a specific metric.

        Args:
            metric: Metric name (e.g., 'loss', 'learning_rate')
            records: Specific records to extract from, or None for all records

        Returns:
            List of metric values
        """
        if records is None:
            records = self.records
        return [r[metric] for r in records if metric in r]

    def get_steps(self, records: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """Extract global step values.

        Args:
            records: Specific records to extract from, or None for all records

        Returns:
            List of step values
        """
        return self.get_metric_values("global_step", records)

    def get_epochs(self, records: Optional[List[Dict[str, Any]]] = None) -> List[float]:
        """Extract epoch values.

        Args:
            records: Specific records to extract from, or None for all records

        Returns:
            List of epoch values
        """
        return self.get_metric_values("epoch", records)

    def get_timestamps(
        self, records: Optional[List[Dict[str, Any]]] = None
    ) -> List[float]:
        """Extract timestamp values.

        Args:
            records: Specific records to extract from, or None for all records

        Returns:
            List of timestamp values
        """
        return self.get_metric_values("timestamp", records)

    def find_best_step(
        self, metric: str, mode: str = "min"
    ) -> Optional[tuple[int, float]]:
        """Find the step with the best value for a metric.

        Args:
            metric: Metric name
            mode: 'min' for lowest value, 'max' for highest value

        Returns:
            Tuple of (step, value) or None if metric not found
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
    """Find all trainer_logs.json files in a project.

    Args:
        project_dir: Project directory to search
        model_name: Optional model name to filter by

    Returns:
        List of paths to log files
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
