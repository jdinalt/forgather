"""Assertion helpers for integration test results."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import TrainingResult
    from .spec import IntegrationSpec


def assert_exit_code(result: TrainingResult) -> None:
    """Assert training process exited successfully."""
    assert result.returncode == 0, (
        f"Training failed with exit code {result.returncode}.\n"
        f"stderr (last 3000 chars):\n{result.stderr[-3000:]}"
    )


def assert_no_forbidden_stderr(result: TrainingResult, spec: IntegrationSpec) -> None:
    """Assert no forbidden patterns appear in stderr."""
    for pattern in spec.stderr.forbidden_patterns:
        assert pattern not in result.stderr, (
            f"Forbidden pattern {pattern!r} found in stderr.\n"
            f"Context: ...{_find_context(result.stderr, pattern)}..."
        )


def assert_expected_files(result: TrainingResult, spec: IntegrationSpec) -> None:
    """Assert expected files exist in the run directory."""
    if not result.log_file:
        if spec.expected_files:
            raise AssertionError(
                "No training log file found -- cannot verify expected files. "
                f"output_root: {result.output_dir}"
            )
        return

    run_dir = result.log_file.parent
    for filename in spec.expected_files:
        path = run_dir / filename
        assert (
            path.exists()
        ), f"Expected file {filename!r} not found in run directory {run_dir}"


def assert_log_metrics(result: TrainingResult, spec: IntegrationSpec) -> None:
    """Assert training log metrics meet spec requirements."""
    assert (
        result.training_log is not None
    ), f"No training log found. output_root: {result.output_dir}"

    training_records = result.training_log.get_training_records()
    assert len(training_records) >= spec.min_steps_logged, (
        f"Expected at least {spec.min_steps_logged} training log entries, "
        f"got {len(training_records)}"
    )

    losses = result.training_log.get_metric_values("loss", training_records)

    # NaN check
    if spec.loss.no_nan:
        nan_steps = [
            r["global_step"]
            for r, loss in zip(training_records, losses)
            if math.isnan(loss)
        ]
        assert not nan_steps, f"NaN loss detected at steps: {nan_steps}"

    # Final loss bounds
    if losses:
        final_loss = losses[-1]
        if spec.loss.final_max is not None:
            assert (
                final_loss <= spec.loss.final_max
            ), f"Final loss {final_loss:.4f} exceeds maximum {spec.loss.final_max}"
        if spec.loss.final_min is not None:
            assert (
                final_loss >= spec.loss.final_min
            ), f"Final loss {final_loss:.4f} below minimum {spec.loss.final_min}"


def check_warn_patterns(result: TrainingResult, spec: IntegrationSpec) -> None:
    """Check for warning patterns in stderr. Issues warnings but does not fail."""
    for pattern in spec.stderr.warn_patterns:
        if pattern in result.stderr:
            warnings.warn(
                f"Warning pattern {pattern!r} found in stderr",
                stacklevel=2,
            )


def _find_context(text: str, pattern: str, context_chars: int = 200) -> str:
    """Extract context around first occurrence of pattern in text."""
    idx = text.find(pattern)
    if idx == -1:
        return ""
    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(pattern) + context_chars)
    return text[start:end]
