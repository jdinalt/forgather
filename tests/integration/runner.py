"""Execute forgather training as a subprocess and capture results."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from forgather.ml.analysis.log_parser import TrainingLog

if TYPE_CHECKING:
    from .spec import IntegrationSpec

# Repo root, used to resolve spec.project_dir to an absolute path.
FG_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingResult:
    """Captured result of a forgather training run."""

    returncode: int
    stdout: str
    stderr: str
    project_dir: Path
    output_dir: Path
    log_file: Path | None
    training_log: TrainingLog | None
    duration_seconds: float


def _build_command(
    spec: IntegrationSpec, project_dir: Path, output_dir: Path
) -> list[str]:
    """Build the forgather train CLI command from a spec."""
    cmd = [
        "forgather",
        "-p",
        str(project_dir),
        "-t",
        spec.config,
        "train",
        "--output-dir",
        str(output_dir),
    ]

    for key, value in spec.dynamic_args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    return cmd


def run_forgather_train(
    spec: IntegrationSpec, output_dir: Path, timeout: int | None = None
) -> TrainingResult:
    """Execute forgather train and return captured results.

    The project runs in-place from its real location (so relative template
    paths work), but all output is redirected to ``output_dir`` via the
    ``--output-dir`` CLI flag.

    Args:
        spec: Integration test specification.
        output_dir: Temporary directory for training output.
        timeout: Subprocess timeout in seconds. Defaults to spec.timeout.

    Returns:
        TrainingResult with exit code, output, logs, and parsed training log.
    """
    if timeout is None:
        timeout = spec.timeout

    project_dir = FG_ROOT / spec.project_dir
    cmd = _build_command(spec, project_dir, output_dir)

    start = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    duration = time.monotonic() - start

    # Discover training log (most recent by mtime).
    # find_log_files looks inside output_models/ under a project dir,
    # but --output-dir redirects output_dir itself (no output_models/
    # subdirectory), so we search from output_dir's parent.
    log_file = None
    training_log = None
    log_files = list(output_dir.glob("runs/*/trainer_logs.json"))
    if not log_files:
        # Fall back to deeper search in case the layout differs
        log_files = list(output_dir.rglob("trainer_logs.json"))
    if log_files:
        log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        log_file = log_files[0]
        try:
            training_log = TrainingLog.from_file(log_file)
        except (ValueError, FileNotFoundError):
            pass

    return TrainingResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        project_dir=project_dir,
        output_dir=output_dir,
        log_file=log_file,
        training_log=training_log,
        duration_seconds=duration,
    )
