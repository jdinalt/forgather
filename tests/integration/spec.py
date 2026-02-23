"""Declarative integration test specification schema and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LossBounds:
    """Bounds on training loss for assertions."""

    final_max: float | None = None
    final_min: float | None = None
    no_nan: bool = True


@dataclass
class StderrAssertions:
    """Patterns to check in stderr output."""

    forbidden_patterns: list[str] = field(default_factory=list)
    warn_patterns: list[str] = field(default_factory=list)


@dataclass
class InferenceSpec:
    """Specification for inference smoke test with perplexity scoring."""

    prompt: str = "Once upon a time"
    max_tokens: int = 50
    temperature: float = 0.7
    perplexity_max: float = 500.0
    server_timeout: int = 60


@dataclass
class IntegrationSpec:
    """Complete specification for an integration test."""

    test_id: str
    project_dir: str
    config: str
    dynamic_args: dict[str, Any] = field(default_factory=dict)
    loss: LossBounds = field(default_factory=LossBounds)
    stderr: StderrAssertions = field(default_factory=StderrAssertions)
    expected_files: list[str] = field(default_factory=lambda: ["trainer_logs.json"])
    min_steps_logged: int = 1
    inference: InferenceSpec | None = None
    gpu_requirement: int = 1
    timeout: int = 300
    markers: list[str] = field(default_factory=lambda: ["integration"])


def _load_spec(path: Path) -> IntegrationSpec:
    """Load a single spec from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    loss_raw = raw.pop("loss", {})
    loss = LossBounds(**loss_raw) if loss_raw else LossBounds()

    stderr_raw = raw.pop("stderr", {})
    stderr = StderrAssertions(**stderr_raw) if stderr_raw else StderrAssertions()

    inference_raw = raw.pop("inference", None)
    inference = InferenceSpec(**inference_raw) if inference_raw else None

    return IntegrationSpec(loss=loss, stderr=stderr, inference=inference, **raw)


def load_all_specs(specs_dir: Path) -> list[IntegrationSpec]:
    """Load all spec YAML files from a directory, sorted by test_id."""
    specs = []
    if not specs_dir.is_dir():
        return specs
    for path in sorted(specs_dir.glob("*.yaml")):
        specs.append(_load_spec(path))
    return specs
