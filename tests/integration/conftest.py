"""Fixtures and hooks for integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from .spec import IntegrationSpec, load_all_specs

SPECS_DIR = Path(__file__).parent / "specs"
FG_ROOT = Path(__file__).resolve().parents[2]


def _available_gpus() -> int:
    """Return number of available CUDA GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _all_specs() -> list[IntegrationSpec]:
    """Load all specs (cached at module level)."""
    return load_all_specs(SPECS_DIR)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    """Parametrize tests that request a 'spec' fixture."""
    if "spec" not in metafunc.fixturenames:
        return

    specs = _all_specs()

    # Filter specs by test function expectations
    func_name = metafunc.function.__name__
    if func_name == "test_inference_with_perplexity":
        specs = [s for s in specs if s.inference is not None]
    elif func_name == "test_training_project":
        specs = [s for s in specs if s.inference is None]

    metafunc.parametrize("spec", specs, ids=[s.test_id for s in specs], indirect=True)


def pytest_collection_modifyitems(items):
    """Apply markers from specs and skip tests based on GPU availability."""
    gpu_count = _available_gpus()

    for item in items:
        # Try to get spec from callspec
        if not (hasattr(item, "callspec") and "spec" in item.callspec.params):
            continue

        spec = item.callspec.params["spec"]

        # Apply markers from spec
        for marker_name in spec.markers:
            item.add_marker(getattr(pytest.mark, marker_name))

        # Skip if not enough GPUs
        if spec.gpu_requirement > gpu_count:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"Requires {spec.gpu_requirement} GPU(s), "
                        f"only {gpu_count} available"
                    )
                )
            )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spec(request):
    """Provide the IntegrationSpec for parametrized tests."""
    return request.param


@pytest.fixture
def output_dir(tmp_path):
    """Provide a temporary output directory for training artifacts.

    Uses the --output-dir dynamic arg to redirect all training output
    (model code, logs, checkpoints) to a temp directory. The project
    itself runs in-place from its real location, so relative template
    paths and cross-project references work correctly.

    Returns:
        Path to the temp output directory.
    """
    return tmp_path / "output"
