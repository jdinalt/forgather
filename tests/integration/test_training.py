"""Integration tests for training projects.

Tests are parametrized by YAML spec files in tests/integration/specs/.
Only specs WITHOUT an inference section are collected here.
"""

import pytest

from .assertions import (
    assert_exit_code,
    assert_expected_files,
    assert_log_metrics,
    assert_no_forbidden_stderr,
    check_warn_patterns,
)
from .runner import run_forgather_train


@pytest.mark.integration
def test_training_project(spec, output_dir):
    """Run a training project and validate results against its spec."""
    result = run_forgather_train(spec, output_dir)

    assert_exit_code(result)
    assert_no_forbidden_stderr(result, spec)
    assert_expected_files(result, spec)
    assert_log_metrics(result, spec)
    check_warn_patterns(result, spec)
