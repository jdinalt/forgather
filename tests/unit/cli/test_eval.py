"""Unit tests for forgather.cli.eval discovery helpers.

These tests exercise the pure-Python discovery path against the builtin
``examples/evaluation/`` projects bundled with the repo — not the full
trainer / torchrun pipeline, which is covered by integration tests (or by
running ``forgather eval test`` manually against a real model).
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from forgather.cli import eval as eval_cli
from forgather.eval_config import find_eval_config, iter_eval_configs


@pytest.fixture
def search_paths():
    """Resolve the builtin eval search path for the current checkout."""
    forgather_dir = eval_cli._forgather_dir()
    # Do not read the real user config during tests — force the builtin default.
    with patch("forgather.cli.eval.eval_search_paths") as mock_paths:
        default = os.path.join(forgather_dir, "examples", "evaluation")
        mock_paths.return_value = [os.path.abspath(default)]
        yield mock_paths.return_value


class TestForgatherDir:
    def test_locates_repo_root_with_templatelib(self):
        root = eval_cli._forgather_dir()
        assert (Path(root) / "templatelib").is_dir()


class TestIterEvalConfigs:
    def test_yields_builtin_configs(self, search_paths):
        names = {name for name, *_ in iter_eval_configs(search_paths)}
        # The five shipped configs must be discoverable.
        assert {
            "c4",
            "openorca",
            "openassistant",
            "fineweb-edu-dedup",
            "tinystories",
        } <= names

    def test_entries_have_required_fields(self, search_paths):
        # Each entry is a TestConfig dataclass; required fields must be set
        # by the YAML config.
        from forgather.eval_config import TestConfig

        for name, _, _, data in iter_eval_configs(search_paths):
            assert isinstance(data, TestConfig)
            assert data.dataset_proj, f"{name} missing dataset_proj"
            assert data.dataset_config, f"{name} missing dataset_config"
            assert data.dataset_target, f"{name} missing dataset_target"
            assert data.name, f"{name} missing display name"

    def test_entries_have_library_default_fallback(self, search_paths):
        # Optional fields pick up TestConfig defaults when the YAML is silent.
        for name, _, _, data in iter_eval_configs(search_paths):
            assert data.default_batch_size > 0, f"{name} has non-positive batch size"
            assert data.default_max_length > 0, f"{name} has non-positive max length"
            assert data.default_stride >= 0, f"{name} has negative stride"

    def test_missing_search_path_is_skipped(self):
        # Nonexistent paths must not raise; they are just skipped.
        results = list(iter_eval_configs(["/nonexistent/xyz/123"]))
        assert results == []


class TestFindEvalConfig:
    def test_known_name_returns_tuple(self, search_paths):
        project_dir, template, data = find_eval_config("tinystories", search_paths)
        assert os.path.isdir(project_dir)
        assert template.endswith(".yaml")
        assert data.eval_name == "tinystories"

    def test_unknown_name_raises_lookup_error(self, search_paths):
        with pytest.raises(LookupError):
            find_eval_config("does-not-exist", search_paths)
