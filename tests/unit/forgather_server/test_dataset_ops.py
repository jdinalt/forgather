"""Tests for tools/forgather_server/dataset_ops.py."""

import pytest
from forgather_server.dataset_ops import build_dataset_command


class TestBuildDatasetCommand:
    def _base_call(self, **kwargs):
        return build_dataset_command(
            project_dir="/some/project",
            config_name="train.yaml",
            **kwargs,
        )

    def test_basic_structure(self):
        cmd = self._base_call()
        assert "forgather.cli" in cmd
        assert "-p" in cmd
        assert "/some/project" in cmd
        assert "-t" in cmd
        assert "train.yaml" in cmd
        assert "dataset" in cmd

    def test_valid_feature_name(self):
        cmd = self._base_call(features=["input_ids", "labels"])
        assert "--features" in cmd
        assert "input_ids" in cmd
        assert "labels" in cmd

    def test_empty_feature_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            self._base_call(features=[""])

    def test_dash_only_feature_raises(self):
        with pytest.raises(ValueError, match="does not start with"):
            self._base_call(features=["-"])

    def test_flag_injection_via_feature_raises(self):
        with pytest.raises(ValueError, match="does not start with"):
            self._base_call(features=["--foo"])

    def test_multiple_features_some_invalid_raises(self):
        with pytest.raises(ValueError):
            self._base_call(features=["valid", "--bad"])

    def test_optional_args_included(self):
        cmd = self._base_call(
            tokenizer_path="/tok",
            pp=True,
            histogram=True,
            histogram_samples=200,
            examples=50,
            seed=42,
        )
        assert "--tokenizer-path" in cmd
        assert "/tok" in cmd
        assert "--pp" in cmd
        assert "--histogram" in cmd
        assert "--histogram-samples" in cmd
        assert "200" in cmd
        assert "--examples" in cmd
        assert "50" in cmd
        assert "--seed" in cmd
        assert "42" in cmd

    def test_none_optional_args_not_included(self):
        cmd = self._base_call()
        for flag in ("--pp", "--histogram", "--tokenized", "--tokenizer-path"):
            assert flag not in cmd
