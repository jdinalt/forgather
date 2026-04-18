"""Unit tests for the standalone ``scripts/eval_script.py``.

``scripts/`` is not a package, so we load the module via ``importlib.util``
and test the pure helpers (``resolve_checkpoint``, ``DTYPE_MAP``,
``parse_args``, formatters). Tests for the trainer construction path itself
live in integration tests.
"""

import argparse
import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_script.py"


@pytest.fixture(scope="module")
def eval_script():
    spec = importlib.util.spec_from_file_location("eval_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestResolveCheckpoint:
    @staticmethod
    def _ns(checkpoint=None, no_checkpoint=False):
        return argparse.Namespace(
            checkpoint=checkpoint, no_checkpoint=no_checkpoint
        )

    def test_default_is_auto_find(self, eval_script):
        arg, use = eval_script.resolve_checkpoint(self._ns())
        assert arg is True
        assert use is True

    def test_no_checkpoint_disables_resume(self, eval_script):
        arg, use = eval_script.resolve_checkpoint(self._ns(no_checkpoint=True))
        assert arg is False
        assert use is False

    def test_explicit_path(self, eval_script):
        arg, use = eval_script.resolve_checkpoint(
            self._ns(checkpoint="/tmp/ckpt-100")
        )
        assert arg == "/tmp/ckpt-100"
        assert use is True

    def test_no_checkpoint_overrides_explicit(self, eval_script):
        # --no-checkpoint takes precedence over --checkpoint.
        arg, use = eval_script.resolve_checkpoint(
            self._ns(checkpoint="/tmp/ckpt", no_checkpoint=True)
        )
        assert arg is False
        assert use is False


class TestDtypeMap:
    def test_canonical_names_present(self, eval_script):
        assert eval_script.DTYPE_MAP["float32"] is torch.float32
        assert eval_script.DTYPE_MAP["float16"] is torch.float16
        assert eval_script.DTYPE_MAP["bfloat16"] is torch.bfloat16

    def test_short_aliases_present(self, eval_script):
        assert eval_script.DTYPE_MAP["fp32"] is torch.float32
        assert eval_script.DTYPE_MAP["fp16"] is torch.float16
        assert eval_script.DTYPE_MAP["bf16"] is torch.bfloat16


class TestParseArgs:
    def test_required_flags(self, eval_script):
        with pytest.raises(SystemExit):
            eval_script.parse_args([])  # missing --eval-project/--eval-config/--model

    def test_minimum_valid_invocation(self, eval_script):
        ns = eval_script.parse_args(
            [
                "--eval-project",
                "/tmp/eval_proj",
                "--eval-config",
                "tinystories.yaml",
                "--model",
                "/tmp/model",
            ]
        )
        assert ns.eval_project == "/tmp/eval_proj"
        assert ns.eval_config == "tinystories.yaml"
        assert ns.model == "/tmp/model"
        assert ns.trainer == "ddp"
        assert ns.dtype == "bfloat16"
        assert ns.max_steps == -1

    def test_trainer_choices_are_enforced(self, eval_script):
        with pytest.raises(SystemExit):
            eval_script.parse_args(
                [
                    "--eval-project",
                    "/tmp/p",
                    "--eval-config",
                    "x.yaml",
                    "--model",
                    "/tmp/m",
                    "--trainer",
                    "bogus",
                ]
            )


class TestFormatters:
    def _record(self):
        return {
            "eval_name": "tinystories",
            "config_name": "Eval: TinyStories",
            "model_path": "/tmp/model",
            "checkpoint_path": None,
            "dataset_proj": "/tmp/ds",
            "dataset_config": "x.yaml",
            "dataset_target": "test_dataset",
            "batch_size": 4,
            "max_length": 256,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "trainer": "ddp",
            "world_size": 2,
            "eval_loss": 1.234567,
            "perplexity": 3.4363,
            "wall_time_s": 12.3,
        }

    def test_format_header_includes_key_fields(self, eval_script):
        header = eval_script.format_header(self._record())
        assert "Evaluation: tinystories" in header
        assert "(Eval: TinyStories)" in header
        assert "/tmp/model" in header
        assert "world_size=2" in header

    def test_format_results_includes_loss_and_perplexity(self, eval_script):
        results = eval_script.format_results(self._record())
        assert "eval_loss:" in results
        assert "1.234567" in results
        assert "perplexity:" in results
        assert "3.4363" in results
        assert "wall_time:" in results

    def test_format_header_includes_checkpoint_when_present(self, eval_script):
        rec = self._record()
        rec["checkpoint_path"] = "/tmp/ckpt-100"
        header = eval_script.format_header(rec)
        assert "/tmp/ckpt-100" in header
