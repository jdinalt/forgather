"""Tests for tools/forgather_server/eval_ops.py — build_eval_command."""

import pytest
from forgather_server.eval_ops import build_eval_command


class TestBuildEvalCommand:
    def _base_call(self, **kwargs):
        defaults = dict(
            eval_project="/eval/proj",
            eval_template="perplexity.yaml",
            model_path="/models/my_model",
        )
        defaults.update(kwargs)
        return build_eval_command(**defaults)

    def test_ddp_uses_torchrun(self):
        cmd = self._base_call(trainer="ddp")
        assert "torchrun" in cmd
        assert "--standalone" in cmd

    def test_simple_does_not_use_torchrun(self):
        cmd = self._base_call(trainer="simple")
        assert "torchrun" not in cmd

    def test_required_args_present(self):
        cmd = self._base_call()
        assert "--eval-project" in cmd
        assert "/eval/proj" in cmd
        assert "--eval-config" in cmd
        assert "perplexity.yaml" in cmd
        assert "--model" in cmd
        assert "/models/my_model" in cmd

    def test_checkpoint_path_included(self):
        cmd = self._base_call(checkpoint_path="/ckpts/step-1000")
        assert "--checkpoint" in cmd
        assert "/ckpts/step-1000" in cmd

    def test_no_checkpoint_included_when_flagged(self):
        cmd = self._base_call(no_checkpoint=True)
        assert "--no-checkpoint" in cmd

    def test_checkpoint_and_no_checkpoint_not_both_present(self):
        # Sanity: with a real checkpoint, --no-checkpoint should not be injected.
        cmd = self._base_call(checkpoint_path="/ckpts/step-1000", no_checkpoint=False)
        assert "--checkpoint" in cmd
        assert "--no-checkpoint" not in cmd

    def test_neither_checkpoint_nor_no_checkpoint_by_default(self):
        cmd = self._base_call()
        assert "--checkpoint" not in cmd
        assert "--no-checkpoint" not in cmd

    def test_compile_flag(self):
        cmd = self._base_call(compile=True)
        assert "--compile" in cmd

    def test_output_dir_included(self):
        cmd = self._base_call(output_dir="/out/eval_run")
        assert "--output-dir" in cmd
        assert "/out/eval_run" in cmd

    def test_max_steps_in_command(self):
        cmd = self._base_call(max_steps=500)
        assert "--max-steps" in cmd
        idx = cmd.index("--max-steps")
        assert cmd[idx + 1] == "500"
