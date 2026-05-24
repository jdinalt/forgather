"""Tests for tools/forgather_server/eval_ops.py — build_eval_command."""

import os

import pytest
from forgather_server import eval_ops
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

    def test_fused_loss_flag(self):
        cmd = self._base_call(fused_loss=True)
        assert "--fused-loss" in cmd

    def test_fused_loss_omitted_when_false(self):
        cmd = self._base_call(fused_loss=False)
        assert "--fused-loss" not in cmd

    def test_unknown_kwarg_ignored(self):
        """The shared spec drives forwarding; unknown keys are silently dropped.

        Defends against the queue store growing legacy keys that no longer
        map to a spec entry — they pass through job_params unchanged but must
        not produce stray argv tokens.
        """
        cmd = self._base_call(some_obsolete_flag="nope")
        assert "--some-obsolete-flag" not in cmd
        assert "nope" not in cmd


class TestConfigureEvalSearchPaths:
    """Behavior of ``configure_eval_search_paths`` + ``_resolve_search_paths``.

    Mutates module-level state, so each test resets to defaults afterwards
    to avoid leaking into the rest of the suite. Tests use ``tmp_path``
    for the extras so the resolver's "exists" check passes.
    """

    @pytest.fixture(autouse=True)
    def _isolate(self):
        yield
        eval_ops.configure_eval_search_paths([])  # reset

    def _bundled(self) -> str:
        from forgather_server.search_roots import forgather_repo_root

        return os.path.abspath(
            os.path.join(forgather_repo_root(), "examples", "evaluation")
        )

    def test_no_args_returns_library_default(self):
        # Without configure_*, _resolve_search_paths reproduces the
        # library's default discovery (bundled examples/evaluation +
        # user config extras, deduped).
        paths = eval_ops._resolve_search_paths()
        assert self._bundled() in paths

    def test_extra_dir_prepended(self, tmp_path):
        extra = tmp_path / "my_evals"
        extra.mkdir()
        eval_ops.configure_eval_search_paths([str(extra)])
        paths = eval_ops._resolve_search_paths()
        assert paths[0] == str(extra)
        # Bundled default still present after the extras.
        assert self._bundled() in paths

    def test_disable_default_removes_bundled(self, tmp_path):
        extra = tmp_path / "my_evals"
        extra.mkdir()
        eval_ops.configure_eval_search_paths([str(extra)], disable_default=True)
        paths = eval_ops._resolve_search_paths()
        assert paths == [str(extra)]
        assert self._bundled() not in paths

    def test_nonexistent_extra_silently_dropped(self, tmp_path):
        # A typo'd --eval-dir shouldn't kill discovery — bundled default
        # is still surfaced.
        eval_ops.configure_eval_search_paths([str(tmp_path / "does/not/exist")])
        paths = eval_ops._resolve_search_paths()
        assert self._bundled() in paths
        # The non-existent path is not in the resolved list.
        assert str(tmp_path / "does/not/exist") not in paths

    def test_dedup_preserves_priority_order(self, tmp_path):
        # If an extra duplicates a path already in the library default,
        # the extras-first ordering wins (the dup at the library-default
        # position is dropped, not the extra).
        d = tmp_path / "shared"
        d.mkdir()
        eval_ops.configure_eval_search_paths([str(d), str(d)])
        paths = eval_ops._resolve_search_paths()
        # str(d) appears exactly once.
        assert paths.count(str(d)) == 1
        # And it's first.
        assert paths[0] == str(d)
