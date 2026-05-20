"""Unit tests for the ``forgather eval`` argument parser."""

import argparse

import pytest

from forgather.cli.eval_args import create_eval_parser


@pytest.fixture
def parser():
    """Eval parser does not consult global_args; pass a dummy namespace."""
    return create_eval_parser(argparse.Namespace())


class TestEvalParser:
    def test_list_subcommand(self, parser):
        ns = parser.parse_args(["list"])
        assert ns.eval_subcommand == "list"

    def test_show_subcommand_requires_name(self, parser):
        ns = parser.parse_args(["show", "c4"])
        assert ns.eval_subcommand == "show"
        assert ns.name == "c4"
        assert ns.pp is False

    def test_show_pp_flag(self, parser):
        ns = parser.parse_args(["show", "c4", "--pp"])
        assert ns.pp is True

    def test_test_defaults(self, parser):
        ns = parser.parse_args(["test", "tinystories"])
        assert ns.eval_subcommand == "test"
        assert ns.name == "tinystories"
        assert ns.trainer == "ddp"
        assert ns.dtype == "bfloat16"
        assert ns.attn_implementation == "sdpa"
        assert ns.max_steps == -1
        assert ns.no_checkpoint is False
        assert ns.compile is False
        assert ns.dry_run is False
        assert ns.model is None
        assert ns.batch_size is None
        assert ns.max_length is None
        assert ns.checkpoint is None
        assert ns.output_dir is None
        assert ns.devices is None

    def test_test_trainer_choices(self, parser):
        for choice in ("ddp", "simple", "pipeline"):
            ns = parser.parse_args(["test", "c4", "--trainer", choice])
            assert ns.trainer == choice

    def test_test_rejects_unknown_trainer(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["test", "c4", "--trainer", "bogus"])

    def test_test_full_flag_set(self, parser):
        ns = parser.parse_args(
            [
                "test",
                "c4",
                "-M",
                "/tmp/model",
                "-d",
                "0,1",
                "--trainer",
                "pipeline",
                "--checkpoint",
                "/tmp/ckpt",
                "--batch-size",
                "4",
                "--max-length",
                "4096",
                "--max-steps",
                "10",
                "--dtype",
                "float16",
                "--attn-implementation",
                "flex_attention",
                "--compile",
                "--output-dir",
                "/tmp/out",
                "--dry-run",
            ]
        )
        assert ns.model.endswith("/tmp/model")
        assert ns.devices == "0,1"
        assert ns.trainer == "pipeline"
        assert ns.checkpoint.endswith("/tmp/ckpt")
        assert ns.batch_size == 4
        assert ns.max_length == 4096
        assert ns.max_steps == 10
        assert ns.dtype == "float16"
        assert ns.attn_implementation == "flex_attention"
        assert ns.compile is True
        assert ns.output_dir.endswith("/tmp/out")
        assert ns.dry_run is True


class TestForwardEvalScriptArgs:
    """Cover the Namespace-driven forwarder used by ``eval.test_cmd``.

    The dict-driven sibling (``forward_eval_script_args_from_params``) is
    exercised indirectly by ``test_eval_ops.TestBuildEvalCommand``; the two
    helpers must produce equivalent output for equivalent inputs even though
    they consult different lookup keys (argparse dest vs ``enqueue_key``).
    """

    def test_checkpoint_uses_argparse_dest_not_enqueue_key(self, parser):
        """--checkpoint's dest is ``checkpoint`` but its enqueue_key is
        ``checkpoint_path``. The Namespace path must read from the dest."""
        from forgather.cli.eval_args import forward_eval_script_args

        ns = parser.parse_args(["test", "c4", "--checkpoint", "/ckpts/step-1000"])
        tokens = forward_eval_script_args(ns)
        assert "--checkpoint" in tokens
        idx = tokens.index("--checkpoint")
        assert tokens[idx + 1] == "/ckpts/step-1000"

    def test_batch_size_zero_is_forwarded_not_dropped(self, parser):
        """value-mode args use ``is not None`` semantics — 0 must round-trip.

        Pre-refactor, ``--batch-size 0`` would have been forwarded from the
        local CLI but dropped by the enqueue payload (truthy check). The
        ``not_none`` enqueue mode fixes the asymmetry; this test pins it.
        """
        from forgather.cli.eval_args import (
            eval_script_args_to_job_params,
            forward_eval_script_args,
        )

        ns = parser.parse_args(["test", "c4", "--batch-size", "0"])
        tokens = forward_eval_script_args(ns)
        assert "--batch-size" in tokens
        assert tokens[tokens.index("--batch-size") + 1] == "0"
        # And the enqueue payload agrees.
        payload = eval_script_args_to_job_params(ns)
        assert payload.get("batch_size") == 0

    def test_flag_omitted_when_false(self, parser):
        from forgather.cli.eval_args import forward_eval_script_args

        ns = parser.parse_args(["test", "c4"])
        tokens = forward_eval_script_args(ns)
        assert "--fused-loss" not in tokens
        assert "--no-checkpoint" not in tokens
        assert "--compile" not in tokens

    def test_namespace_and_dict_paths_produce_identical_argv(self, parser):
        """``forward_eval_script_args`` (Namespace) and
        ``forward_eval_script_args_from_params`` (dict) must agree byte-for-byte
        for the same logical input. Locks the contract that the dest /
        enqueue_key divergence on ``--checkpoint`` (and any future renames)
        doesn't desync the two helpers.
        """
        from forgather.cli.eval_args import (
            eval_script_args_to_job_params,
            forward_eval_script_args,
            forward_eval_script_args_from_params,
        )

        ns = parser.parse_args(
            [
                "test", "c4",
                "--trainer", "simple",
                "--checkpoint", "/ckpts/step-42",
                "--fused-loss",
                "--batch-size", "8",
                "--max-length", "4096",
                "--max-steps", "100",
                "--dtype", "bfloat16",
                "--attn-implementation", "sdpa",
                "--output-dir", "/tmp/out",
            ]
        )
        local = forward_eval_script_args(ns)
        server = forward_eval_script_args_from_params(
            eval_script_args_to_job_params(ns)
        )
        assert local == server


class TestSpecValidation:
    """The spec validation runs at import; this asserts the catch behavior
    exists (the import succeeded means the real spec is valid)."""

    def test_invalid_forward_mode_raises(self):
        from forgather.cli import eval_args

        bogus = dict(eval_args._EVAL_SCRIPT_ARGS[0])
        bogus["forward"] = "bogus_mode"
        original = eval_args._EVAL_SCRIPT_ARGS
        try:
            eval_args._EVAL_SCRIPT_ARGS = [bogus]
            with pytest.raises(RuntimeError, match="forward="):
                eval_args._validate_spec()
        finally:
            eval_args._EVAL_SCRIPT_ARGS = original

    def test_duplicate_enqueue_key_raises(self):
        from forgather.cli import eval_args

        a = dict(eval_args._EVAL_SCRIPT_ARGS[0])  # trainer
        b = dict(eval_args._EVAL_SCRIPT_ARGS[0])  # also trainer — dup key
        original = eval_args._EVAL_SCRIPT_ARGS
        try:
            eval_args._EVAL_SCRIPT_ARGS = [a, b]
            with pytest.raises(RuntimeError, match="duplicate enqueue_key"):
                eval_args._validate_spec()
        finally:
            eval_args._EVAL_SCRIPT_ARGS = original
