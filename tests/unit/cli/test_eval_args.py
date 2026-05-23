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
