"""Unit tests for the ``forgather diloco`` argument parser (parity verbs)."""

import argparse
import sys

import pytest

from forgather.cli.diloco_args import _diloco_subcommand, create_diloco_parser


class TestDilocoSubcommandDetection:
    def test_worker(self):
        assert _diloco_subcommand(["diloco", "worker", "--count", "2"]) == "worker"

    def test_with_global_flags_before_diloco(self):
        assert (
            _diloco_subcommand(["-t", "cfg", "-p", "d", "diloco", "worker"]) == "worker"
        )

    def test_other_subcommand(self):
        assert _diloco_subcommand(["diloco", "status", "--queues"]) == "status"

    def test_help_flag_is_not_a_subcommand(self):
        assert _diloco_subcommand(["diloco", "--help"]) is None

    def test_no_diloco(self):
        assert _diloco_subcommand(["train", "--max-steps", "5"]) is None


@pytest.fixture
def parser():
    # create_diloco_parser takes global_args but the parity verbs here
    # don't consult it; a bare namespace is enough.
    return create_diloco_parser(argparse.Namespace())


class TestServersVerb:
    def test_defaults(self, parser):
        ns = parser.parse_args(["servers"])
        assert ns.diloco_subcommand == "servers"
        assert ns.via_server is None
        assert ns.json is False

    def test_flags(self, parser):
        ns = parser.parse_args(["servers", "--json", "--via-server", "https://h:8765"])
        assert ns.json is True
        assert ns.via_server == "https://h:8765"


class TestLogsVerb:
    def test_requires_job(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["logs"])

    def test_job_and_follow(self, parser):
        ns = parser.parse_args(["logs", "spectacular-fox", "-f"])
        assert ns.diloco_subcommand == "logs"
        assert ns.job == "spectacular-fox"
        assert ns.follow is True

    def test_follow_default_false(self, parser):
        ns = parser.parse_args(["logs", "q123"])
        assert ns.follow is False


class TestDynamicArgGating:
    """Dynamic-arg loading is gated to a `worker` invocation (so sibling
    verbs don't pay a config load), and uses the project default when -t is
    omitted (parity with `forgather train`)."""

    def test_worker_dynamic_names_not_propagated_to_diloco_parser(self, monkeypatch):
        """Regression: the worker's dynamic-arg names must NOT land on the
        top-level diloco parser. main.py partitions the *whole* chosen
        subcommand against that list, and the dynamic set includes
        framework names like ``output_dir`` — propagating them would strip
        ``--output-dir`` from a sibling like ``diloco server``."""
        import forgather.cli.dynamic_args as da

        def fake_parse(parser, ga):
            parser._dynamic_arg_names = ["output_dir", "max_steps"]

        # Gate sees a worker invocation, so parse_dynamic_args runs.
        monkeypatch.setattr(sys, "argv", ["forgather", "diloco", "worker"])
        monkeypatch.setattr(da, "parse_dynamic_args", fake_parse)
        ga = argparse.Namespace(config_template="x", project_dir=".", no_dyn=False)
        parser = create_diloco_parser(ga)
        # Not smeared onto the top-level parser.
        assert getattr(parser, "_dynamic_arg_names", []) == []
        # `diloco server` still keeps its required --output-dir.
        ns = parser.parse_args(["server", "-o", "/out", "-n", "2"])
        assert ns.output_dir == "/out"

    def test_loads_for_worker_without_explicit_config(self, monkeypatch):
        # No -t (config_template=None) must still load dynamic args for a
        # worker invocation — parity with `forgather train`.
        import forgather.cli.dynamic_args as da

        calls = []
        monkeypatch.setattr(sys, "argv", ["forgather", "diloco", "worker"])
        monkeypatch.setattr(da, "parse_dynamic_args", lambda p, ga: calls.append(ga))
        ga = argparse.Namespace(config_template=None, project_dir=".", no_dyn=False)
        create_diloco_parser(ga)
        assert len(calls) == 1

    def test_skips_for_non_worker_subcommand(self, monkeypatch):
        # A sibling verb (status) must NOT pay the config load, even with -t.
        import forgather.cli.dynamic_args as da

        calls = []
        monkeypatch.setattr(sys, "argv", ["forgather", "diloco", "status"])
        monkeypatch.setattr(da, "parse_dynamic_args", lambda p, ga: calls.append(ga))
        ga = argparse.Namespace(config_template="x", project_dir=".", no_dyn=False)
        create_diloco_parser(ga)
        assert calls == []

    def test_skips_when_no_dyn(self, monkeypatch):
        import forgather.cli.dynamic_args as da

        calls = []
        monkeypatch.setattr(sys, "argv", ["forgather", "diloco", "worker"])
        monkeypatch.setattr(da, "parse_dynamic_args", lambda p, ga: calls.append(ga))
        ga = argparse.Namespace(config_template=None, project_dir=".", no_dyn=True)
        create_diloco_parser(ga)
        assert calls == []

    def test_server_host_short_and_long(self, parser):
        # -H is the short alias for --host (consistent with `forgather
        # server` / `dataset-server start`).
        ns = parser.parse_args(["server", "-o", "/out", "-n", "2", "-H", "0.0.0.0"])
        assert ns.host == "0.0.0.0"
        ns = parser.parse_args(["server", "-o", "/out", "-n", "2", "--host", "1.2.3.4"])
        assert ns.host == "1.2.3.4"
        # Default unchanged.
        assert parser.parse_args(["server", "-o", "/o", "-n", "1"]).host == "127.0.0.1"


class TestStatusEnrichment:
    def test_new_flags_default_off(self, parser):
        ns = parser.parse_args(["status"])
        assert ns.queues is False
        assert ns.json is False
        assert ns.local_only is False
        assert ns.local_fallback is False
        assert ns.via_server is None
        # --server is now optional (auto-discovered / loopback-default).
        assert ns.server is None

    def test_new_flags_parse(self, parser):
        ns = parser.parse_args(
            ["status", "--server", "h:9000", "--queues", "--json", "--local-only"]
        )
        assert ns.server == "h:9000"
        assert ns.queues is True
        assert ns.json is True
        assert ns.local_only is True
