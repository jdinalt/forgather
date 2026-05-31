"""Unit tests for the ``forgather diloco`` argument parser (parity verbs)."""

import argparse

import pytest

from forgather.cli.diloco_args import create_diloco_parser


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


class TestDynamicArgPartitionScoping:
    def test_worker_dynamic_names_not_propagated_to_diloco_parser(self, monkeypatch):
        """Regression: the worker's dynamic-arg names must NOT land on the
        top-level diloco parser. main.py partitions the *whole* chosen
        subcommand against that list, and the dynamic set includes
        framework names like ``output_dir`` — propagating them would strip
        ``--output-dir`` from a sibling like ``diloco server``."""
        import forgather.cli.dynamic_args as da

        def fake_parse(parser, ga):
            parser._dynamic_arg_names = ["output_dir", "max_steps"]

        monkeypatch.setattr(da, "parse_dynamic_args", fake_parse)
        ga = argparse.Namespace(config_template="x", project_dir=".", no_dyn=False)
        parser = create_diloco_parser(ga)
        # Not smeared onto the top-level parser.
        assert getattr(parser, "_dynamic_arg_names", []) == []
        # `diloco server` still keeps its required --output-dir.
        ns = parser.parse_args(["server", "-o", "/out", "-n", "2"])
        assert ns.output_dir == "/out"


class TestStatusEnrichment:
    def test_new_flags_default_off(self, parser):
        ns = parser.parse_args(["status"])
        assert ns.queues is False
        assert ns.json is False
        assert ns.direct is False
        assert ns.via_server is None
        # existing direct-path defaults preserved
        assert ns.server == "localhost:8512"

    def test_new_flags_parse(self, parser):
        ns = parser.parse_args(
            ["status", "--server", "h:9000", "--queues", "--json", "--direct"]
        )
        assert ns.server == "h:9000"
        assert ns.queues is True
        assert ns.json is True
        assert ns.direct is True
