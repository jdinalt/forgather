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
