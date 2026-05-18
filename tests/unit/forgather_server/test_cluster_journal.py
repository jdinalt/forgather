"""Tests for tools/forgather_server/cluster_journal.py."""

import json

import forgather_server.cluster_journal as cj
import pytest
from forgather_server import paths


@pytest.fixture(autouse=True)
def isolated_journal(tmp_path, monkeypatch):
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    monkeypatch.setattr(paths, "cluster_journal_dir", lambda: journal_dir)
    cj._reset_for_tests()
    yield journal_dir
    cj._reset_for_tests()


class TestAppendReplay:
    def test_append_writes_jsonl(self, isolated_journal):
        ev = cj.append("queue_add", {"queue_id": "q1"}, origin_node_id="n1")
        path = isolated_journal / cj.JOURNAL_FILENAME
        line = path.read_text().strip()
        d = json.loads(line)
        assert d["type"] == "queue_add"
        assert d["payload"] == {"queue_id": "q1"}
        assert d["origin_node_id"] == "n1"
        assert d["seq"] == ev.seq == 1

    def test_seq_monotonic_within_session(self, isolated_journal):
        a = cj.append("e1", origin_node_id="n")
        b = cj.append("e2", origin_node_id="n")
        c = cj.append("e3", origin_node_id="n")
        assert [a.seq, b.seq, c.seq] == [1, 2, 3]

    def test_replay_yields_in_order(self, isolated_journal):
        cj.append("a", origin_node_id="n", ts=1.0)
        cj.append("b", origin_node_id="n", ts=2.0)
        cj.append("c", origin_node_id="n", ts=3.0)
        events = list(cj.replay())
        assert [e.type for e in events] == ["a", "b", "c"]
        assert [e.seq for e in events] == [1, 2, 3]

    def test_replay_skips_malformed_lines(self, isolated_journal):
        path = isolated_journal / cj.JOURNAL_FILENAME
        path.write_text(
            json.dumps(
                {
                    "seq": 1,
                    "ts": 0.0,
                    "origin_node_id": "n",
                    "type": "ok",
                    "payload": {},
                }
            )
            + "\n"
            + "{not valid json\n"
            + "\n"
            + json.dumps(
                {
                    "seq": 2,
                    "ts": 0.0,
                    "origin_node_id": "n",
                    "type": "also_ok",
                    "payload": {},
                }
            )
            + "\n"
        )
        types = [e.type for e in cj.replay()]
        assert types == ["ok", "also_ok"]

    def test_init_recovers_seq_from_disk(self, isolated_journal):
        cj.append("a", origin_node_id="n")
        cj.append("b", origin_node_id="n")
        cj._reset_for_tests()
        # First append after reset must continue from 3, not 1.
        ev = cj.append("c", origin_node_id="n")
        assert ev.seq == 3

    def test_empty_event_type_rejected(self, isolated_journal):
        with pytest.raises(ValueError):
            cj.append("", origin_node_id="n")

    def test_payload_defaults_to_empty_dict(self, isolated_journal):
        ev = cj.append("e", origin_node_id="n")
        assert ev.payload == {}


class TestSubscribers:
    def test_subscriber_fires_on_append(self, isolated_journal):
        seen: list = []

        def sub(ev):
            seen.append(ev)

        cj.subscribe(sub)
        cj.append("hello", {"x": 1}, origin_node_id="n")
        assert len(seen) == 1
        assert seen[0].type == "hello"
        assert seen[0].payload == {"x": 1}

    def test_subscriber_exception_does_not_break_append(
        self, isolated_journal
    ):
        def boom(ev):
            raise RuntimeError("oops")

        cj.subscribe(boom)
        # Append must still succeed even though the subscriber raises.
        ev = cj.append("e", origin_node_id="n")
        assert ev.seq == 1
        assert next(cj.replay()).type == "e"

    def test_unsubscribe(self, isolated_journal):
        seen: list = []
        sub = lambda ev: seen.append(ev)
        cj.subscribe(sub)
        cj.append("a", origin_node_id="n")
        cj.unsubscribe(sub)
        cj.append("b", origin_node_id="n")
        assert [e.type for e in seen] == ["a"]


class TestEventSerialization:
    def test_round_trip(self):
        ev = cj.JournalEvent(
            seq=42,
            ts=1234567890.5,
            origin_node_id="abc",
            type="queue_add",
            payload={"queue_id": "q1", "priority": 5},
        )
        line = ev.to_jsonl()
        round_tripped = cj.JournalEvent.from_jsonl(line)
        assert round_tripped == ev
