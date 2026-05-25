"""Tests for the DILOCO_* env-var translation in scheduler._build_training."""

from __future__ import annotations

from forgather_server.scheduler import _diloco_env_from_job_params

# Stable queue_id used as the fallback worker_id in most tests.
QID = "q-test-1234"


def test_empty_dict_emits_nothing():
    assert _diloco_env_from_job_params({}, QID) == {}


def test_missing_server_addr_short_circuits():
    # Server addr is the gate — even if other keys are set, no env
    # vars get emitted when the worker is opting out of DiLoCo entirely.
    assert _diloco_env_from_job_params({"sync_every": 500}, QID) == {}


def test_server_addr_alone_yields_minimal_env():
    # When DiLoCo is on but no operator-supplied worker_id is present,
    # DILOCO_WORKER_ID falls back to the queue_id — this is the
    # "do-the-right-thing" default that lets config preprocessing
    # always derive a unique output dir.
    env = _diloco_env_from_job_params({"server_addr": "host:8512"}, QID)
    assert env == {
        "DILOCO_SERVER": "host:8512",
        "DILOCO_WORKER_ID": QID,
    }


def test_full_payload_translates_all_keys():
    env = _diloco_env_from_job_params(
        {
            "server_addr": "host:8512",
            "sync_every": 500,
            "num_fragments": 4,
            "dylu": True,
            "bf16_comm": False,
            "heartbeat_interval": 15.0,
            "worker_id": "w1",
        },
        QID,
    )
    assert env == {
        "DILOCO_SERVER": "host:8512",
        "DILOCO_SYNC_EVERY": "500",
        "DILOCO_NUM_FRAGMENTS": "4",
        "DILOCO_DYLU": "1",
        "DILOCO_BF16_COMM": "0",
        "DILOCO_HEARTBEAT_INTERVAL": "15.0",
        "DILOCO_WORKER_ID": "w1",
    }


def test_bf16_true_translates_to_1():
    env = _diloco_env_from_job_params({"server_addr": "h:1", "bf16_comm": True}, QID)
    assert env["DILOCO_BF16_COMM"] == "1"


def test_dylu_false_translates_to_0():
    env = _diloco_env_from_job_params({"server_addr": "h:1", "dylu": False}, QID)
    assert env["DILOCO_DYLU"] == "0"


def test_empty_worker_id_falls_back_to_queue_id():
    # The config-preprocessing-time output dir derivation needs
    # DILOCO_WORKER_ID to be non-empty. An empty operator value falls
    # back to queue_id rather than being dropped.
    env = _diloco_env_from_job_params({"server_addr": "h:1", "worker_id": ""}, QID)
    assert env["DILOCO_WORKER_ID"] == QID


def test_whitespace_worker_id_falls_back_to_queue_id():
    env = _diloco_env_from_job_params({"server_addr": "h:1", "worker_id": "   "}, QID)
    assert env["DILOCO_WORKER_ID"] == QID


def test_explicit_worker_id_wins_over_queue_id():
    env = _diloco_env_from_job_params({"server_addr": "h:1", "worker_id": "alpha"}, QID)
    assert env["DILOCO_WORKER_ID"] == "alpha"


def test_none_typed_fields_are_skipped():
    # None values shouldn't emit env vars — the callback's constructor
    # then falls back to its own DILOCO_* env reads (which would be
    # unset), then to its default. ``worker_id`` is the one exception:
    # it always gets set (queue_id fallback).
    env = _diloco_env_from_job_params(
        {
            "server_addr": "h:1",
            "sync_every": None,
            "num_fragments": None,
            "dylu": None,
            "bf16_comm": None,
            "heartbeat_interval": None,
        },
        QID,
    )
    assert env == {"DILOCO_SERVER": "h:1", "DILOCO_WORKER_ID": QID}
