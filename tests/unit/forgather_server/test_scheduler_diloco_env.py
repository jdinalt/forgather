"""Tests for the DILOCO_* env-var translation in scheduler._build_training."""

from __future__ import annotations

from forgather_server.scheduler import _diloco_env_from_job_params


def test_empty_dict_emits_nothing():
    assert _diloco_env_from_job_params({}) == {}


def test_missing_server_addr_short_circuits():
    # Server addr is the gate — even if other keys are set, no env
    # vars get emitted when the worker is opting out of DiLoCo entirely.
    assert _diloco_env_from_job_params({"sync_every": 500}) == {}


def test_server_addr_alone_yields_minimal_env():
    env = _diloco_env_from_job_params({"server_addr": "host:8512"})
    assert env == {"DILOCO_SERVER": "host:8512"}


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
        }
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
    env = _diloco_env_from_job_params({"server_addr": "h:1", "bf16_comm": True})
    assert env["DILOCO_BF16_COMM"] == "1"


def test_dylu_false_translates_to_0():
    env = _diloco_env_from_job_params({"server_addr": "h:1", "dylu": False})
    assert env["DILOCO_DYLU"] == "0"


def test_empty_worker_id_is_ignored():
    # The callback's env-var fallback for worker_id requires a non-empty
    # string; an empty value should NOT shadow it.
    env = _diloco_env_from_job_params({"server_addr": "h:1", "worker_id": ""})
    assert "DILOCO_WORKER_ID" not in env


def test_none_typed_fields_are_skipped():
    # None values shouldn't emit env vars — the callback's constructor
    # then falls back to its own DILOCO_* env reads (which would be
    # unset), then to its default.
    env = _diloco_env_from_job_params(
        {
            "server_addr": "h:1",
            "sync_every": None,
            "num_fragments": None,
            "dylu": None,
            "bf16_comm": None,
            "heartbeat_interval": None,
        }
    )
    assert env == {"DILOCO_SERVER": "h:1"}
