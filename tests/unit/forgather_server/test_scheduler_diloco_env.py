"""Tests for the DILOCO_* env-var translation in scheduler._build_training."""

from __future__ import annotations

import pytest

from forgather_server.scheduler import _diloco_env_from_job_params

# Stable queue_id used in most tests. Distinct from the worker_id default
# now that the scheduler mints a memorable name rather than falling back
# to the queue_id when no worker_id is set.
QID = "q-test-1234"


@pytest.fixture
def stable_generate_name(monkeypatch):
    """Pin ``generate_name`` for the worker_id-default tests.

    The real generator returns a random adjective-species pair; pinning it
    lets the tests assert a single concrete value rather than re-derive
    the format. The scheduler imports ``generate_name`` lazily from
    ``forgather.utils`` inside ``_diloco_env_from_job_params``, so the
    monkeypatch has to target that module.
    """
    from forgather import utils as forgather_utils

    monkeypatch.setattr(forgather_utils, "generate_name", lambda: "spectacular-fox")
    return "spectacular-fox"


def test_empty_dict_emits_nothing():
    assert _diloco_env_from_job_params({}, QID) == {}


def test_missing_server_addr_short_circuits():
    # Server addr is the gate — even if other keys are set, no env
    # vars get emitted when the worker is opting out of DiLoCo entirely.
    assert _diloco_env_from_job_params({"heartbeat_interval": 15}, QID) == {}


def test_server_addr_alone_yields_minimal_env(stable_generate_name):
    # When DiLoCo is on but no operator-supplied worker_id is present,
    # DILOCO_WORKER_ID falls back to a freshly-minted memorable name
    # (``spectacular-fox``-style) — not the queue_id. The queue route
    # normally fills the default at submission time; this fallback
    # covers any path that bypasses the route.
    env = _diloco_env_from_job_params({"server_addr": "host:8512"}, QID)
    assert env == {
        "DILOCO_SERVER": "host:8512",
        "DILOCO_WORKER_ID": stable_generate_name,
    }


def test_full_payload_forwards_only_client_local_keys():
    # sync_every / num_fragments / dylu / bf16_comm are server-authoritative
    # now — the worker reads them from /info, so the scheduler must NOT
    # forward them even when the (legacy) submission carries them.
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
        "DILOCO_HEARTBEAT_INTERVAL": "15.0",
        "DILOCO_WORKER_ID": "w1",
    }


def test_server_authoritative_keys_never_forwarded():
    # Even alone, the must-match settings are not translated to env vars.
    env = _diloco_env_from_job_params(
        {"server_addr": "h:1", "sync_every": 500, "dylu": True, "bf16_comm": True},
        QID,
    )
    assert "DILOCO_SYNC_EVERY" not in env
    assert "DILOCO_DYLU" not in env
    assert "DILOCO_BF16_COMM" not in env
    assert "DILOCO_NUM_FRAGMENTS" not in env


def test_empty_worker_id_falls_back_to_generated_name(stable_generate_name):
    # The config-preprocessing-time output dir derivation needs
    # DILOCO_WORKER_ID to be non-empty. An empty operator value falls
    # back to a memorable two-word name rather than being dropped.
    env = _diloco_env_from_job_params({"server_addr": "h:1", "worker_id": ""}, QID)
    assert env["DILOCO_WORKER_ID"] == stable_generate_name


def test_whitespace_worker_id_falls_back_to_generated_name(stable_generate_name):
    env = _diloco_env_from_job_params({"server_addr": "h:1", "worker_id": "   "}, QID)
    assert env["DILOCO_WORKER_ID"] == stable_generate_name


def test_explicit_worker_id_wins_over_default(stable_generate_name):
    env = _diloco_env_from_job_params({"server_addr": "h:1", "worker_id": "alpha"}, QID)
    assert env["DILOCO_WORKER_ID"] == "alpha"
    assert env["DILOCO_WORKER_ID"] != stable_generate_name


def test_none_typed_fields_are_skipped(stable_generate_name):
    # None values shouldn't emit env vars. ``worker_id`` is the one
    # exception: it always gets set (memorable-name fallback). The
    # server-authoritative fields are never forwarded regardless.
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
    assert env == {"DILOCO_SERVER": "h:1", "DILOCO_WORKER_ID": stable_generate_name}


def test_worker_id_default_does_not_contain_queue_id():
    # Regression: the prior implementation leaked the queue_id (a
    # ``q_<timestamp>_<hex>`` string) into the worker identity.
    # Defense-in-depth assertion that the default is at least *not*
    # the queue_id, even without pinning the generator.
    env = _diloco_env_from_job_params({"server_addr": "h:1"}, QID)
    assert env["DILOCO_WORKER_ID"] != QID
    # The default also shouldn't be empty / whitespace.
    assert env["DILOCO_WORKER_ID"].strip()
