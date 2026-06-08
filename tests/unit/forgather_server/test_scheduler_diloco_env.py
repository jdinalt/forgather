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


@pytest.fixture(autouse=True)
def mock_query(monkeypatch):
    """Stub the launch-time ``/info`` query (issue #154).

    With no explicit ``backend`` in the submission, ``_diloco_env_from_job_params``
    derives the backend from the param server's ``/info`` — a real network call.
    Most tests here exercise the env translation, not the network, so by default
    the stub returns a reachable ``http`` server (the common case). Tests that
    care about the *derived* backend set ``holder['info']``; ``holder['error']``
    simulates an unreachable server (the query raising).
    """
    holder = {
        "info": {
            "expected_client_settings": {"backend": "http"},
            "num_workers": 1,
        },
        "error": None,
    }

    import forgather_server.scheduler as sched

    def fake_query(server_addr, queue_id):
        if holder["error"] is not None:
            raise holder["error"]
        return holder["info"]

    monkeypatch.setattr(sched, "_diloco_query_info", fake_query)
    return holder


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


# --- explicit backend (issue #154): the --local-only / back-compat path, where
# the submission carries the backend inline and the scheduler honors it verbatim
# (no /info query, no topology cross-check) --------------------------------


def test_shared_memory_backend_derives_shm_env():
    # The submission carries backend + group id + size; the scheduler turns
    # those into the three env vars the worker reads, deriving a per-submit
    # region dir from the group id under the host temp dir.
    import os
    import tempfile

    env = _diloco_env_from_job_params(
        {
            "server_addr": "h:1",
            "worker_id": "w1",
            "backend": "shared_memory",
            "shm_group_id": "abc123def456",
            "shm_group_size": 3,
        },
        QID,
    )
    assert env["DILOCO_BACKEND"] == "shared_memory"
    assert env["DILOCO_SHM_GROUP_DIR"] == os.path.join(
        tempfile.gettempdir(), "diloco_shm_abc123def456"
    )
    assert env["DILOCO_SHM_GROUP_SIZE"] == "3"
    # The coordination-plane keys are still set.
    assert env["DILOCO_SERVER"] == "h:1"
    assert env["DILOCO_WORKER_ID"] == "w1"


def test_http_backend_emits_no_shm_env():
    # The default / explicit http backend is a no-op for the shm derivation.
    for diloco in (
        {"server_addr": "h:1"},
        {"server_addr": "h:1", "backend": "http"},
    ):
        env = _diloco_env_from_job_params(diloco, QID)
        assert "DILOCO_BACKEND" not in env
        assert "DILOCO_SHM_GROUP_DIR" not in env
        assert "DILOCO_SHM_GROUP_SIZE" not in env


def test_shared_memory_without_group_id_omits_dir():
    # Defensive: a malformed shm submission missing the group id still flags
    # the backend + size but can't derive a region dir (no rendezvous).
    env = _diloco_env_from_job_params(
        {"server_addr": "h:1", "backend": "shared_memory", "shm_group_size": 2},
        QID,
    )
    assert env["DILOCO_BACKEND"] == "shared_memory"
    assert env["DILOCO_SHM_GROUP_SIZE"] == "2"
    assert "DILOCO_SHM_GROUP_DIR" not in env


# --- collective backend (issue #154): one torchrun job; the scheduler derives
# the backend + replicate degree (nproc is the separate job_params.nproc path) ---


def test_collective_backend_derives_replicate_env():
    env = _diloco_env_from_job_params(
        {
            "server_addr": "h:1",
            "worker_id": "run1",
            "backend": "collective",
            "diloco_replicate": 4,
        },
        QID,
    )
    assert env["DILOCO_BACKEND"] == "collective"
    assert env["DILOCO_REPLICATE"] == "4"
    # The coordination-plane keys are still set; no shm keys leak in.
    assert env["DILOCO_SERVER"] == "h:1"
    assert env["DILOCO_WORKER_ID"] == "run1"
    assert "DILOCO_SHM_GROUP_DIR" not in env


def test_collective_without_replicate_omits_it():
    env = _diloco_env_from_job_params(
        {"server_addr": "h:1", "backend": "collective"}, QID
    )
    assert env["DILOCO_BACKEND"] == "collective"
    assert "DILOCO_REPLICATE" not in env


def test_http_backend_emits_no_collective_env():
    env = _diloco_env_from_job_params({"server_addr": "h:1", "backend": "http"}, QID)
    assert "DILOCO_BACKEND" not in env
    assert "DILOCO_REPLICATE" not in env


# --- derived backend (issue #154): the orchestrated default. The submission
# carries NO backend; the scheduler queries the server's /info at launch and
# shapes the env from the declared backend (mocked via the mock_query fixture). --


def test_derived_http_emits_no_backend_env(mock_query):
    # Server declares http -> no DILOCO_BACKEND (trainer defaults to http), no
    # shm / collective env. The control-plane keys are still forwarded.
    mock_query["info"] = {"expected_client_settings": {"backend": "http"}}
    env = _diloco_env_from_job_params({"server_addr": "h:1", "worker_id": "w1"}, QID)
    assert env["DILOCO_SERVER"] == "h:1"
    assert env["DILOCO_WORKER_ID"] == "w1"
    assert "DILOCO_BACKEND" not in env
    assert "DILOCO_SHM_GROUP_DIR" not in env
    assert "DILOCO_REPLICATE" not in env


def test_derived_shared_memory_uses_server_identity_and_worker_count(mock_query):
    # Server declares shared_memory + a worker count; the scheduler derives the
    # group dir from the SERVER (stable, base_url-derived) and the size from the
    # server's num_workers — no submit-time group id involved.
    import os
    import tempfile

    from forgather_server import cluster_diloco_inventory as cdi

    mock_query["info"] = {
        "expected_client_settings": {"backend": "shared_memory"},
        "num_workers": 4,
    }
    server = "https://param.example:8512"
    env = _diloco_env_from_job_params({"server_addr": server, "worker_id": "w1"}, QID)
    assert env["DILOCO_BACKEND"] == "shared_memory"
    expected_id = cdi.server_id_for(cdi._normalize(server))
    assert env["DILOCO_SHM_GROUP_DIR"] == os.path.join(
        tempfile.gettempdir(), f"diloco_shm_{expected_id}"
    )
    assert env["DILOCO_SHM_GROUP_SIZE"] == "4"


def test_derived_shared_memory_same_server_yields_same_group_dir(mock_query):
    # Two co-located workers against the same server compute the SAME region dir
    # without any shared submit-time id (the server's identity is the group key).
    mock_query["info"] = {
        "expected_client_settings": {"backend": "shared_memory"},
        "num_workers": 2,
    }
    server = "https://param.example:8512"
    a = _diloco_env_from_job_params({"server_addr": server, "worker_id": "a"}, QID)
    b = _diloco_env_from_job_params({"server_addr": server, "worker_id": "b"}, QID)
    assert a["DILOCO_SHM_GROUP_DIR"] == b["DILOCO_SHM_GROUP_DIR"]


def test_derived_collective_sets_replicate(mock_query):
    # A collective-shaped job (diloco_replicate set) against a server that
    # declares collective derives the backend + replicate degree.
    mock_query["info"] = {"expected_client_settings": {"backend": "collective"}}
    env = _diloco_env_from_job_params(
        {"server_addr": "h:1", "worker_id": "run1", "diloco_replicate": 3}, QID
    )
    assert env["DILOCO_BACKEND"] == "collective"
    assert env["DILOCO_REPLICATE"] == "3"


def test_derived_collective_job_against_non_collective_server_raises(mock_query):
    # The job is collective-shaped but the server declares http: a fatal
    # topology/backend mismatch (fail the launch).
    mock_query["info"] = {"expected_client_settings": {"backend": "http"}}
    with pytest.raises(RuntimeError, match="collective"):
        _diloco_env_from_job_params({"server_addr": "h:1", "diloco_replicate": 2}, QID)


def test_derived_collective_server_against_worker_job_raises(mock_query):
    # The server declares collective but this was submitted as an independent
    # worker (no replicate): also a fatal mismatch.
    mock_query["info"] = {"expected_client_settings": {"backend": "collective"}}
    with pytest.raises(RuntimeError, match="collective"):
        _diloco_env_from_job_params({"server_addr": "h:1"}, QID)


def test_unreachable_server_raises(mock_query):
    # The backend is server-authoritative: an unreachable server has no safe
    # default, so the query raises and _launch turns it into a failed job.
    mock_query["error"] = RuntimeError("server unreachable")
    with pytest.raises(RuntimeError, match="unreachable"):
        _diloco_env_from_job_params({"server_addr": "h:1"}, QID)
