"""
Round-trip tests for RemoteBackend against an in-process uvicorn-hosted
DatasetServer. Also exercises the policy gates (HF cache / path / local
mappings) and bearer-token auth.

We host an InMemoryBackend on a localhost server (OS-assigned port)
and exercise the proxy via both the bare backend interface AND the
ComposableIterableDataset wrapper. The point is to validate that the
abstract IterableDatasetBackend contract is sufficient for a
real-world remote backend — every higher-level operation (map, slice,
shard, state_dict) should "just work" through the wrapper without
the proxy needing to know about them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from forgather.ml.datasets import (
    ComposableIterableDataset,
    InMemoryBackend,
    IterableDatasetBackend,
    RemoteBackend,
)

# tools/ isn't on sys.path; add it so we can import dataset_server.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "tools"))
from dataset_server import ServerState, TestServer  # noqa: E402


def _examples(n: int):
    return [{"id": i, "text": f"example_{i}", "tags": [i, i * 2]} for i in range(n)]


@pytest.fixture
def server():
    """Spin up a no-auth test server on an OS-assigned port."""
    srv = TestServer(host="127.0.0.1", port=0, auth_token=None)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def _client(server: TestServer, handle: str, **kwargs) -> RemoteBackend:
    return RemoteBackend(server.url, handle, **kwargs)


# ---------------------------------------------------------------------
# Backend interface conformance for the proxy
# ---------------------------------------------------------------------


class TestRemoteBackendConformance:
    def test_implements_interface(self, server):
        server.register("toy", InMemoryBackend(_examples(3)))
        client = _client(server, "toy")
        assert isinstance(client, IterableDatasetBackend)

    def test_len_matches_server(self, server):
        server.register("toy", InMemoryBackend(_examples(7)))
        client = _client(server, "toy")
        assert len(client) == 7

    def test_iter_yields_all(self, server):
        server.register("toy", InMemoryBackend(_examples(10)))
        client = _client(server, "toy")
        ids = [ex["id"] for ex in client]
        assert ids == list(range(10))
        assert client.position() == 10

    def test_iter_position_zero_is_repeatable(self, server):
        """Regression: the server's shared backend instance has its
        position cursor advanced after a /iter call; without an
        unconditional seek, a second /iter?position=0 from a fresh
        client would still see the cursor at end-of-iteration and
        yield nothing. Common path in trainers that re-iterate the
        eval split each evaluation step."""
        server.register("toy", InMemoryBackend(_examples(10)))
        first = list(_client(server, "toy"))
        second = list(_client(server, "toy"))
        assert [e["id"] for e in first] == list(range(10))
        assert [e["id"] for e in second] == list(range(10))

    def test_iter_preserves_complex_values(self, server):
        server.register("toy", InMemoryBackend(_examples(3)))
        client = _client(server, "toy")
        out = list(client)
        assert out[0] == {"id": 0, "text": "example_0", "tags": [0, 0]}
        assert out[1] == {"id": 1, "text": "example_1", "tags": [1, 2]}

    def test_seek_returns_new_instance(self, server):
        server.register("toy", InMemoryBackend(_examples(10)))
        client = _client(server, "toy")
        c2 = client.seek(5)
        assert c2 is not client
        assert client.position() == 0
        assert c2.position() == 5
        assert [ex["id"] for ex in c2] == [5, 6, 7, 8, 9]

    def test_shuffle_changes_order_deterministically(self, server):
        server.register("toy", InMemoryBackend(_examples(20)))
        client = _client(server, "toy")
        a = [ex["id"] for ex in client.shuffle(seed=42)]
        b = [ex["id"] for ex in client.shuffle(seed=42)]
        c = [ex["id"] for ex in client.shuffle(seed=99)]
        assert a == b
        assert sorted(a) == list(range(20))
        assert a != list(range(20))
        assert a != c

    def test_shuffle_then_seek(self, server):
        server.register("toy", InMemoryBackend(_examples(20)))
        client = _client(server, "toy")
        full = [ex["id"] for ex in client.shuffle(seed=42)]
        from_5 = [ex["id"] for ex in client.shuffle(seed=42).seek(5)]
        assert from_5 == full[5:]

    def test_partial_iteration_then_seek_continues(self, server):
        server.register("toy", InMemoryBackend(_examples(20)))
        client = _client(server, "toy")
        it = iter(client)
        first_three = [next(it) for _ in range(3)]
        assert client.position() == 3
        rest = list(client.seek(client.position()))
        assert [ex["id"] for ex in first_three + rest] == list(range(20))


# ---------------------------------------------------------------------
# Proxy matches direct backend behavior on the same data
# ---------------------------------------------------------------------


class TestRemoteMatchesLocal:
    def test_iter_order_matches(self, server):
        server.register("match", InMemoryBackend(_examples(50)))
        remote = _client(server, "match")
        assert [e["id"] for e in remote] == [
            e["id"] for e in InMemoryBackend(_examples(50))
        ]

    def test_shuffle_order_matches(self, server):
        server.register("match", InMemoryBackend(_examples(50)))
        remote_ids = [e["id"] for e in _client(server, "match").shuffle(seed=7)]
        local_ids = [e["id"] for e in InMemoryBackend(_examples(50)).shuffle(7)]
        assert remote_ids == local_ids


# ---------------------------------------------------------------------
# ComposableIterableDataset over the proxy — the whole point
# ---------------------------------------------------------------------


class TestWrapperOverRemote:
    def test_basic_iter(self, server):
        server.register("w", InMemoryBackend(_examples(20)))
        ds = ComposableIterableDataset(_client(server, "w"))
        assert [ex["id"] for ex in ds] == list(range(20))
        assert len(ds) == 20

    def test_slice(self, server):
        server.register("w", InMemoryBackend(_examples(50)))
        ds = ComposableIterableDataset(_client(server, "w")).slice(10, 30)
        assert [ex["id"] for ex in ds] == list(range(10, 30))

    def test_shard_disjoint_complete(self, server):
        server.register("w", InMemoryBackend(_examples(40)))
        ds = ComposableIterableDataset(_client(server, "w"))
        all_ids: set[int] = set()
        for i in range(4):
            shard_ids = [ex["id"] for ex in ds.shard(num_shards=4, index=i)]
            assert set(shard_ids).isdisjoint(all_ids)
            all_ids |= set(shard_ids)
        assert all_ids == set(range(40))

    def test_map(self, server):
        server.register("w", InMemoryBackend(_examples(5)))
        ds = ComposableIterableDataset(_client(server, "w")).map(
            lambda ex: {"squared": ex["id"] ** 2}
        )
        out = list(ds)
        assert [ex["squared"] for ex in out] == [0, 1, 4, 9, 16]

    def test_filter(self, server):
        server.register("w", InMemoryBackend(_examples(10)))
        ds = ComposableIterableDataset(_client(server, "w")).filter(
            lambda ex: ex["id"] % 2 == 0
        )
        assert [ex["id"] for ex in ds] == [0, 2, 4, 6, 8]

    def test_shuffle_with_buffer(self, server):
        server.register("w", InMemoryBackend(_examples(50)))
        ds = ComposableIterableDataset(_client(server, "w")).shuffle(
            seed=42, buffer_size=10
        )
        ids = [ex["id"] for ex in ds]
        assert sorted(ids) == list(range(50))

    def test_state_dict_roundtrip(self, server):
        server.register("w", InMemoryBackend(_examples(30)))
        ds = ComposableIterableDataset(_client(server, "w"))
        it = iter(ds)
        partial = [next(it) for _ in range(8)]
        state = ds.state_dict()

        ds2 = ComposableIterableDataset(_client(server, "w"))
        ds2.load_state_dict(state)
        rest = list(ds2)
        full = [ex["id"] for ex in partial + rest]
        assert full == list(range(30))

    def test_state_dict_roundtrip_with_shuffle_and_slice(self, server):
        server.register("w", InMemoryBackend(_examples(60)))
        ref = list(
            ComposableIterableDataset(_client(server, "w"))
            .shuffle(seed=11, buffer_size=0)
            .slice(5, 50)
        )
        ds = (
            ComposableIterableDataset(_client(server, "w"))
            .shuffle(seed=11, buffer_size=0)
            .slice(5, 50)
        )
        it = iter(ds)
        partial = [next(it) for _ in range(7)]
        state = ds.state_dict()

        ds2 = (
            ComposableIterableDataset(_client(server, "w"))
            .shuffle(seed=11, buffer_size=0)
            .slice(5, 50)
        )
        ds2.load_state_dict(state)
        rest = list(ds2)
        assert partial + rest == ref


# ---------------------------------------------------------------------
# Server lifecycle / error handling
# ---------------------------------------------------------------------


class TestServerLifecycle:
    def test_unknown_handle_404(self, server):
        with pytest.raises(Exception):
            len(_client(server, "missing"))

    def test_handles_listing(self, server):
        server.register("a", InMemoryBackend(_examples(1)))
        server.register("b", InMemoryBackend(_examples(2)))
        assert sorted(server.list_handles()) == ["a", "b"]

    def test_context_manager(self):
        srv = TestServer(host="127.0.0.1", port=0, auth_token=None)
        srv.register("ctx", InMemoryBackend(_examples(3)))
        with srv:
            client = RemoteBackend(srv.url, "ctx")
            assert len(client) == 3
        with pytest.raises(Exception):
            len(RemoteBackend(srv.url, "ctx"))


# ---------------------------------------------------------------------
# Bearer-token auth
# ---------------------------------------------------------------------


class TestAuth:
    def test_no_auth_health_open(self, server):
        # /v1/health should be open even when auth IS required, but
        # this fixture has auth disabled — confirm 200 either way.
        with urlopen(f"{server.url}/v1/health", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["status"] == "ok"

    def test_auth_status_endpoint(self, server):
        with urlopen(f"{server.url}/v1/auth/status", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload == {"auth_required": False}

    def test_with_auth_required_401_without_header(self):
        srv = TestServer(host="127.0.0.1", port=0, auth_token="secret")
        with srv:
            with pytest.raises(HTTPError) as ei:
                urlopen(f"{srv.url}/v1/datasets", timeout=5).read()
            assert ei.value.code == 401

    def test_with_auth_required_401_wrong_token(self):
        srv = TestServer(host="127.0.0.1", port=0, auth_token="secret")
        with srv:
            req = Request(
                f"{srv.url}/v1/datasets",
                headers={"Authorization": "Bearer wrong"},
            )
            with pytest.raises(HTTPError) as ei:
                urlopen(req, timeout=5).read()
            assert ei.value.code == 401

    def test_with_auth_required_200_correct_token(self):
        srv = TestServer(host="127.0.0.1", port=0, auth_token="secret")
        srv.register("a", InMemoryBackend(_examples(2)))
        with srv:
            req = Request(
                f"{srv.url}/v1/datasets",
                headers={"Authorization": "Bearer secret"},
            )
            with urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            assert any(h["handle"] == "a" for h in payload["handles"])

    def test_auth_status_reports_required(self):
        srv = TestServer(host="127.0.0.1", port=0, auth_token="secret")
        with srv:
            with urlopen(f"{srv.url}/v1/auth/status", timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        assert payload == {"auth_required": True}

    def test_remote_backend_passes_token(self):
        srv = TestServer(host="127.0.0.1", port=0, auth_token="hunter2")
        srv.register("a", InMemoryBackend(_examples(5)))
        with srv:
            be = RemoteBackend(srv.url, "a", token="hunter2")
            assert len(be) == 5
            assert [e["id"] for e in be] == list(range(5))


# ---------------------------------------------------------------------
# Loading policy: --no-hf, --allow-paths, local mappings
# ---------------------------------------------------------------------


class TestLoadPolicy:
    def test_post_load_requires_path(self, server):
        req = Request(
            f"{server.url}/v1/load",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as ei:
            urlopen(req, timeout=5).read()
        assert ei.value.code == 400

    def test_path_loading_disabled_by_default(self, server, tmp_path):
        from datasets import Dataset

        ds_path = tmp_path / "tiny"
        Dataset.from_dict({"id": [0, 1, 2]}).save_to_disk(str(ds_path))
        body = json.dumps({"path": str(ds_path)}).encode("utf-8")
        req = Request(
            f"{server.url}/v1/load",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as ei:
            urlopen(req, timeout=10).read()
        assert ei.value.code == 403

    def test_path_loading_with_allow_paths(self, tmp_path):
        from datasets import Dataset

        ds_path = tmp_path / "tiny"
        Dataset.from_dict({"id": list(range(7))}).save_to_disk(str(ds_path))

        state = ServerState(allow_paths=True)
        srv = TestServer(host="127.0.0.1", port=0, auth_token=None, state=state)
        with srv:
            body = json.dumps({"path": str(ds_path)}).encode("utf-8")
            req = Request(
                f"{srv.url}/v1/load",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            assert payload["length"] == 7
            assert payload["source"] == "path"

    def test_local_mapping_resolves(self, tmp_path):
        from datasets import Dataset

        ds_path = tmp_path / "stories"
        Dataset.from_dict({"id": list(range(11))}).save_to_disk(str(ds_path))

        state = ServerState()
        state.add_local("stories", str(ds_path))
        srv = TestServer(host="127.0.0.1", port=0, auth_token=None, state=state)
        with srv:
            body = json.dumps({"path": "local/stories"}).encode("utf-8")
            req = Request(
                f"{srv.url}/v1/load",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            assert payload["length"] == 11
            assert payload["source"] == "local"

            # And the /v1/local diagnostic endpoint reflects it.
            with urlopen(f"{srv.url}/v1/local", timeout=5) as resp:
                local_payload = json.loads(resp.read().decode("utf-8"))
            names = [it["name"] for it in local_payload["local"]]
            assert names == ["stories"]

    def test_unknown_local_404(self):
        srv = TestServer(host="127.0.0.1", port=0, auth_token=None)
        with srv:
            body = json.dumps({"path": "local/missing"}).encode("utf-8")
            req = Request(
                f"{srv.url}/v1/load",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(HTTPError) as ei:
                urlopen(req, timeout=5).read()
            assert ei.value.code == 404

    def test_no_hf_blocks_hf_id(self):
        # With --no-hf we should reject anything that looks like an HF
        # repo id (i.e. doesn't exist on disk and isn't a local/* mapping).
        state = ServerState(hf_cache_enabled=False)
        srv = TestServer(host="127.0.0.1", port=0, auth_token=None, state=state)
        with srv:
            body = json.dumps({"path": "allenai/c4", "name": "en"}).encode("utf-8")
            req = Request(
                f"{srv.url}/v1/load",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(HTTPError) as ei:
                urlopen(req, timeout=5).read()
            assert ei.value.code == 403


# ---------------------------------------------------------------------
# Diagnostic endpoint: /v1/cache/hf
# ---------------------------------------------------------------------


class TestHFCacheEndpoint:
    def test_cache_endpoint_empty(self, server, tmp_path, monkeypatch):
        # Point HF_DATASETS_CACHE at an empty dir so the result is
        # deterministic regardless of what's actually on the test host.
        monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
        with urlopen(f"{server.url}/v1/cache/hf", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["cache_root"] == str(tmp_path)
        assert payload["datasets"] == []

    def test_cache_endpoint_finds_repo(self, server, tmp_path, monkeypatch):
        # Synthesize a fake HF cache layout and verify the walker
        # surfaces it (no real `datasets` calls needed).
        cache = tmp_path / "hf_cache"
        repo_dir = cache / "allenai___c4"
        cfg_dir = repo_dir / "en" / "0.0.0" / "abc123"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "dataset_info.json").write_text(
            json.dumps(
                {
                    "config_name": "en",
                    "splits": {
                        "train": {"name": "train", "num_examples": 1000},
                        "validation": {"name": "validation", "num_examples": 50},
                    },
                }
            )
        )
        # A small placeholder file so size_bytes is nonzero.
        (cfg_dir / "data-00000-of-00001.arrow").write_bytes(b"x" * 128)

        monkeypatch.setenv("HF_DATASETS_CACHE", str(cache))
        with urlopen(f"{server.url}/v1/cache/hf", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        repos = {d["repo"]: d for d in payload["datasets"]}
        assert "allenai/c4" in repos
        configs = repos["allenai/c4"]["configs"]
        assert any(c["config"] == "en" for c in configs)
        en_cfg = next(c for c in configs if c["config"] == "en")
        split_names = {s["name"] for s in en_cfg["splits"]}
        assert split_names == {"train", "validation"}


# ---------------------------------------------------------------------
# Env-var routing through the loader
# ---------------------------------------------------------------------


class TestEnvVarRouting:
    def test_env_var_routes_through_server(self, tmp_path, monkeypatch):
        from datasets import Dataset
        from forgather.ml.datasets import fast_load_iterable_dataset
        from forgather.ml.datasets.fast_hf_loader import (
            DATASET_SERVER_ENV_VAR,
            _local_load_iterable_dataset,
        )

        ds_path = tmp_path / "tiny"
        Dataset.from_dict(
            {"id": list(range(20)), "text": [f"r{i}" for i in range(20)]}
        ).save_to_disk(str(ds_path))

        local_ids = [ex["id"] for ex in _local_load_iterable_dataset(path=str(ds_path))]

        state = ServerState(allow_paths=True)
        srv = TestServer(host="127.0.0.1", port=0, auth_token=None, state=state)
        with srv:
            monkeypatch.setenv(DATASET_SERVER_ENV_VAR, srv.url)
            ds = fast_load_iterable_dataset(path=str(ds_path))
            assert isinstance(ds, ComposableIterableDataset)
            assert isinstance(ds.backend, RemoteBackend)
            assert [ex["id"] for ex in ds] == local_ids

    def test_env_var_with_local_mapping(self, tmp_path, monkeypatch):
        from datasets import Dataset
        from forgather.ml.datasets import fast_load_iterable_dataset
        from forgather.ml.datasets.fast_hf_loader import DATASET_SERVER_ENV_VAR

        ds_path = tmp_path / "stories"
        Dataset.from_dict({"id": list(range(15))}).save_to_disk(str(ds_path))

        state = ServerState()
        state.add_local("stories", str(ds_path))
        srv = TestServer(host="127.0.0.1", port=0, auth_token=None, state=state)
        with srv:
            monkeypatch.setenv(DATASET_SERVER_ENV_VAR, srv.url)
            ds = fast_load_iterable_dataset(path="local/stories")
            assert [ex["id"] for ex in ds] == list(range(15))
