"""
Round-trip tests for RemoteBackend against an in-process
DatasetServer.

We host an InMemoryBackend on a localhost server (OS-assigned port)
and exercise the proxy via both the bare backend interface AND the
ComposableIterableDataset wrapper. The point is to validate that the
abstract IterableDatasetBackend contract is sufficient for a
real-world remote backend — every higher-level operation (map, slice,
shard, state_dict) should "just work" through the wrapper without
the proxy needing to know about them.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from dataset_server import DatasetServer  # noqa: E402


def _examples(n: int):
    return [{"id": i, "text": f"example_{i}", "tags": [i, i * 2]} for i in range(n)]


@pytest.fixture
def server():
    """Spin up a server on an OS-assigned port for this test."""
    srv = DatasetServer(host="127.0.0.1", port=0)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def _client(server: DatasetServer, handle: str, **kwargs) -> RemoteBackend:
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
        # Position has advanced to end.
        assert client.position() == 10

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
        assert a != list(range(20))  # actually permuted
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
        # Resume from where we are by seeking.
        rest = list(client.seek(client.position()))
        assert [ex["id"] for ex in first_three + rest] == list(range(20))


# ---------------------------------------------------------------------
# Proxy matches direct backend behavior on the same data
# ---------------------------------------------------------------------


class TestRemoteMatchesLocal:
    def test_iter_order_matches(self, server):
        local = InMemoryBackend(_examples(50))
        server.register("match", local)
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
    """All the higher-level ops should work through the wrapper without
    the proxy needing to know about them."""

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

        # Reconstruct a fresh wrapper and resume from saved state.
        ds2 = ComposableIterableDataset(_client(server, "w"))
        ds2.load_state_dict(state)
        rest = list(ds2)
        full = [ex["id"] for ex in partial + rest]
        assert full == list(range(30))

    def test_state_dict_roundtrip_with_shuffle_and_slice(self, server):
        server.register("w", InMemoryBackend(_examples(60)))
        ds = (
            ComposableIterableDataset(_client(server, "w"))
            .shuffle(seed=11, buffer_size=0)
            .slice(5, 50)
        )
        # Reference full sequence for the same configuration.
        ref = list(
            ComposableIterableDataset(_client(server, "w"))
            .shuffle(seed=11, buffer_size=0)
            .slice(5, 50)
        )
        # Partial-then-resume should match.
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
# Server lifecycle / error-handling sanity
# ---------------------------------------------------------------------


class TestServerLifecycle:
    def test_unknown_handle_404(self, server):
        with pytest.raises(Exception):
            len(_client(server, "missing"))

    def test_handles_listing(self, server):
        server.register("a", InMemoryBackend(_examples(1)))
        server.register("b", InMemoryBackend(_examples(2)))
        assert server.list_handles() == ["a", "b"]

    def test_context_manager(self):
        srv = DatasetServer(host="127.0.0.1", port=0)
        srv.register("ctx", InMemoryBackend(_examples(3)))
        with srv:
            client = RemoteBackend(srv.url, "ctx")
            assert len(client) == 3
        # After exit the server is stopped — further requests fail.
        with pytest.raises(Exception):
            len(RemoteBackend(srv.url, "ctx"))


# ---------------------------------------------------------------------
# Load-on-demand (POST /v1/load) and FORGATHER_DATASET_SERVER routing
# ---------------------------------------------------------------------


class TestLoadOnDemand:
    def test_post_load_disabled_by_default(self, server):
        """Without allow_load=True, /v1/load returns 403."""
        import json
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        req = Request(
            f"{server.url}/v1/load",
            data=json.dumps({"path": "wikitext"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as ei:
            urlopen(req, timeout=5).read()
        assert ei.value.code == 403

    def test_post_load_requires_path(self):
        import json
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        srv = DatasetServer(host="127.0.0.1", port=0, allow_load=True)
        with srv:
            req = Request(
                f"{srv.url}/v1/load",
                data=json.dumps({}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(HTTPError) as ei:
                urlopen(req, timeout=5).read()
            assert ei.value.code == 400

    def test_load_on_demand_serves_local_dataset(self, tmp_path):
        """End-to-end: server lazy-loads via _local_load_iterable_dataset
        and returns a working handle."""
        import json
        from urllib.request import Request, urlopen

        # Build a tiny on-disk dataset to avoid network in tests.
        from datasets import Dataset

        ds_path = tmp_path / "tiny"
        Dataset.from_dict(
            {"id": list(range(15)), "text": [f"row_{i}" for i in range(15)]}
        ).save_to_disk(str(ds_path))

        srv = DatasetServer(host="127.0.0.1", port=0, allow_load=True)
        with srv:
            req = Request(
                f"{srv.url}/v1/load",
                data=json.dumps({"path": str(ds_path)}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = json.loads(urlopen(req, timeout=10).read().decode("utf-8"))
            assert resp["length"] == 15
            handle = resp["handle"]

            # Now read examples through the returned handle.
            client = RemoteBackend(srv.url, handle)
            assert len(client) == 15
            assert [ex["id"] for ex in client] == list(range(15))

    def test_env_var_routing(self, tmp_path, monkeypatch):
        """fast_load_iterable_dataset routes through the server when
        FORGATHER_DATASET_SERVER is set; output should match the
        local-load result."""
        from datasets import Dataset
        from forgather.ml.datasets import fast_load_iterable_dataset
        from forgather.ml.datasets.fast_hf_loader import (
            DATASET_SERVER_ENV_VAR,
            _local_load_iterable_dataset,
        )

        # Tiny on-disk dataset.
        ds_path = tmp_path / "tiny"
        Dataset.from_dict(
            {"id": list(range(20)), "text": [f"r{i}" for i in range(20)]}
        ).save_to_disk(str(ds_path))

        # Reference: local load.
        local_ds = _local_load_iterable_dataset(path=str(ds_path))
        local_ids = [ex["id"] for ex in local_ds]

        # Routed through server.
        srv = DatasetServer(host="127.0.0.1", port=0, allow_load=True)
        with srv:
            monkeypatch.setenv(DATASET_SERVER_ENV_VAR, srv.url)
            remote_ds = fast_load_iterable_dataset(path=str(ds_path))
            from forgather.ml.datasets import (
                ComposableIterableDataset,
                RemoteBackend,
            )

            assert isinstance(remote_ds, ComposableIterableDataset)
            assert isinstance(remote_ds.backend, RemoteBackend)
            remote_ids = [ex["id"] for ex in remote_ds]

        assert remote_ids == local_ids

    def test_env_var_routing_with_split_notation(self, tmp_path, monkeypatch):
        """Split-notation slice (e.g. 'train[5:15]') is parsed
        client-side and applied as a wrapper-level slice over the
        server's base-split backend."""
        from datasets import Dataset
        from forgather.ml.datasets import fast_load_iterable_dataset
        from forgather.ml.datasets.fast_hf_loader import DATASET_SERVER_ENV_VAR

        ds_path = tmp_path / "tiny"
        Dataset.from_dict({"id": list(range(30))}).save_to_disk(str(ds_path / "train"))
        # save_to_disk on a single Dataset writes to ds_path itself,
        # not a subdir, so use the root.
        ds_path2 = tmp_path / "tiny2"
        Dataset.from_dict({"id": list(range(30))}).save_to_disk(str(ds_path2))

        srv = DatasetServer(host="127.0.0.1", port=0, allow_load=True)
        with srv:
            monkeypatch.setenv(DATASET_SERVER_ENV_VAR, srv.url)
            ds = fast_load_iterable_dataset(path=str(ds_path2), split="train[5:15]")
            ids = [ex["id"] for ex in ds]
        assert ids == list(range(5, 15))
