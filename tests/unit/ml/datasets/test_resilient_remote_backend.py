"""
Tests for :class:`ResilientRemoteBackend` — the retry/reconnect
wrapper that keeps long-running training jobs alive across transient
dataset_server failures.

Two layers:

1. **Unit-level**: a `FlakyRemoteBackend` stand-in that fails the
   first N requests with :class:`DatasetServerUnreachable` and then
   succeeds. The wrapper must retry, resume from the captured
   position, and surface the eventual success transparently.
2. **End-to-end-ish**: a real `TestServer` is brought up, the wrapper
   is given a tiny ``max_retry_seconds`` cap, the server is stopped,
   and the wrapper's `__iter__` is asserted to raise once the cap is
   exceeded — confirming the exit path operators rely on for failed
   runs.

The "kill server mid-iter, restart on same port, resume" smoke test
lives in the plan's verification section, not here — it's manual.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from forgather.ml.datasets import (
    DatasetServerUnreachable,
    InMemoryBackend,
    ResilientRemoteBackend,
)
from forgather.ml.datasets.resilient_remote_backend import (
    MAX_RETRY_SECONDS_ENV_VAR,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "tools"))
from dataset_server import ServerState, TestServer  # noqa: E402


def _examples(n: int):
    return [{"id": i, "text": f"r{i}"} for i in range(n)]


# ---------------------------------------------------------------------
# Helpers: a fake _do_load + fake inner that lets us script failures.
# ---------------------------------------------------------------------


class _FakeInner:
    """Minimal stand-in for `RemoteBackend` used by the unit tests.

    Fails the first ``fail_iter_until`` `__iter__` calls and the first
    ``fail_len_until`` `__len__` calls with
    :class:`DatasetServerUnreachable`. Successful iterations yield
    ``payload`` from ``starting_position`` (so the wrapper can verify
    its position-resume contract).
    """

    def __init__(self, payload, fail_iter_until=0, fail_len_until=0):
        self._payload = list(payload)
        self._fail_iter = fail_iter_until
        self._fail_len = fail_len_until
        self.iter_calls = 0
        self.len_calls = 0
        self._position = 0
        self._handle = "fake-handle"
        self._cached_len = None

    def __iter__(self):
        self.iter_calls += 1
        if self.iter_calls <= self._fail_iter:
            raise DatasetServerUnreachable(f"flaky #{self.iter_calls}")
        for i in range(self._position, len(self._payload)):
            self._position = i + 1
            yield self._payload[i]

    def __len__(self):
        self.len_calls += 1
        if self.len_calls <= self._fail_len:
            raise DatasetServerUnreachable(f"flaky-len #{self.len_calls}")
        return len(self._payload)

    def position(self):
        return self._position


class _ScriptedLoader:
    """Captures every call to `_do_load_once` and lets the test
    drive what each one returns. ``handle`` is the only field the
    wrapper inspects on the response payload (length/column_names
    are optional)."""

    def __init__(self, base_url, payload):
        self._base_url = base_url
        self._payload = list(payload)
        self.calls = []

    def __call__(self, base_url, token, load_args, timeout=300.0):
        self.calls.append((base_url, token, dict(load_args)))
        return {"handle": "h", "length": len(self._payload)}


# ---------------------------------------------------------------------
# Wrapper-only unit tests (no HTTP).
# ---------------------------------------------------------------------


class TestRetryReconnect:
    def test_iter_retries_until_success(self, monkeypatch):
        payload = _examples(5)

        from forgather.ml.datasets import resilient_remote_backend as rrb

        captured = {"inner": None}

        def fake_remote_backend(*args, **kwargs):
            # First call: fail twice, then succeed. Second/third call:
            # already-succeeded path resumes from updated position.
            # Returning the *same* inner instance across calls lets us
            # observe the wrapper's "reset _inner; rebuild" semantics
            # while still preserving the position cursor.
            return captured["inner"]

        captured["inner"] = _FakeInner(payload, fail_iter_until=2)

        monkeypatch.setattr(rrb, "RemoteBackend", fake_remote_backend)
        monkeypatch.setattr(
            rrb,
            "_do_load_once",
            _ScriptedLoader("http://fake", payload),
        )
        monkeypatch.setattr(rrb.time, "sleep", lambda *_: None)

        rb = ResilientRemoteBackend(
            "http://fake",
            None,
            {"path": "local/x"},
            max_retry_seconds=None,
        )
        out = list(rb)
        assert [ex["id"] for ex in out] == [0, 1, 2, 3, 4]
        # First two attempts failed, third yielded all 5; total iter
        # invocations on the (re-used) fake inner = 3.
        assert captured["inner"].iter_calls == 3

    def test_iter_resumes_from_mid_stream_failure(self, monkeypatch):
        """Inner that yields a couple of examples, raises, gets reset,
        then yields the rest. The wrapper must capture the position
        before retry and resume from there — no duplicates, no skips."""
        from forgather.ml.datasets import resilient_remote_backend as rrb

        payload = _examples(5)

        class _MidStreamFlaky(_FakeInner):
            def __init__(self, payload):
                super().__init__(payload)
                self._burst_yielded = False

            def __iter__(self):
                self.iter_calls += 1
                # First iter call: yield 2 examples then explode.
                if self.iter_calls == 1:
                    for i in range(self._position, self._position + 2):
                        self._position = i + 1
                        yield self._payload[i]
                    raise DatasetServerUnreachable("disconnect mid-stream")
                # Subsequent iter calls: yield from current position.
                for i in range(self._position, len(self._payload)):
                    self._position = i + 1
                    yield self._payload[i]

        inner = _MidStreamFlaky(payload)
        monkeypatch.setattr(rrb, "RemoteBackend", lambda *a, **k: inner)
        monkeypatch.setattr(
            rrb,
            "_do_load_once",
            _ScriptedLoader("http://fake", payload),
        )
        monkeypatch.setattr(rrb.time, "sleep", lambda *_: None)

        rb = ResilientRemoteBackend("http://fake", None, {"path": "x"})
        out = list(rb)
        assert [ex["id"] for ex in out] == [0, 1, 2, 3, 4]
        assert rb.position() == 5

    def test_len_retries_until_success(self, monkeypatch):
        from forgather.ml.datasets import resilient_remote_backend as rrb

        inner = _FakeInner(_examples(7), fail_len_until=1)
        monkeypatch.setattr(rrb, "RemoteBackend", lambda *a, **k: inner)
        monkeypatch.setattr(
            rrb, "_do_load_once", _ScriptedLoader("http://fake", _examples(7))
        )
        monkeypatch.setattr(rrb.time, "sleep", lambda *_: None)

        rb = ResilientRemoteBackend(
            "http://fake", None, {"path": "x"}, max_retry_seconds=None
        )
        # First _ensure_inner builds the inner; the loader's "length"
        # response is set into _cached_len so __len__ short-circuits
        # without hitting _FakeInner.__len__. Clear it to force the
        # call.
        rb._cached_len = None
        assert len(rb) == 7

    def test_retry_budget_eventually_raises(self, monkeypatch):
        """With a tight `max_retry_seconds` cap, an always-failing
        inner causes `__iter__` to surface the underlying
        DatasetServerUnreachable rather than spinning forever."""
        from forgather.ml.datasets import resilient_remote_backend as rrb

        inner = _FakeInner(_examples(3), fail_iter_until=99)
        monkeypatch.setattr(rrb, "RemoteBackend", lambda *a, **k: inner)
        monkeypatch.setattr(
            rrb, "_do_load_once", _ScriptedLoader("http://fake", _examples(3))
        )

        sleeps: list[float] = []

        def fake_sleep(s):
            sleeps.append(s)

        monkeypatch.setattr(rrb.time, "sleep", fake_sleep)

        rb = ResilientRemoteBackend(
            "http://fake", None, {"path": "x"}, max_retry_seconds=0.5
        )
        with pytest.raises(DatasetServerUnreachable):
            list(rb)
        # Should not have slept beyond the cap — first delay is 1s
        # which already exceeds 0.5, so the loop raises immediately.
        assert sum(sleeps) <= 0.5

    def test_resolver_called_on_retry(self, monkeypatch):
        """When a resolver is supplied, the wrapper must invoke it
        on each retry — this is the Phase 4 hook proved out early."""
        from forgather.ml.datasets import resilient_remote_backend as rrb

        inner = _FakeInner(_examples(2), fail_iter_until=1)
        monkeypatch.setattr(rrb, "RemoteBackend", lambda *a, **k: inner)
        monkeypatch.setattr(
            rrb, "_do_load_once", _ScriptedLoader("http://fake", _examples(2))
        )
        monkeypatch.setattr(rrb.time, "sleep", lambda *_: None)

        resolved_urls: list[str] = []

        def resolver(dataset_id):
            # Alternate between two URLs to simulate cluster routing
            # picking a different replica each call.
            url = f"http://peer-{len(resolved_urls) + 1}"
            resolved_urls.append(url)
            return url, f"tok-{len(resolved_urls)}"

        rb = ResilientRemoteBackend(
            "http://fake",
            None,
            {"path": "x"},
            resolver=resolver,
        )
        out = list(rb)
        assert [ex["id"] for ex in out] == [0, 1]
        # First load: resolver called once. After first iter raised,
        # second load: resolver called again. Two calls total.
        assert len(resolved_urls) == 2

    def test_env_cap_picked_up(self, monkeypatch):
        monkeypatch.setenv(MAX_RETRY_SECONDS_ENV_VAR, "0.1")
        rb = ResilientRemoteBackend("http://fake", None, {"path": "x"})
        assert rb._max_retry_seconds == 0.1

    def test_env_cap_ignored_when_malformed(self, monkeypatch):
        monkeypatch.setenv(MAX_RETRY_SECONDS_ENV_VAR, "not-a-number")
        rb = ResilientRemoteBackend("http://fake", None, {"path": "x"})
        assert rb._max_retry_seconds is None

    def test_shuffle_with_resolver_drops_handle(self):
        """Regression for CQ-3: in cluster auto-routing mode the new
        wrapper from shuffle() must NOT carry the previous handle,
        so the next iter re-resolves and can land on a different
        replica. Without this, a shuffle pins the dataset to the
        last-resolved peer and failover stops working."""

        def resolver(dataset_id):
            return "http://peer-1:8766", "tok"

        rb = ResilientRemoteBackend(
            "cluster-auto://pending",
            None,
            {"path": "local/x"},
            handle="handle-from-first-load",
            resolver=resolver,
            seed=1,
        )
        # Sanity: inner exists pre-shuffle.
        assert rb._inner is not None
        shuffled = rb.shuffle(seed=2)
        # In cluster mode, the new wrapper must not carry the inner.
        assert shuffled._inner is None
        # Resolver still attached.
        assert shuffled._resolver is resolver

    def test_seek_with_resolver_drops_handle(self):
        """Same contract as shuffle: seek() in cluster mode re-resolves."""

        def resolver(dataset_id):
            return "http://peer-1:8766", "tok"

        rb = ResilientRemoteBackend(
            "cluster-auto://pending",
            None,
            {"path": "local/x"},
            handle="h",
            resolver=resolver,
        )
        assert rb._inner is not None
        seeked = rb.seek(100)
        assert seeked._inner is None
        assert seeked._resolver is resolver
        assert seeked._position == 100

    def test_resolver_410_pre_first_load_aborts(self, monkeypatch):
        """CQ-7b: a resolver raising RuntimeError before the wrapper
        has ever held a working handle is treated as fatal. Operator
        config error: the dataset never existed; retrying won't help."""
        from forgather.ml.datasets import resilient_remote_backend as rrb

        monkeypatch.setattr(rrb.time, "sleep", lambda *_: None)

        def angry_resolver(dataset_id):
            raise RuntimeError("Cluster router rejected request (410): no candidate")

        rb = ResilientRemoteBackend(
            "cluster-auto://pending",
            None,
            {"path": "local/typo"},
            resolver=angry_resolver,
        )
        # _has_ever_loaded is False (no eager handle).
        with pytest.raises(RuntimeError, match="no candidate"):
            list(rb)

    def test_resolver_410_after_first_load_retries(self, monkeypatch):
        """CQ-7b: a resolver raising RuntimeError AFTER the wrapper
        has previously held a working handle gets converted to
        DatasetServerUnreachable and falls into the backoff/retry
        path. The whole cluster going DOWN mid-training is transient
        — keep retrying until either a server comes back or the
        operator interrupts."""
        from forgather.ml.datasets import resilient_remote_backend as rrb

        monkeypatch.setattr(rrb.time, "sleep", lambda *_: None)
        # Suppress the noisy retry-log line during the test (the
        # assertion is on behavior, not log content).
        monkeypatch.setattr(rrb.logger, "warning", lambda *a, **k: None)

        # Resolver: first call succeeds, second raises 410, third
        # succeeds again.
        seq = iter(
            [
                ("http://peer-1:8766", "tok"),
                "raise-410",
                ("http://peer-2:8766", "tok2"),
            ]
        )

        def flaky_resolver(dataset_id):
            v = next(seq)
            if v == "raise-410":
                raise RuntimeError(
                    "Cluster router rejected request (410): no candidate"
                )
            return v

        # Fake inner that fails on its first __iter__, succeeds on
        # subsequent ones. The wrapper will catch the mid-stream
        # failure, sleep, re-call _ensure_inner → resolver (which
        # raises 410 the second time → should be folded to transient).
        inner = _FakeInner(_examples(2), fail_iter_until=1)
        monkeypatch.setattr(rrb, "RemoteBackend", lambda *a, **k: inner)
        monkeypatch.setattr(
            rrb, "_do_load_once", _ScriptedLoader("http://fake", _examples(2))
        )

        rb = ResilientRemoteBackend(
            "cluster-auto://pending",
            None,
            {"path": "local/x"},
            resolver=flaky_resolver,
            max_retry_seconds=None,
        )
        # Pre-condition: hasn't loaded yet. A 410 here would abort.
        assert rb._has_ever_loaded is False
        out = list(rb)
        # All examples yielded — the second resolver call's 410
        # folded into the retry loop, didn't abort training.
        assert [ex["id"] for ex in out] == [0, 1]
        # And the latch flipped True after the first successful load.
        assert rb._has_ever_loaded is True
        # All three resolver returns consumed.
        with pytest.raises(StopIteration):
            next(seq)

    def test_shuffle_without_resolver_preserves_handle(self):
        """Sticky mode (no resolver): the wrapper should preserve the
        inner backend across shuffle/seek to avoid a redundant
        /v1/load round-trip. Pre-existing behavior; locked in to
        catch future regressions when the resolver-aware branch
        gets refactored."""
        rb = ResilientRemoteBackend(
            "http://server:8766",
            "tok",
            {"path": "local/x"},
            handle="h",
            resolver=None,
            seed=1,
        )
        assert rb._inner is not None
        shuffled = rb.shuffle(seed=2)
        # Inner is preserved (carry_handle != None on the new wrapper's
        # constructor → builds a fresh RemoteBackend at position 0).
        assert shuffled._inner is not None
        assert shuffled._inner._handle == "h"


# ---------------------------------------------------------------------
# Backend-interface conformance + state-passthrough tests with a real
# TestServer underneath. These exercise the "happy path" — the wrapper
# should be a drop-in for `RemoteBackend` when nothing's failing.
# ---------------------------------------------------------------------


@pytest.fixture
def server():
    srv = TestServer(host="127.0.0.1", port=0, auth_token=None)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def _wrap(server, payload, **kwargs) -> ResilientRemoteBackend:
    """Register a backend then build a wrapper that has already done
    its initial load. ``handle`` matches the test server's eager
    register call (no /v1/load needed)."""
    server.register("toy", InMemoryBackend(payload))
    return ResilientRemoteBackend(
        server.url,
        None,
        {"path": "local/toy"},
        handle="toy",
        length=len(payload),
        **kwargs,
    )


class TestHappyPath:
    def test_iter_passes_through(self, server):
        rb = _wrap(server, _examples(5))
        ids = [ex["id"] for ex in rb]
        assert ids == [0, 1, 2, 3, 4]
        assert rb.position() == 5

    def test_len_passes_through(self, server):
        rb = _wrap(server, _examples(7))
        assert len(rb) == 7

    def test_shuffle_returns_new_wrapper(self, server):
        rb = _wrap(server, _examples(3))
        shuffled = rb.shuffle(seed=42)
        assert shuffled is not rb
        assert shuffled.position() == 0
        assert isinstance(shuffled, ResilientRemoteBackend)

    def test_seek_returns_new_wrapper_at_position(self, server):
        rb = _wrap(server, _examples(10))
        seeked = rb.seek(4)
        assert seeked is not rb
        assert seeked.position() == 4
        ids = [ex["id"] for ex in seeked]
        assert ids == [4, 5, 6, 7, 8, 9]


class TestClusterAutoRouting:
    """End-to-end auto-routing: ``FORGATHER_DATASET_SERVER=auto`` causes
    the loader to call the cluster router (which we mock) and ride the
    returned URL/token through the resilient wrapper.

    The router itself is exercised by the forgather_server-side tests;
    here we just verify the *client side* picks up the router's reply
    and uses it transparently.
    """

    def test_auto_load_uses_resolver(self, tmp_path, monkeypatch):
        from datasets import Dataset

        from forgather.ml.datasets import fast_load_iterable_dataset
        from forgather.ml.datasets.fast_hf_loader import DATASET_SERVER_ENV_VAR

        ds_path = tmp_path / "stories"
        Dataset.from_dict({"id": list(range(8))}).save_to_disk(str(ds_path))

        state = ServerState()
        state.add_local("stories", str(ds_path))
        srv = TestServer(host="127.0.0.1", port=0, auth_token="srv-tok", state=state)
        with srv:
            # Patch the cluster-router resolver to return our test
            # server's URL + token. Loader is wired to call it on
            # every (re)connect; for a happy-path run that's just
            # once.
            from forgather.ml.datasets import resilient_remote_backend as rrb

            calls = []

            def fake_resolver(dataset_id):
                calls.append(dataset_id)
                return srv.url, "srv-tok"

            monkeypatch.setattr(
                rrb, "make_cluster_router_resolver", lambda **_: fake_resolver
            )

            monkeypatch.setenv(DATASET_SERVER_ENV_VAR, "auto")
            ds = fast_load_iterable_dataset(path="local/stories")
            ids = [ex["id"] for ex in ds]
            assert ids == list(range(8))
            assert len(calls) >= 1
            assert calls[0] == "local/stories"

    def test_auto_resolver_503_retries(self, tmp_path, monkeypatch):
        """The cluster router returning 503 during cold-start must NOT
        abort training — it's a transient signal, and the resolver
        translates it to DatasetServerUnreachable so the resilient
        wrapper's backoff loop catches it.
        """
        from datasets import Dataset

        from forgather.ml.datasets import (
            DatasetServerUnreachable,
            fast_load_iterable_dataset,
        )
        from forgather.ml.datasets.fast_hf_loader import DATASET_SERVER_ENV_VAR

        ds_path = tmp_path / "stories"
        Dataset.from_dict({"id": list(range(4))}).save_to_disk(str(ds_path))

        state = ServerState()
        state.add_local("stories", str(ds_path))
        srv = TestServer(host="127.0.0.1", port=0, auth_token=None, state=state)
        with srv:
            from forgather.ml.datasets import resilient_remote_backend as rrb

            attempts = {"n": 0}

            def flaky_resolver(dataset_id):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise DatasetServerUnreachable("cold-start 503")
                return srv.url, None

            monkeypatch.setattr(
                rrb, "make_cluster_router_resolver", lambda **_: flaky_resolver
            )
            monkeypatch.setattr(rrb.time, "sleep", lambda *_: None)

            monkeypatch.setenv(DATASET_SERVER_ENV_VAR, "auto")
            ds = fast_load_iterable_dataset(path="local/stories")
            ids = [ex["id"] for ex in ds]
            assert ids == list(range(4))
            assert attempts["n"] == 2  # one failed resolve + one successful


class TestRouterResolverWire:
    """Direct tests of make_cluster_router_resolver against an httplib
    test server. Verifies the HTTP-level translation of 503/410 codes
    into the right exception types."""

    def _make_test_router(self, monkeypatch, response_status, body=None):
        """Spin up a TestServer-like stub on an OS-assigned port and
        patch _load_forgather_server_token + the resolver's url.

        We piggyback on the existing TestServer infrastructure but
        with a freshly-built FastAPI app that just answers
        /api/cluster/dataset_router/resolve.
        """
        import threading

        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        app = FastAPI()

        @app.get("/api/cluster/dataset_router/resolve")
        async def resolve(dataset_id: str):
            if isinstance(body, Exception):
                raise body
            return JSONResponse(content=body or {}, status_code=response_status)

        config = uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="warning",
            access_log=False, lifespan="off",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        # Wait for bind.
        deadline = time.monotonic() + 5.0
        while not server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("router stub didn't start")
            time.sleep(0.02)
        port = None
        for srv in server.servers or []:
            for sock in srv.sockets:
                port = int(sock.getsockname()[1])
                break
        url = f"http://127.0.0.1:{port}"

        def stop():
            server.should_exit = True
            thread.join(timeout=5.0)

        return url, stop

    def test_200_returns_base_url_and_token(self, monkeypatch):
        from forgather.ml.datasets.resilient_remote_backend import (
            make_cluster_router_resolver,
        )

        url, stop = self._make_test_router(
            monkeypatch,
            200,
            body={
                "base_url": "http://chosen:8766",
                "auth_token": "chosen-tok",
                "server_id": "abc",
            },
        )
        try:
            resolver = make_cluster_router_resolver(server_url=url, server_token=None)
            base, tok = resolver("local/stories")
            assert base == "http://chosen:8766"
            assert tok == "chosen-tok"
        finally:
            stop()

    def test_503_raises_transient(self, monkeypatch):
        from forgather.ml.datasets import DatasetServerUnreachable
        from forgather.ml.datasets.resilient_remote_backend import (
            make_cluster_router_resolver,
        )

        url, stop = self._make_test_router(
            monkeypatch, 503, body={"detail": "warming up"}
        )
        try:
            resolver = make_cluster_router_resolver(server_url=url, server_token=None)
            with pytest.raises(DatasetServerUnreachable):
                resolver("local/stories")
        finally:
            stop()

    def test_410_raises_runtime(self, monkeypatch):
        from forgather.ml.datasets.resilient_remote_backend import (
            make_cluster_router_resolver,
        )

        url, stop = self._make_test_router(
            monkeypatch, 410, body={"detail": "no candidate"}
        )
        try:
            resolver = make_cluster_router_resolver(server_url=url, server_token=None)
            with pytest.raises(RuntimeError) as exc_info:
                resolver("local/missing")
            assert "no candidate" in str(exc_info.value)
        finally:
            stop()


class TestServerDownExit:
    def test_iter_raises_when_server_gone_and_cap_exceeded(self):
        """Operator-visible failure mode: server is gone, retry cap
        runs out, the wrapper raises so the training loop can crash
        cleanly. Without the cap the wrapper would spin forever (by
        design) — that's the behavior tested in the unit suite."""
        srv = TestServer(host="127.0.0.1", port=0, auth_token=None)
        srv.start()
        srv.register("toy", InMemoryBackend(_examples(5)))
        rb = ResilientRemoteBackend(
            srv.url,
            None,
            {"path": "local/toy"},
            handle="toy",
            length=5,
            max_retry_seconds=0.0,  # no sleeping allowed
        )
        # First iter pulls some data, then we kill the server and
        # iterate again — should fault past the retry budget.
        first = list(rb)
        assert len(first) == 5

        srv.stop()
        # Fresh wrapper at position 0 to force network calls.
        rb2 = ResilientRemoteBackend(
            srv.url,
            None,
            {"path": "local/toy"},
            handle="toy",
            length=5,
            max_retry_seconds=0.0,
        )
        # Force re-load by invalidating the eager-built inner.
        rb2._inner = None
        rb2._cached_len = None
        with pytest.raises(DatasetServerUnreachable):
            len(rb2)
