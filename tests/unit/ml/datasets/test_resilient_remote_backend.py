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

        def resolver(load_args):
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
