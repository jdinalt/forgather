"""Tests for the server-managed global token budget (#224).

When the aggregated cross-worker token count reaches ``token_budget``, the server
relays ``save_and_stop`` to every worker (once). Workers run open-ended; this is
the controlling stop for async runs where a per-worker step budget can't bound
the global token count.
"""

import torch

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import DiLoCoServer

from .conftest import make_initial_checkpoint


def _server(tmp_path, **kw):
    sd = {"w": torch.zeros(4)}
    ckpt = make_initial_checkpoint(sd, tmp_path / "init")
    kw.setdefault("num_workers", 2)
    kw.setdefault("save_every_n_rounds", 0)
    return DiLoCoServer(output_dir=str(tmp_path), from_checkpoint=ckpt, port=0, **kw)


def _register(server, *ids):
    clients = []
    for wid in ids:
        c = DiLoCoClient(f"localhost:{server.port}", timeout=20)
        c.register(wid)
        clients.append(c)
    return clients


def _pending(server, *ids):
    return [server._workers[i].pending_command for i in ids]


class TestTokenBudgetTrigger:
    def test_relays_save_and_stop_once_at_threshold(self, tmp_path):
        server = _server(tmp_path, token_budget=1000)
        server.start()
        try:
            _register(server, "w0", "w1")
            server._stats.total_tokens = 999
            server._maybe_trigger_token_budget()
            assert _pending(server, "w0", "w1") == [None, None]  # under budget

            server._stats.total_tokens = 1000  # reaches the budget
            server._maybe_trigger_token_budget()
            assert _pending(server, "w0", "w1") == ["save_and_stop", "save_and_stop"]
            assert server._budget_stop_sent

            # Clear and re-evaluate: the one-shot guard prevents a re-broadcast.
            for w in server._workers.values():
                w.pending_command = None
            server._maybe_trigger_token_budget()
            assert _pending(server, "w0", "w1") == [None, None]
        finally:
            server.stop()

    def test_heartbeat_drives_trigger(self, tmp_path):
        """End-to-end: a heartbeat whose stats push total_tokens to the budget
        triggers the stop, delivered on the worker's next heartbeat."""
        server = _server(tmp_path, token_budget=1000)
        server.start()
        try:
            (c0,) = _register(server, "w0")
            # First report establishes the cumulative; 500 < budget.
            c0.heartbeat("w0", stats={"tokens_total": 500})
            assert server._workers["w0"].pending_command is None
            # Crosses the budget (delta 500 -> total 1000); trigger queues the stop.
            c0.heartbeat("w0", stats={"tokens_total": 1000})
            # The command rides the NEXT heartbeat (it was read before the trigger).
            resp = c0.heartbeat("w0", stats={"tokens_total": 1000})
            assert resp.get("command") == "save_and_stop"
        finally:
            server.stop()

    def test_disabled_is_noop(self, tmp_path):
        server = _server(tmp_path, token_budget=0)
        server.start()
        try:
            _register(server, "w0")
            server._stats.total_tokens = 10**9
            server._maybe_trigger_token_budget()
            assert _pending(server, "w0") == [None]
        finally:
            server.stop()

    def test_retriggers_after_restart(self, tmp_path):
        """The one-shot guard isn't persisted, so a server still over budget on
        restart re-sends the stop (total_tokens persists in the aggregate)."""
        server = _server(tmp_path, token_budget=500)
        server.start()
        try:
            _register(server, "w0")
            server._stats.total_tokens = 500
            server._maybe_trigger_token_budget()
            assert _pending(server, "w0") == ["save_and_stop"]
            # Simulate a restart: guard reset, total_tokens restored from ckpt.
            server._budget_stop_sent = False
            server._workers["w0"].pending_command = None
            server._maybe_trigger_token_budget()
            assert _pending(server, "w0") == ["save_and_stop"]
        finally:
            server.stop()


class TestTokenBudgetRuntimeControl:
    def test_set_below_total_stops_now(self, tmp_path):
        server = _server(tmp_path, token_budget=0)
        server.start()
        try:
            (c0,) = _register(server, "w0")
            server._stats.total_tokens = 5000
            resp = c0.set_token_budget(1000)  # below the total -> stop now
            assert resp["token_budget"] == 1000
            assert resp["budget_stop_sent"] is True
            assert server.token_budget == 1000
            assert _pending(server, "w0") == ["save_and_stop"]
        finally:
            server.stop()

    def test_set_zero_disables(self, tmp_path):
        """Setting the budget to 0 at runtime disables it (no further stops)."""
        server = _server(tmp_path, token_budget=1000)
        server.start()
        try:
            (c0,) = _register(server, "w0")
            server._stats.total_tokens = 10**9  # way over the old budget
            resp = c0.set_token_budget(0)
            assert resp["token_budget"] == 0
            assert resp["budget_stop_sent"] is False
            assert _pending(server, "w0") == [None]  # disabled -> no stop
        finally:
            server.stop()

    def test_raise_clears_prior_stop(self, tmp_path):
        server = _server(tmp_path, token_budget=1000)
        server.start()
        try:
            (c0,) = _register(server, "w0")
            server._stats.total_tokens = 1000
            server._maybe_trigger_token_budget()
            assert server._budget_stop_sent
            # Raise above the total: the guard resets, no fresh stop yet.
            server._workers["w0"].pending_command = None
            resp = c0.set_token_budget(10000)
            assert resp["budget_stop_sent"] is False
            assert _pending(server, "w0") == [None]
        finally:
            server.stop()


class TestTokenProgress:
    """The progress/ETA gauge the webui bar and CLI status line consume."""

    def test_none_without_budget(self, tmp_path):
        server = _server(tmp_path, token_budget=0)
        assert server._token_progress() is None
        # No samples are recorded when there's no budget.
        server._stats.total_tokens = 5
        server._record_token_sample()
        assert len(server._token_samples) == 0

    def test_progress_fields_and_fraction(self, tmp_path):
        server = _server(tmp_path, token_budget=1000)
        server._stats.total_tokens = 250
        prog = server._token_progress()
        assert prog["tokens_completed"] == 250
        assert prog["token_budget"] == 1000
        assert prog["fraction"] == 0.25
        assert prog["rate_window_seconds"] == server._token_rate_window_s
        # No rate from a single sample => no ETA.
        assert prog["tokens_per_second"] is None
        assert prog["eta_seconds"] is None

    def test_fraction_clamps_at_one(self, tmp_path):
        server = _server(tmp_path, token_budget=1000)
        server._stats.total_tokens = 1500  # overshoot past the budget
        assert server._token_progress()["fraction"] == 1.0

    def test_rolling_rate_and_eta(self, tmp_path):
        server = _server(tmp_path, token_budget=2000)
        # Seed (monotonic ts, total_tokens) samples directly: 100 tokens/10s.
        server._token_samples.extend((1000.0 + i * 10.0, i * 100) for i in range(6))
        server._stats.total_tokens = 500  # = last sample
        rate = server._rolling_token_rate()
        assert abs(rate - 10.0) < 1e-9  # 100 tokens / 10 s
        prog = server._token_progress()
        assert abs(prog["tokens_per_second"] - 10.0) < 1e-9
        # remaining 1500 tokens / 10 tok/s = 150 s
        assert abs(prog["eta_seconds"] - 150.0) < 1e-6

    def test_rate_window_uses_recent_baseline(self, tmp_path):
        server = _server(tmp_path, token_budget=10_000)
        server._token_rate_window_s = 50.0
        # Old slow segment then a recent fast one; only the last 50s counts.
        samples = [(0.0, 0), (10.0, 10), (100.0, 100), (110.0, 200), (120.0, 300)]
        server._token_samples.extend(samples)
        server._stats.total_tokens = 300
        # window [70,120]: baseline (100.0,100) -> (300-100)/(120-100)=10 tok/s
        assert abs(server._rolling_token_rate() - 10.0) < 1e-9

    def test_stall_reports_zero_rate_not_none(self, tmp_path):
        """After a long stall the only in-window sample is the tail; fall back to
        the previous sample so the gauge reads ~0 tok/s (visible slowdown), not a
        blank None."""
        server = _server(tmp_path, token_budget=10_000)
        server._token_rate_window_s = 50.0
        # All progress is old; the last two samples are flat (a stall) and only
        # the tail falls inside the 50s window.
        server._token_samples.extend(
            [(0.0, 1000), (10.0, 1000), (200.0, 1000), (260.0, 1000)]
        )
        server._stats.total_tokens = 1000
        rate = server._rolling_token_rate()
        assert rate is not None
        assert rate == 0.0  # baseline = samples[-2]=(200,1000), now=(260,1000)
        # A 0 rate yields no ETA (can't divide), but progress still renders.
        prog = server._token_progress()
        assert prog["tokens_per_second"] == 0.0
        assert prog["eta_seconds"] is None
