"""DiLoCo sync-log metrics: rendered columns, publish-gating, windowed means,
and the server-authoritative ``verbose_sync`` flag.

The step-table renderer shows only keys present in the log dict, so the worker's
``sync_metrics`` is the single control point for which DiLoCo columns appear: it
publishes up/down rates only for backends that move tensors over a wire, and the
rates are the mean over the syncs since the last log step (windowed), not a
single last-sync sample.
"""

import torch

from forgather.ml.diloco.sync_backend import OuterSyncBackend
from forgather.ml.diloco.worker import DiLoCoWorker
from forgather.ml.trainer.logging import (
    ColumnSpec,
    default_step_columns,
    format_train_header,
)


class _FakeBackend(OuterSyncBackend):
    """Minimal backend so a worker can be constructed without a server."""

    def __init__(self, reports_transfer_bytes=True):
        self.reports_transfer_bytes = reports_transfer_bytes

    def join(self, **k):
        return {}

    def synchronize(self, **k):
        pass

    def synchronize_fragment(self, **k):
        pass

    def current_global_params(self):
        return {}

    def leave(self, **k):
        pass


def _worker(reports_transfer_bytes=True):
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    return DiLoCoWorker(
        model,
        opt,
        server_addr="localhost:0",
        sync_every=20,
        heartbeat_interval=0,
        backend=_FakeBackend(reports_transfer_bytes),
    )


def test_columns_defined_for_diloco_metrics():
    cols = default_step_columns()
    for key in (
        "diloco/sync_count",
        "diloco/sync_send_mb",
        "diloco/sync_recv_mb",
        "diloco/sync_time",
    ):
        assert key in cols


def test_sync_metrics_windowed_mean():
    w = _worker()
    # 3 syncs in the window: 6 MB up, 12 MB down, 1.5s total.
    w._win_sync_count = 3
    w._win_send_bytes = 6_000_000
    w._win_recv_bytes = 12_000_000
    w._win_sync_time = 1.5
    m = w.sync_metrics
    assert m["diloco/sync_time"] == 0.5
    assert m["diloco/sync_send_mb"] == 2.0
    assert m["diloco/sync_recv_mb"] == 4.0


def test_sync_metrics_empty_window_omits_rates():
    w = _worker()  # no syncs in the window
    m = w.sync_metrics
    assert "diloco/sync_time" not in m
    assert "diloco/sync_send_mb" not in m
    assert "diloco/sync_count" in m  # count is always published


def test_shared_memory_omits_transfer_columns():
    w = _worker(reports_transfer_bytes=False)
    w._win_sync_count = 2
    w._win_sync_time = 1.0
    m = w.sync_metrics
    # No wire: up/down rates omitted, but sync time still published.
    assert "diloco/sync_send_mb" not in m
    assert "diloco/sync_recv_mb" not in m
    assert m["diloco/sync_time"] == 0.5


def test_note_logged_resets_window():
    w = _worker()
    w._win_sync_count = 5
    w._win_send_bytes = 10
    w._win_recv_bytes = 20
    w._win_sync_time = 2.0
    w.note_logged()
    assert w._win_sync_count == 0
    assert w._win_send_bytes == 0
    assert w._win_recv_bytes == 0
    assert w._win_sync_time == 0.0


def test_rendered_header_gates_on_published_keys():
    cols = [ColumnSpec(key=k, **v) for k, v in default_step_columns().items()]
    # shared-memory-style metrics: sync + sync_s, no up/dn.
    shm = {"loss": 3.2, "diloco/sync_count": 9, "diloco/sync_time": 0.5}
    hdr = format_train_header(cols, shm)
    assert "sync" in hdr and "sync_s" in hdr
    assert "up_mb" not in hdr and "dn_mb" not in hdr
    # http-style: also up/dn.
    http = dict(shm)
    http["diloco/sync_send_mb"] = 68.8
    http["diloco/sync_recv_mb"] = 137.7
    hdr2 = format_train_header(cols, http)
    assert "up_mb" in hdr2 and "dn_mb" in hdr2


def test_server_verbose_sync_default_off():
    # The server carries a verbose_sync flag (advertised via /info), default off.
    import inspect

    from forgather.ml.diloco.server import DiLoCoServer

    sig = inspect.signature(DiLoCoServer.__init__)
    assert sig.parameters["verbose_sync"].default is False
