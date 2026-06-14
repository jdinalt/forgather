"""Issue #218: a worker whose backend can't be derived yet (param server still
starting / momentarily busy) is *requeued* for a later dispatch tick, not failed
on the single timed-out probe. After the retry budget is spent it fails for real.
"""

from __future__ import annotations

import time

import forgather_server.scheduler as sched
import pytest
from forgather_server.queue_store import QueueItem


def _item(qid="q-retry-1"):
    return QueueItem(
        queue_id=qid,
        project_dir="/proj",
        config="diloco.yaml",
        requested_gpus=1,
        job_type="training",
        job_params={"diloco": {"server_addr": "https://host:8512"}},
    )


@pytest.fixture
def spy_storage(monkeypatch):
    """Mock the durable queue/record calls so _launch runs without files."""
    calls = {
        "add_record": [],
        "remove_record": [],
        "remove_item": [],
        "add_item": [],
        "updates": [],
    }
    monkeypatch.setattr(
        sched.job_records, "add_record", lambda r: calls["add_record"].append(r)
    )
    monkeypatch.setattr(
        sched.job_records,
        "remove_record",
        lambda q: calls["remove_record"].append(q) or True,
    )
    monkeypatch.setattr(
        sched.queue_store,
        "remove_item",
        lambda q: calls["remove_item"].append(q) or True,
    )
    monkeypatch.setattr(
        sched.queue_store, "add_item", lambda it: calls["add_item"].append(it) or it
    )
    monkeypatch.setattr(
        sched.job_records,
        "update_record",
        lambda q, **kw: calls["updates"].append((q, kw)) or None,
    )
    # Keep the bookkeeping dict clean between tests.
    sched._diloco_derive_first_seen.clear()
    yield calls
    sched._diloco_derive_first_seen.clear()


def _raise_unreachable(*a, **k):
    raise sched._DilocoServerUnreachable(
        "server at https://host:8512 is unreachable (timed out)"
    )


def test_query_info_raises_typed_unreachable(monkeypatch):
    """A network failure on the /info probe raises the requeue-able type."""
    import urllib.request

    def boom(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(sched, "_diloco_server_token", lambda s: None)
    with pytest.raises(sched._DilocoServerUnreachable):
        sched._diloco_query_info("http://host:8512", "q-x")


def test_unreachable_requeues_within_budget(spy_storage, monkeypatch):
    monkeypatch.setattr(sched, "_build_training", _raise_unreachable)
    sched._launch(_item(), [0])
    # Requeued: record dropped, item put back, NOT marked failed.
    assert spy_storage["remove_record"] == ["q-retry-1"]
    assert len(spy_storage["add_item"]) == 1
    assert spy_storage["add_item"][0].queue_id == "q-retry-1"
    assert spy_storage["updates"] == []  # no failed transition
    assert "q-retry-1" in sched._diloco_derive_first_seen


def test_unreachable_fails_after_budget(spy_storage, monkeypatch):
    monkeypatch.setattr(sched, "_build_training", _raise_unreachable)
    # Pretend the first failed attempt was longer ago than the budget.
    sched._diloco_derive_first_seen["q-retry-1"] = (
        time.monotonic() - sched._DILOCO_DERIVE_RETRY_BUDGET_S - 1
    )
    sched._launch(_item(), [0])
    # Budget spent: failed for real, not requeued.
    assert spy_storage["add_item"] == []
    assert spy_storage["updates"], "expected a failed update"
    q, kw = spy_storage["updates"][-1]
    assert q == "q-retry-1" and kw.get("status") == "failed"
    assert "q-retry-1" not in sched._diloco_derive_first_seen


def test_cancel_queued_clears_retry_bookkeeping(monkeypatch):
    """Cancelling a worker mid-retry must not leak its dict entry."""
    monkeypatch.setattr(sched.queue_store, "remove_item", lambda q: True)
    sched._diloco_derive_first_seen.clear()
    sched._diloco_derive_first_seen["q-retry-1"] = time.monotonic()
    assert sched.cancel_queued("q-retry-1") is True
    assert "q-retry-1" not in sched._diloco_derive_first_seen


def test_success_clears_retry_bookkeeping(spy_storage, monkeypatch):
    class _Res:
        proc = object()
        pid = 4321

    monkeypatch.setattr(sched, "_build_training", lambda *a, **k: _Res())
    monkeypatch.setattr(sched._state, "running", {}, raising=False)
    sched._diloco_derive_first_seen["q-retry-1"] = time.monotonic()
    sched._launch(_item(), [0])
    # Launched cleanly -> bookkeeping cleared, status -> running.
    assert "q-retry-1" not in sched._diloco_derive_first_seen
    assert any(kw.get("status") == "running" for _, kw in spy_storage["updates"])
