"""Tests for tools/forgather_server/queue_store.py."""

import forgather_server.queue_store as queue_store
import pytest


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Redirect server state to a temp dir so tests are hermetic."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # queue_store imports queue_file from paths; patch the local binding.
    monkeypatch.setattr(queue_store, "queue_file", lambda: state_dir / "queue.json")
    yield state_dir


def _make_item(**kwargs):
    defaults = dict(
        project_dir="/some/project",
        config="train.yaml",
        dynamic_args={},
        requested_gpus=1,
        priority=0,
    )
    defaults.update(kwargs)
    return queue_store.QueueItem.new(**defaults)


class TestQueueItemNew:
    def test_id_generated(self):
        item = _make_item()
        assert item.queue_id.startswith("q_")

    def test_ids_are_unique(self):
        a = _make_item()
        b = _make_item()
        assert a.queue_id != b.queue_id

    def test_requested_gpus_clamped_to_zero(self):
        item = _make_item(requested_gpus=-5)
        assert item.requested_gpus == 0

    def test_fields_stored(self):
        item = _make_item(project_dir="/my/proj", config="cfg.yaml", priority=3)
        assert item.project_dir == "/my/proj"
        assert item.config == "cfg.yaml"
        assert item.priority == 3

    def test_job_type_default(self):
        item = _make_item()
        assert item.job_type == "training"

    def test_submitted_at_is_set(self):
        item = _make_item()
        assert item.submitted_at > 0


class TestQueuePersistence:
    def test_add_and_list(self):
        item = _make_item()
        queue_store.add_item(item)
        items = queue_store.list_items()
        assert len(items) == 1
        assert items[0].queue_id == item.queue_id

    def test_list_empty_initially(self):
        assert queue_store.list_items() == []

    def test_multiple_items(self):
        a = _make_item(priority=1)
        b = _make_item(priority=2)
        queue_store.add_item(a)
        queue_store.add_item(b)
        ids = {it.queue_id for it in queue_store.list_items()}
        assert ids == {a.queue_id, b.queue_id}

    def test_remove_existing_item(self):
        item = _make_item()
        queue_store.add_item(item)
        removed = queue_store.remove_item(item.queue_id)
        assert removed is True
        assert queue_store.list_items() == []

    def test_remove_nonexistent_returns_false(self):
        assert queue_store.remove_item("nonexistent_id") is False

    def test_persistence_across_calls(self, tmp_path):
        """Write queue then verify reading it back gives the same data."""
        item = _make_item(project_dir="/persistent/project", config="persisted.yaml")
        queue_store.add_item(item)
        # Re-read without any in-memory cache involved.
        reread = queue_store.list_items()
        assert len(reread) == 1
        assert reread[0].project_dir == "/persistent/project"
        assert reread[0].config == "persisted.yaml"

    def test_roundtrip_preserves_dynamic_args(self):
        args = {"lr": 1e-4, "epochs": 10, "label": "run-A"}
        item = _make_item(dynamic_args=args)
        queue_store.add_item(item)
        reread = queue_store.list_items()[0]
        assert reread.dynamic_args == args

    def test_get_item(self):
        item = _make_item()
        queue_store.add_item(item)
        got = queue_store.get_item(item.queue_id)
        assert got is not None
        assert got.queue_id == item.queue_id

    def test_get_item_missing_returns_none(self):
        assert queue_store.get_item("no_such_id") is None
