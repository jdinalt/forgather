"""Tests for tools/forgather_server/cluster_jobs.py."""

import forgather_server.cluster_jobs as cluster_jobs
import forgather_server.cluster_journal as cluster_journal
import pytest
from forgather_server import paths


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    cluster_dir = tmp_path / "cluster"
    cluster_dir.mkdir()
    journal_dir = cluster_dir / "journal"
    journal_dir.mkdir()
    monkeypatch.setattr(paths, "cluster_state_dir", lambda: cluster_dir)
    monkeypatch.setattr(paths, "cluster_journal_dir", lambda: journal_dir)
    cluster_jobs._reset_for_tests()
    cluster_journal._reset_for_tests()
    yield
    cluster_jobs._reset_for_tests()
    cluster_journal._reset_for_tests()


def _make_job(cjid="cj_abc"):
    return cluster_jobs.ClusterJob(
        cluster_job_id=cjid,
        project_dir="/proj",
        config="train.yaml",
        submitted_at=100.0,
        rdzv_endpoint="wopr:29400",
        rdzv_id="rdzv-xyz",
        rdzv_node_id="node-0",
        members=[
            cluster_jobs.MemberAssignment(
                node_id="node-0",
                hostname="wopr",
                address="192.168.1.27",
                port=8765,
                queue_id="q_w_0",
                nproc_per_node=2,
                node_rank=0,
            ),
            cluster_jobs.MemberAssignment(
                node_id="node-1",
                hostname="muthur",
                address="192.168.1.162",
                port=8765,
                queue_id="q_m_0",
                nproc_per_node=1,
                node_rank=1,
            ),
        ],
    )


class TestAddListGet:
    def test_add_and_get(self):
        job = cluster_jobs.add_job(_make_job())
        assert cluster_jobs.get_job(job.cluster_job_id) is job

    def test_collision_rejected(self):
        cluster_jobs.add_job(_make_job("cj_a"))
        with pytest.raises(ValueError):
            cluster_jobs.add_job(_make_job("cj_a"))

    def test_list_newest_first(self):
        a = cluster_jobs.add_job(_make_job("cj_a"))
        b_job = _make_job("cj_b")
        b_job.submitted_at = 200.0
        b = cluster_jobs.add_job(b_job)
        ordered = cluster_jobs.list_jobs()
        assert [j.cluster_job_id for j in ordered] == [b.cluster_job_id, a.cluster_job_id]


class TestCancel:
    def test_mark_cancelled_sets_status(self):
        cluster_jobs.add_job(_make_job("cj_a"))
        job = cluster_jobs.mark_cancelled("cj_a")
        assert job is not None
        assert job.status == "cancelled"
        assert job.cancelled_at is not None

    def test_mark_cancelled_idempotent(self):
        cluster_jobs.add_job(_make_job("cj_a"))
        job1 = cluster_jobs.mark_cancelled("cj_a")
        ts1 = job1.cancelled_at
        job2 = cluster_jobs.mark_cancelled("cj_a")
        # Second call returns the same record without bumping the
        # cancelled_at timestamp.
        assert job2.cancelled_at == ts1

    def test_cancel_unknown_returns_none(self):
        assert cluster_jobs.mark_cancelled("cj_nope") is None


class TestJournalIntegration:
    def test_submit_emits_journal_event(self):
        cluster_jobs.add_job(_make_job("cj_a"))
        events = list(cluster_journal.replay())
        types = [e.type for e in events]
        assert "multinode_job_submitted" in types
        ev = next(e for e in events if e.type == "multinode_job_submitted")
        assert ev.payload["cluster_job_id"] == "cj_a"
        assert len(ev.payload["members"]) == 2

    def test_cancel_emits_journal_event(self):
        cluster_jobs.add_job(_make_job("cj_a"))
        cluster_jobs.mark_cancelled("cj_a")
        events = list(cluster_journal.replay())
        types = [e.type for e in events]
        assert types.count("multinode_job_submitted") == 1
        assert types.count("multinode_job_cancelled") == 1


class TestSerialization:
    def test_round_trip(self):
        job = _make_job("cj_a")
        d = job.to_dict()
        restored = cluster_jobs.ClusterJob.from_dict(d)
        assert restored == job
