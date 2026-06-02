"""Tests for promoting externally-launched trainer endpoints to JobRecords,
and the dead-endpoint-dir GC sweep (tools/forgather_server)."""

import time
import types

import forgather_server._gc as gc
import forgather_server.job_records as job_records
import forgather_server.scheduler as scheduler
import pytest


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(job_records, "server_state_dir", lambda: state_dir)
    # Reset the scheduler's in-memory running map between tests.
    with scheduler._state._lock:
        scheduler._state.running.clear()
    yield state_dir
    with scheduler._state._lock:
        scheduler._state.running.clear()


def _ep(job_id, pid, *, started_at=1000.0, logging_dir="/logs", output_dir="/out"):
    return types.SimpleNamespace(
        job_id=job_id,
        pid=pid,
        started_at=started_at,
        logging_dir=logging_dir,
        output_dir=output_dir,
    )


def _patch_endpoints(monkeypatch, endpoints, *, alive=True, ancestors=None):
    monkeypatch.setattr(scheduler.trainer_control, "list_jobs", lambda: endpoints)
    monkeypatch.setattr(
        scheduler.trainer_control,
        "is_endpoint_pid_alive",
        lambda pid, started_at: alive,
    )
    monkeypatch.setattr(
        scheduler, "_pid_ancestors", lambda pid: (ancestors or {}).get(pid, [pid])
    )


class TestPromotion:
    def test_promotes_external_endpoint(self, monkeypatch):
        _patch_endpoints(monkeypatch, [_ep("trainer-1", 4242)])
        scheduler._promote_external_endpoints()

        rec = job_records.get_record("ext_trainer-1")
        assert rec is not None
        assert rec.externally_launched is True
        assert rec.status == "running"
        assert rec.pid == 4242
        assert rec.job_id == "trainer-1"
        assert rec.gpu_indices == []
        assert rec.requested_gpus == 0
        # registered for reaping
        assert "ext_trainer-1" in scheduler._state.running

    def test_skips_scheduler_launched(self, monkeypatch):
        # A scheduler-launched trainer: its record carries the torchrun pid as
        # launcher; the endpoint pid descends from it. Must NOT be promoted.
        job_records.add_record(
            job_records.JobRecord(queue_id="q1", status="running", pid=100)
        )
        _patch_endpoints(
            monkeypatch,
            [_ep("trainer-x", 4242)],
            ancestors={4242: [4242, 100]},  # 100 is the launcher
        )
        scheduler._promote_external_endpoints()
        assert job_records.get_record("ext_trainer-x") is None

    def test_skips_dead_endpoint(self, monkeypatch):
        _patch_endpoints(monkeypatch, [_ep("trainer-dead", 5555)], alive=False)
        scheduler._promote_external_endpoints()
        assert job_records.get_record("ext_trainer-dead") is None

    def test_no_double_promote(self, monkeypatch):
        _patch_endpoints(monkeypatch, [_ep("trainer-1", 4242)])
        scheduler._promote_external_endpoints()
        scheduler._promote_external_endpoints()  # second tick
        ext = [r for r in job_records.list_records() if r.job_id == "trainer-1"]
        assert len(ext) == 1

    def test_skips_already_correlated_record(self, monkeypatch):
        # A record already correlated to this endpoint's job_id exists.
        job_records.add_record(
            job_records.JobRecord(
                queue_id="q9", status="running", pid=100, job_id="trainer-1"
            )
        )
        _patch_endpoints(monkeypatch, [_ep("trainer-1", 4242)])
        scheduler._promote_external_endpoints()
        assert job_records.get_record("ext_trainer-1") is None


class TestEndpointDirSweep:
    @pytest.fixture
    def jobs_dir(self, tmp_path, monkeypatch):
        import forgather.preprocess as pp

        cfg = tmp_path / "cfg"
        (cfg / "jobs").mkdir(parents=True)
        monkeypatch.setattr(pp, "forgather_config_dir", lambda: str(cfg))
        return cfg / "jobs"

    def _patch_alive(self, monkeypatch, alive_ids):
        import forgather.trainer_control as tc

        eps = [_ep(jid, 1) for jid in alive_ids]
        monkeypatch.setattr(tc, "list_jobs", lambda: eps)
        monkeypatch.setattr(tc, "is_endpoint_pid_alive", lambda pid, started_at: True)

    def _old_dir(self, jobs_dir, name, age_s=7200):
        d = jobs_dir / name
        d.mkdir()
        (d / "endpoint.json").write_text("{}")
        old = time.time() - age_s
        import os

        os.utime(d, (old, old))
        return d

    def test_removes_dead_old_dir(self, jobs_dir, monkeypatch):
        d = self._old_dir(jobs_dir, "dead-job")
        self._patch_alive(monkeypatch, alive_ids=set())  # nothing alive
        removed = gc.sweep_dead_endpoint_dirs()
        assert removed == 1
        assert not d.exists()

    def test_protects_live_dir(self, jobs_dir, monkeypatch):
        d = self._old_dir(jobs_dir, "live-job")
        self._patch_alive(monkeypatch, alive_ids={"live-job"})
        removed = gc.sweep_dead_endpoint_dirs()
        assert removed == 0
        assert d.exists()

    def test_protects_fresh_dir(self, jobs_dir, monkeypatch):
        d = self._old_dir(jobs_dir, "fresh-job", age_s=10)  # within TTL
        self._patch_alive(monkeypatch, alive_ids=set())
        removed = gc.sweep_dead_endpoint_dirs()
        assert removed == 0
        assert d.exists()
