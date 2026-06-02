"""Tests for promoting externally-launched trainer endpoints to JobRecords,
and the dead-endpoint-dir GC sweep (tools/forgather_server)."""

import json
import os
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
    with scheduler._state._lock:
        scheduler._state.running.clear()
    scheduler._promote_grace = set()
    yield state_dir
    with scheduler._state._lock:
        scheduler._state.running.clear()
    scheduler._promote_grace = set()


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


def _promote_twice():
    # The one-tick grace means an external endpoint is promoted on the second
    # observation, not the first.
    scheduler._promote_external_endpoints()
    scheduler._promote_external_endpoints()


class TestPromotion:
    def test_promotes_external_endpoint(self, monkeypatch):
        _patch_endpoints(monkeypatch, [_ep("trainer-1", 4242)])
        _promote_twice()

        rec = job_records.get_record("ext_trainer-1")
        assert rec is not None
        assert rec.externally_launched is True
        assert rec.status == "running"
        assert rec.pid == 4242
        assert rec.job_id == "trainer-1"
        assert rec.gpu_indices == []
        assert rec.requested_gpus == 0
        assert "ext_trainer-1" in scheduler._state.running

    def test_grace_delays_promotion_one_tick(self, monkeypatch):
        _patch_endpoints(monkeypatch, [_ep("trainer-1", 4242)])
        scheduler._promote_external_endpoints()  # first sighting: grace
        assert job_records.get_record("ext_trainer-1") is None
        scheduler._promote_external_endpoints()  # second: promote
        assert job_records.get_record("ext_trainer-1") is not None

    def test_skips_scheduler_launched(self, monkeypatch):
        # Endpoint pid descends from a running record's launcher pid → never
        # promoted, regardless of the grace.
        job_records.add_record(
            job_records.JobRecord(queue_id="q1", status="running", pid=100)
        )
        _patch_endpoints(
            monkeypatch,
            [_ep("trainer-x", 4242)],
            ancestors={4242: [4242, 100]},  # 100 is the launcher
        )
        _promote_twice()
        assert job_records.get_record("ext_trainer-x") is None

    def test_skips_dead_endpoint(self, monkeypatch):
        _patch_endpoints(monkeypatch, [_ep("trainer-dead", 5555)], alive=False)
        _promote_twice()
        assert job_records.get_record("ext_trainer-dead") is None

    def test_no_double_promote(self, monkeypatch):
        _patch_endpoints(monkeypatch, [_ep("trainer-1", 4242)])
        _promote_twice()
        scheduler._promote_external_endpoints()  # extra ticks
        scheduler._promote_external_endpoints()
        ext = [r for r in job_records.list_records() if r.job_id == "trainer-1"]
        assert len(ext) == 1

    def test_skips_already_correlated_record(self, monkeypatch):
        job_records.add_record(
            job_records.JobRecord(
                queue_id="q9", status="running", pid=100, job_id="trainer-1"
            )
        )
        _patch_endpoints(monkeypatch, [_ep("trainer-1", 4242)])
        _promote_twice()
        assert job_records.get_record("ext_trainer-1") is None


class TestEndpointDirSweep:
    @pytest.fixture
    def jobs_dir(self, tmp_path, monkeypatch):
        import forgather.preprocess as pp

        cfg = tmp_path / "cfg"
        (cfg / "jobs").mkdir(parents=True)
        monkeypatch.setattr(pp, "forgather_config_dir", lambda: str(cfg))
        return cfg / "jobs"

    def _dir(self, jobs_dir, name, *, age_s=7200, endpoint="{}"):
        d = jobs_dir / name
        d.mkdir()
        if endpoint is not None:
            (d / "endpoint.json").write_text(endpoint)
        old = time.time() - age_s
        os.utime(d, (old, old))
        return d

    def test_removes_dead_old_dir(self, jobs_dir, monkeypatch):
        d = self._dir(jobs_dir, "dead-job", endpoint=json.dumps({"pid": 777}))
        monkeypatch.setattr(gc, "_pid_definitely_gone", lambda pid: True)
        assert gc.sweep_dead_endpoint_dirs() == 1
        assert not d.exists()

    def test_protects_live_dir(self, jobs_dir, monkeypatch):
        d = self._dir(jobs_dir, "live-job", endpoint=json.dumps({"pid": 777}))
        monkeypatch.setattr(gc, "_pid_definitely_gone", lambda pid: False)
        assert gc.sweep_dead_endpoint_dirs() == 0
        assert d.exists()

    def test_protects_indeterminate_pid(self, jobs_dir, monkeypatch):
        # _pid_definitely_gone returns False when it can't prove death.
        d = self._dir(jobs_dir, "maybe-job", endpoint=json.dumps({"pid": 777}))
        monkeypatch.setattr(gc, "_pid_definitely_gone", lambda pid: False)
        assert gc.sweep_dead_endpoint_dirs() == 0
        assert d.exists()

    def test_protects_unparseable_endpoint(self, jobs_dir, monkeypatch):
        d = self._dir(jobs_dir, "weird-job", endpoint="not json {")
        monkeypatch.setattr(gc, "_pid_definitely_gone", lambda pid: True)
        assert gc.sweep_dead_endpoint_dirs() == 0
        assert d.exists()

    def test_protects_fresh_dir(self, jobs_dir, monkeypatch):
        d = self._dir(jobs_dir, "fresh-job", age_s=10, endpoint=json.dumps({"pid": 1}))
        monkeypatch.setattr(gc, "_pid_definitely_gone", lambda pid: True)
        assert gc.sweep_dead_endpoint_dirs() == 0
        assert d.exists()

    def test_removes_orphan_without_endpoint(self, jobs_dir):
        d = self._dir(jobs_dir, "orphan-job", endpoint=None)  # no endpoint.json
        assert gc.sweep_dead_endpoint_dirs() == 1
        assert not d.exists()


class TestExternalKillSafety:
    """A promoted external record must NOT be killed via process-group signal —
    its pid is in the operator's shell session, not a server-spawned group."""

    def _external_rec(self):
        return job_records.JobRecord(
            queue_id="ext_x",
            job_id="x",
            externally_launched=True,
            job_type="training",
            status="running",
            pid=4242,
        )

    @pytest.mark.parametrize("action", ["kill", "force-kill"])
    def test_kill_refused_for_external_record(self, monkeypatch, action):
        from fastapi import HTTPException
        from forgather_server.routes import jobs as jobs_routes

        monkeypatch.setattr(
            jobs_routes.job_records, "get_record", lambda jid: self._external_rec()
        )

        def _boom(*a, **k):
            raise AssertionError("must not signal an external job's process group")

        monkeypatch.setattr(jobs_routes.scheduler, "force_kill_record", _boom)
        monkeypatch.setattr(jobs_routes.scheduler, "abort_record", _boom)

        with pytest.raises(HTTPException) as exc:
            jobs_routes.job_control("ext_x", action)
        assert exc.value.status_code == 400
        assert "externally-launched" in exc.value.detail
