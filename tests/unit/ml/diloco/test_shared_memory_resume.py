"""Resume coherence for the shared-memory server-aggregator (#197/#198).

The regression these guard: in shared-memory mode the trained global model +
outer-optimizer momentum used to live only in an aggregator *worker's* ephemeral
region and were never persisted, so a stop/resume reseeded the region from the
untrained server seed and training diverged (loss regressed then climbed). Under
Flavor 2 the server IS the aggregator — it owns the master + outer optimizer and
checkpoints them through save_state/load_state — so a resume must continue the
outer trajectory exactly, as if there had been no restart.

These run a real DiLoCoServer in shared_memory mode with a single co-located
follower (group_size=1, so the server's aggregation thread drives each round)
and compare a stop/resume run against a no-restart control.
"""

import torch

from forgather.ml.diloco.server import DiLoCoServer
from forgather.ml.diloco.shared_memory_backend import SharedMemoryBackend

from .conftest import make_initial_checkpoint


def _sd(dim=6):
    torch.manual_seed(7)
    return {"a.weight": torch.randn(dim, dim), "b.weight": torch.randn(dim)}


def _server(model_dir, ckpt, group_dir, **kw):
    s = DiLoCoServer(
        output_dir=str(model_dir),
        from_checkpoint=str(ckpt),
        num_workers=1,
        port=0,
        backend="shared_memory",
        save_every_n_rounds=0,  # save manually for determinism
        **kw,
    )
    # Keep the region inside the test's tmp dir (default is /tmp/diloco_shm_p<port>).
    s._shm_group_dir = str(group_dir)
    return s


def _run_rounds(group_dir, pg, rounds, wid="w0"):
    """Drive ``rounds`` sync rounds as a single follower; return the last
    published master. The server's aggregation thread does the outer step."""
    be = SharedMemoryBackend(group_dir=group_dir, group_size=1, follower_only=True)
    be.join(worker_id=wid)
    last = None
    try:
        for _ in range(rounds):
            res = be.synchronize(worker_id=wid, pseudograds=pg)
            last = {k: v.clone() for k, v in res.params.items()}
    finally:
        be.leave(worker_id=wid)
    return last


def test_shm_resume_is_transparent(tmp_path):
    sd = _sd()
    pg = {"a.weight": torch.ones(6, 6) * 0.05, "b.weight": torch.ones(6) * 0.05}
    K = 4

    # Control: K+1 rounds straight through, no restart.
    mdir_c = tmp_path / "ctrl_model"
    ckpt_c = make_initial_checkpoint(sd, mdir_c)
    sc = _server(mdir_c, ckpt_c, tmp_path / "shm_ctrl")
    sc.start()
    try:
        ctrl_master = _run_rounds(sc._shm_group_dir, pg, K + 1)
        assert sc._sync_round == K + 1
    finally:
        sc.stop()

    # Test: K rounds, checkpoint, stop, resume, then 1 more round.
    mdir_t = tmp_path / "test_model"
    ckpt_t = make_initial_checkpoint(sd, mdir_t)
    st = _server(mdir_t, ckpt_t, tmp_path / "shm_test")
    st.start()
    try:
        _run_rounds(st._shm_group_dir, pg, K)
        assert st._sync_round == K
        cpt = tmp_path / "ckpt_at_K"
        st.save_state(path=str(cpt))
    finally:
        st.stop()

    # Resume from the K-round checkpoint: round + master + momentum restored.
    st2 = _server(mdir_t, cpt, tmp_path / "shm_test2")
    assert st2._sync_round == K
    st2.start()
    try:
        resumed_master = _run_rounds(st2._shm_group_dir, pg, 1)
        assert st2._sync_round == K + 1
    finally:
        st2.stop()

    # The resumed trajectory must match the no-restart control exactly: if the
    # master or the outer-optimizer momentum were restored incoherently, step
    # K+1 would differ (the old bug diverged here).
    for k in ctrl_master:
        delta = (ctrl_master[k] - resumed_master[k]).abs().max().item()
        assert torch.allclose(
            ctrl_master[k], resumed_master[k], atol=1e-5
        ), f"resume diverged on {k}: max|delta|={delta}"


def test_shm_checkpoint_round_advances(tmp_path):
    """#197: the server-side checkpoint is named from an advancing sync_round
    (not the frozen 0 of the old worker-aggregator path)."""
    sd = _sd()
    pg = {"a.weight": torch.ones(6, 6) * 0.05, "b.weight": torch.ones(6) * 0.05}
    mdir = tmp_path / "model"
    ckpt = make_initial_checkpoint(sd, mdir)
    s = _server(mdir, ckpt, tmp_path / "shm", save_total_limit=5)
    s.start()
    try:
        _run_rounds(s._shm_group_dir, pg, 3)
        assert s._sync_round == 3
        s.save_state()  # auto-named checkpoint-{sync_round}
    finally:
        s.stop()
    # The checkpoint must be checkpoint-3 (the trained round), not checkpoint-0.
    ckpt_dir = mdir / "checkpoints"
    names = {p.name for p in ckpt_dir.iterdir()} if ckpt_dir.exists() else set()
    assert "checkpoint-3" in names, f"expected checkpoint-3, got {sorted(names)}"
    assert "checkpoint-0" not in names
