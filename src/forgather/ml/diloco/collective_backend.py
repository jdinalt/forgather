"""
Collective DiLoCo backend (issue #154).

The first *collective* :class:`~forgather.ml.diloco.sync_backend.OuterSyncBackend`.
Instead of a central HTTP parameter server or a shared CPU region, every worker
is an **independent DiLoCo replica** — its own data shard, no per-step DDP
gradient all-reduce — that, once per ``sync_every`` steps, ``all_reduce``s its
pseudo-gradient with its peers and runs an **identical replicated outer
optimizer** locally. Because every rank reduces the same pseudo-grads to the same
mean and steps an identical optimizer over identical weights, all ranks land on
bit-identical new global params with nothing crossing a central server. This is
the single-host DDP-alternative regime from #154: ``all_reduce`` over NVLink
(NCCL) every ``H`` steps in place of DDP's per-step all-reduce.

Outer step (every rank, symmetric — there is no leader/follower):

1. ``all_reduce(pseudograd, SUM)`` each parameter in a fixed name order, then
   divide by ``group_size`` -> the mean pseudo-gradient, identical on every rank.
2. Set ``.grad`` on a private master :class:`torch.nn.ParameterList`, step the
   outer optimizer, ``zero_grad`` — reproducing
   :meth:`DiLoCoServer._apply_outer_optimizer` (per-name mean over contributors,
   fp32, SGD-Nesterov) exactly, like the shared-memory backend.
3. The stepped master *is* the new global. No result broadcast: every rank
   computed the same thing (see the determinism note below), so ``synchronize``
   returns those params directly.

Determinism. Two requirements: (a) identical optimizer-state evolution — every
rank starts from the *same* init (rank 0 loads the checkpoint and broadcasts it)
and applies the same sequence of identical mean-grads, so SGD-Nesterov momentum
stays in lockstep; (b) the ``all_reduce(SUM)`` output is bitwise-identical across
ranks — relied upon (it is what DDP gradient averaging already depends on). The
correctness test asserts exact cross-rank equality, so any divergence fails the
build. ``broadcast_result=True`` is a one-line safety valve (broadcast rank 0's
stepped params) if a real cluster ever shows per-rank NCCL-algorithm divergence.

Scope (increment 1): the backend + protocol + correctness, plus the worker-loop
symmetric-participation integration. Fault tolerance (a dead peer hangs the
``all_reduce`` — ``fault_tolerant=False``), the cross-host wire cast, and
streaming fragments are follow-ups.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch

from .sync_backend import OuterSyncBackend, SyncResult

if TYPE_CHECKING:
    StateDict = Dict[str, torch.Tensor]


def _default_outer_optimizer_factory(params):
    """SGD with Nesterov momentum — the DiLoCo paper defaults, matching the
    server's ``_default_outer_optimizer_factory`` and the shared-memory
    backend's. The replicated outer step must use the *same* optimizer config on
    every rank for the results to stay bit-identical."""
    return torch.optim.SGD(params, lr=0.7, momentum=0.9, nesterov=True)


class CollectiveBackend(OuterSyncBackend):
    """Collective, replicated-outer-optimizer backend.

    Constructed with a ``torch.distributed`` process group (``None`` = the
    default group), an ``init_checkpoint`` directory rank 0 seeds the initial
    weights from, and the ``outer_opt_factory`` every rank builds its identical
    outer optimizer with. ``group_size`` / ``rank`` default to the group's
    world-size / rank.

    The backend *borrows* its process group from the launcher (torchrun); it
    never creates or destroys one. The all-reduce runs on the group's native
    device (CUDA for an NCCL group, CPU for gloo); the master weights + outer
    optimizer stay on CPU in fp32 (cheap, and bit-matches the server reference).
    """

    runs_outer_optimizer = "replicated"
    supports_async = False
    fault_tolerant = False  # a dead peer hangs the all_reduce (quorum is a follow-up)
    # join is a tensor-path op (collective broadcast), not an HTTP register —
    # the worker registers separately for coordinator membership / diagnostics.
    registers_with_coordinator = False

    def __init__(
        self,
        *,
        init_checkpoint: str,
        process_group=None,
        group_size: Optional[int] = None,
        rank: Optional[int] = None,
        outer_opt_factory: Optional[Callable] = None,
        broadcast_result: bool = False,
        init_broadcast: bool = True,
    ):
        import torch.distributed as dist

        self.init_checkpoint = init_checkpoint
        self.process_group = process_group
        self.outer_opt_factory = outer_opt_factory or _default_outer_optimizer_factory
        self.broadcast_result = bool(broadcast_result)
        self.init_broadcast = bool(init_broadcast)

        have_group = dist.is_available() and dist.is_initialized()
        if group_size is not None:
            self.group_size = int(group_size)
        elif have_group:
            self.group_size = dist.get_world_size(process_group)
        else:
            self.group_size = 1
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if rank is not None:
            self.rank = int(rank)
        elif have_group:
            self.rank = dist.get_rank(process_group)
        else:
            self.rank = 0

        # The collective runs on the group's native device: CUDA for an NCCL
        # group (the NVLink fast path), CPU for gloo. The master + outer
        # optimizer stay on CPU regardless. group_size==1 needs no collective.
        self._collective_device = torch.device("cpu")
        if have_group and self.group_size > 1:
            backend_name = str(dist.get_backend(process_group)).lower()
            if backend_name == "nccl":
                # torchrun always sets LOCAL_RANK (the physical GPU index). The
                # fallback is for the no-torchrun case and assumes inner=1
                # (group-local rank == local rank); at inner>1 the group-local
                # rank is not the device index, so LOCAL_RANK must be set.
                local_rank = int(
                    os.environ.get("LOCAL_RANK", self.rank % torch.cuda.device_count())
                )
                self._collective_device = torch.device("cuda", local_rank)

        # Set at join().
        self._names: List[str] = []
        self._global: Dict[str, torch.Tensor] = {}
        self._master_params: Optional[torch.nn.ParameterList] = None
        self._outer_optimizer: Optional[torch.optim.Optimizer] = None

    # ----- collective helpers ----------------------------------------------

    def _broadcast_object(self, obj, group_src: int):
        import torch.distributed as dist

        holder = [obj]
        kwargs = {"group": self.process_group}
        if self._collective_device.type == "cuda":
            kwargs["device"] = self._collective_device
        # group_src (not src): self.rank is the group-local diloco rank, so the
        # source must be group-local too — src= would be a *global* rank, wrong
        # for any diloco sub-group not rooted at global rank 0 (inner > 1).
        dist.broadcast_object_list(holder, group_src=group_src, **kwargs)
        return holder[0]

    def _broadcast_tensor(self, tensor: Optional[torch.Tensor], group_src: int):
        """Broadcast one fp32 tensor from group-local rank ``group_src`` to the
        group; non-root ranks allocate from the broadcast shape. Returns a CPU
        fp32 tensor."""
        import torch.distributed as dist

        meta = None
        if self.rank == group_src:
            meta = tuple(tensor.shape)
        shape = self._broadcast_object(meta, group_src=group_src)
        if self.rank == group_src:
            buf = (
                tensor.detach().to(self._collective_device, torch.float32).contiguous()
            )
        else:
            buf = torch.empty(
                shape, dtype=torch.float32, device=self._collective_device
            )
        dist.broadcast(buf, group=self.process_group, group_src=group_src)
        return buf.cpu()

    # ----- OuterSyncBackend -------------------------------------------------

    def join(
        self,
        *,
        worker_id: str,
        worker_info: Optional[dict] = None,
        outer_opt_factory: Optional[Callable] = None,
    ) -> "StateDict":
        """Agree on the initial global params and build the replicated optimizer.

        Rank 0 loads the init checkpoint and (``init_broadcast``) broadcasts the
        names + each tensor to the group, so every rank starts bit-identical —
        the precondition for the replicated outer step staying in lockstep. Each
        rank then builds its own master ``ParameterList`` + outer optimizer.
        """
        from forgather.ml.sharded_checkpoint import load_checkpoint

        factory = outer_opt_factory or self.outer_opt_factory

        # A worker contributes only the param names it owns. For an inner
        # pipeline each rank owns a *slice* (PipelineParamView); the master and
        # the per-round all-reduce must be over exactly that slice, or
        # ``synchronize`` would fail loud on the names this rank doesn't hold.
        # ``worker_info["param_shapes"]`` carries the slice's names; for a
        # non-pipeline worker that is the full model, so the filter is a no-op.
        slice_names = None
        if worker_info and isinstance(worker_info.get("param_shapes"), dict):
            slice_names = set(worker_info["param_shapes"].keys())

        master: Dict[str, torch.Tensor] = {}
        if self.rank == 0:
            init = load_checkpoint(self.init_checkpoint, module=None, device="cpu")
            # Preserve the checkpoint's name order (the canonical order the
            # replicated outer step relies on), filtered to this slice.
            master = {
                k: v.detach().float().cpu().contiguous()
                for k, v in init.items()
                if slice_names is None or k in slice_names
            }
            names = list(master.keys())
        else:
            names = None

        if self.init_broadcast and self.group_size > 1:
            names = self._broadcast_object(names, group_src=0)
            for name in names:
                src_t = master.get(name) if self.rank == 0 else None
                master[name] = self._broadcast_tensor(src_t, group_src=0)
        elif names is None:
            # group_size==1 or init_broadcast disabled: every rank must have
            # loaded the master itself. Fail loud rather than proceed nameless.
            raise RuntimeError(
                "CollectiveBackend.join: non-root rank has no params and "
                "init_broadcast is disabled."
            )

        self._names = list(names)
        self._global = {n: master[n].clone() for n in self._names}
        self._master_params = torch.nn.ParameterList(
            [
                torch.nn.Parameter(master[n].clone(), requires_grad=False)
                for n in self._names
            ]
        )
        self._outer_optimizer = factory(self._master_params.parameters())
        return {n: self._global[n].clone() for n in self._names}

    def synchronize(self, *, worker_id: str, pseudograds: "StateDict") -> SyncResult:
        """All-reduce this replica's pseudo-gradient to the group mean, run the
        replicated outer step, and return the new (identical-across-ranks)
        global params."""
        from torch.distributed import ReduceOp, all_reduce

        sent = 0
        reduced: Dict[str, torch.Tensor] = {}
        for name in self._names:
            pg = pseudograds.get(name)
            if pg is None:
                # Fail loud (mirrors the shared-mem backend): this backend
                # divides by group_size, which is only correct when every
                # replica contributes the full parameter set.
                raise ValueError(
                    f"CollectiveBackend: pseudograds missing '{name}'; every "
                    "replica must contribute the full parameter set."
                )
            buf = pg.detach().to(self._collective_device, torch.float32).contiguous()
            sent += buf.numel() * 4  # fp32 on the wire; no cast in increment 1
            if self.group_size > 1:
                all_reduce(buf, op=ReduceOp.SUM, group=self.process_group)
            buf.div_(self.group_size)  # SUM / N = mean across replicas
            reduced[name] = buf.cpu()

        # Replicated outer step on the CPU fp32 master, reproducing
        # server._apply_outer_optimizer (per-name mean grad, SGD-Nesterov). Every
        # rank steps an identical optimizer over identical inputs -> identical
        # output, so no result broadcast is needed.
        for i, name in enumerate(self._names):
            self._master_params[i].grad = reduced[name].reshape(
                self._master_params[i].shape
            )
        self._outer_optimizer.step()
        self._outer_optimizer.zero_grad()
        for i, name in enumerate(self._names):
            self._global[name] = self._master_params[i].data.clone()

        if self.broadcast_result and self.group_size > 1:
            # Safety valve: make group-local rank 0's stepped params
            # authoritative instead of relying on cross-rank all_reduce
            # bit-identity. Off by default.
            for name in self._names:
                t = self._global[name].to(self._collective_device)
                self._broadcast_in_place(t, group_src=0)
                self._global[name] = t.cpu()

        params = {n: self._global[n].clone() for n in self._names}
        # A collective sends and receives the same logical tensor volume.
        return SyncResult(
            params=params, committed=True, sent_bytes=sent, recv_bytes=sent
        )

    def _broadcast_in_place(self, tensor: torch.Tensor, group_src: int) -> None:
        import torch.distributed as dist

        dist.broadcast(tensor, group=self.process_group, group_src=group_src)

    def synchronize_fragment(
        self, *, worker_id: str, fragment_id: int, pseudograds: "StateDict"
    ) -> SyncResult:
        raise NotImplementedError(
            "CollectiveBackend does not support streaming-fragment sync "
            "(num_fragments > 1) yet; use full-model sync."
        )

    def current_global_params(self) -> "StateDict":
        return {n: self._global[n].clone() for n in self._names}

    def leave(self, *, worker_id: str) -> None:
        # Best-effort teardown of this process's state. The process group is
        # borrowed from the launcher (torchrun) — never destroy it here.
        self._outer_optimizer = None
        self._master_params = None
        self._global = {}
        self._names = []
