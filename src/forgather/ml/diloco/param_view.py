"""ParamView — abstraction over the worker's read/write access to model params.

Introduced for issue #84 (DiLoCo + pipeline parallel). The worker's
five model-touching call sites — registration fingerprint, snapshot,
pseudo-gradient compute, global-params apply, and the (no-op-under-
pure-pipeline) DDP-follower broadcast — all route through a
``ParamView``. This lets the DiLoCoWorker stay model-shape-agnostic
while the view encapsulates whether parameters live on a single
``nn.Module`` or are distributed across a list of pipeline stages.

Two implementations:

- ``SimpleModelParamView(model)`` — wraps a single ``nn.Module``. The
  pre-#84 behavior of DiLoCoWorker. Used for non-pipeline trainers
  (single-GPU, DDP, FSDP2).
- ``PipelineParamView(pipeline_modules, sharing_metadata)`` — wraps
  the per-rank ``List[nn.Module]`` that a pipeline-parallel trainer
  stores on ``trainer.pipeline_modules``. Each rank's slice is
  exposed independently; the worker submits only its rank's slice.

Tied-parameter handling.

  - ``SimpleModelParamView`` uses ``remove_duplicate=True`` (PyTorch
    default) so tied parameters submit once under the canonical name.
    Compatible with HF/safetensors checkpoints, where the server's
    ``_param_names`` is the deduplicated set.
  - ``PipelineParamView`` uses ``remove_duplicate=False`` to match
    the pipeline trainer's checkpoint format
    (``make_state_dict(remove_duplicate=False)`` at
    ``pipeline_trainer.py:783``), where the server's ``_param_names``
    contains aliases. Within a single rank, ``retie_parameters``
    ensures aliases share storage so ``apply_global`` to one name
    updates all. Across ranks (alias on stage 0, alias on stage N),
    storage is NOT shared; each rank computes its own pseudo-gradient
    for its alias and the server's per-name averaging blends them —
    which is harmless for truly tied params (the pre-image is shared,
    so the local pseudo-gradients tend to be identical) but masks a
    real bug if the same name is held by two ranks for unrelated
    reasons. The seal-time coverage check doesn't distinguish these
    cases; a follow-up may add an explicit sharing-metadata block.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn


class ParamView:
    """Read/write abstraction over a worker's parameter set.

    Implementations encapsulate the shape of the model (single module
    vs. pipeline-distributed) so DiLoCoWorker can stay shape-agnostic.
    """

    def param_shapes(self) -> Dict[str, List[int]]:
        """Return ``{name: shape}`` for the slice this view exposes.

        Used at registration time so the server can verify the slice
        against its master param set. For a non-pipeline worker this
        is the full model; for a pipeline rank it's the rank's slice.
        """
        raise NotImplementedError

    def snapshot(self) -> Dict[str, torch.Tensor]:
        """CPU-resident clone of every parameter this view exposes.

        Returned dict is the "global snapshot" used as the pre-image
        for pseudo-gradient computation. Tensors are detached and
        live on CPU.
        """
        raise NotImplementedError

    def compute_pseudograds(
        self,
        global_snapshot: Dict[str, torch.Tensor],
        bf16: bool,
    ) -> Dict[str, torch.Tensor]:
        """Return ``{name: snapshot[name] - current_param.cpu()}``.

        ``bf16=True`` casts each pseudo-gradient to bfloat16 to halve
        the bandwidth of the upstream submission. The server upcasts
        back to float32 before applying the outer optimizer step.
        """
        raise NotImplementedError

    def apply_global(self, global_params: Dict[str, torch.Tensor]) -> None:
        """Copy server-returned global params back into live model storage.

        For tied-parameter aliases the view's underlying parameters
        share storage (the trainer's ``retie_parameters`` is called at
        construction time), so writing into one alias updates the
        others. Names in ``global_params`` that this view doesn't
        own are silently ignored — the server may return the full
        union for a fragment that includes other ranks' names.
        """
        raise NotImplementedError

    def named_parameters(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield ``(name, tensor)`` for every parameter this view owns."""
        raise NotImplementedError


class SimpleModelParamView(ParamView):
    """ParamView backed by a single ``nn.Module``.

    Pre-#84 behavior of DiLoCoWorker. Used when no pipeline trainer is
    detected — single-GPU, DDP, FSDP2, etc.

    Uses ``remove_duplicate=True`` (the PyTorch default) so tied
    parameters submit once under the canonical name. This preserves
    compatibility with HF/safetensors checkpoints (and any other
    ``state_dict`` saved with the default ``remove_duplicate=True``),
    where the server's ``_param_names`` is the deduplicated set. Sending
    both alias names would surface as "missing on server" → 422 from
    the slice fingerprint check.

    The flip side: a server initialized from a pipeline-trained
    checkpoint (``make_state_dict(remove_duplicate=False)``) has BOTH
    alias slots, and a solo worker submitting only the canonical name
    leaves the aliased slot un-updated. This is a known limitation of
    the cross-mode (solo worker ↔ pipeline-saved checkpoint) path;
    workers running under a pipeline trainer use ``PipelineParamView``
    which submits the full alias set.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def named_parameters(self) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from self.model.named_parameters()

    def param_shapes(self) -> Dict[str, List[int]]:
        return {name: list(p.shape) for name, p in self.named_parameters()}

    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {
            name: p.data.detach().clone().cpu() for name, p in self.named_parameters()
        }

    def compute_pseudograds(
        self,
        global_snapshot: Dict[str, torch.Tensor],
        bf16: bool,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, p in self.named_parameters():
            pg = global_snapshot[name] - p.data.cpu()
            if bf16:
                pg = pg.to(torch.bfloat16)
            out[name] = pg
        return out

    def apply_global(self, global_params: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, p in self.named_parameters():
                src = global_params.get(name)
                if src is None:
                    # Server returned a subset (e.g. fragment response).
                    # Skipping is the right behavior — the names we own
                    # but the server didn't update stay at their current
                    # value, which is what fragment-streaming intends.
                    continue
                p.data.copy_(src.to(dtype=p.dtype, device=p.device))


class PipelineParamView(ParamView):
    """ParamView backed by a list of pipeline-stage modules on one rank.

    The trainer (``pipeline_trainer.py``) materialises only its rank's
    stages on-device and stores them in ``trainer.pipeline_modules``.
    This view iterates those modules. Each module's
    ``named_parameters(remove_duplicate=False)`` is yielded with its
    fully-qualified name; the trainer's ``retie_parameters`` ensures
    tied aliases within a rank share storage.

    ``sharing_metadata`` is the trainer's ``create_sharing_metadata``
    output (``List[List[str]]``): each inner list is an equivalence
    class of aliased parameter names. Held for diagnostics; the actual
    sharing is in the underlying storage.
    """

    def __init__(
        self,
        pipeline_modules: List[nn.Module],
        sharing_metadata: Optional[List[List[str]]] = None,
    ):
        self.pipeline_modules = pipeline_modules
        self.sharing_metadata = sharing_metadata or []

    def named_parameters(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for mod in self.pipeline_modules:
            yield from mod.named_parameters(remove_duplicate=False)

    def param_shapes(self) -> Dict[str, List[int]]:
        return {name: list(p.shape) for name, p in self.named_parameters()}

    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {
            name: p.data.detach().clone().cpu() for name, p in self.named_parameters()
        }

    def compute_pseudograds(
        self,
        global_snapshot: Dict[str, torch.Tensor],
        bf16: bool,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, p in self.named_parameters():
            pg = global_snapshot[name] - p.data.cpu()
            if bf16:
                pg = pg.to(torch.bfloat16)
            out[name] = pg
        return out

    def apply_global(self, global_params: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, p in self.named_parameters():
                src = global_params.get(name)
                if src is None:
                    # Not in the server's response — either the server
                    # is sending only our slice's portion (good) or this
                    # is a fragment response that doesn't touch this
                    # name. Either way, skip.
                    continue
                p.data.copy_(src.to(dtype=p.dtype, device=p.device))
