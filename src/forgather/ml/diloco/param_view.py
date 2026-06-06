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


def _cast_for_upload(
    pg: torch.Tensor, upload_dtype: str, upload_sr: bool
) -> torch.Tensor:
    """Cast a single pseudo-gradient tensor to the configured wire dtype.

    Shared by :class:`SimpleModelParamView`, :class:`PipelineParamView`,
    and the fragment-streaming code in
    :mod:`forgather.ml.diloco.fragments`. Centralising the cast means a
    future fp8 enum extension lands in exactly one place.

    Semantics:

    * ``upload_dtype == "fp32"``: pass through (no cast). Any input
      dtype is accepted; tensors smaller than fp32 are silently
      upcast by ``_send_tensor_response``'s serialisation if needed.
    * ``upload_dtype == "bf16"``: cast to bf16. With
      ``upload_sr=True`` and an fp32-resolution input, route through
      :func:`fp32_to_bf16_stochastic_round` to preserve sub-ULP signal
      in expectation; otherwise use round-to-nearest via
      ``.bfloat16()``. When the input is already bf16 the cast is
      identity and SR has no effect.
    """
    if upload_dtype == "fp32":
        return pg
    if upload_dtype == "bf16":
        if upload_sr and pg.dtype == torch.float32:
            from forgather.ml.optim.rounding_utils import (
                fp32_to_bf16_stochastic_round,
            )

            return fp32_to_bf16_stochastic_round(pg)
        return pg.to(torch.bfloat16)
    raise ValueError(f"unsupported upload_dtype: {upload_dtype!r}")


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
        upload_dtype: str = "bf16",
        upload_sr: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Return ``{name: snapshot[name] - current_param.cpu()}``.

        Both operands' dtypes are determined by the **live model
        dtype**: ``snapshot()`` clones the live params, and
        ``apply_global`` writes server responses back into the same
        live storage (any wire-side bf16 bytes get padded to the model
        dtype at ``apply_global``-time via ``.to(p.dtype)``). So the
        subtraction's dtype is whatever dtype the model carries — fp32
        for AMP / fp32-weight workers, bf16 for ``default_dtype=
        bfloat16`` workers — independent of ``download_dtype``.

        The result is then cast to ``upload_dtype`` for the wire —
        ``"bf16"`` halves submission bandwidth at the cost of
        precision; the server upcasts back to float32 before applying
        the outer optimizer step.

        When ``upload_dtype="bf16"`` and ``upload_sr=True`` and the
        subtraction landed in fp32 (AMP), the cast goes through
        :func:`forgather.ml.optim.rounding_utils.fp32_to_bf16_stochastic_round`
        — preserving sub-ULP signal in expectation. When the
        subtraction is already bf16 (true-bf16 worker) the cast is an
        identity and ``upload_sr`` has no effect.
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
        upload_dtype: str = "bf16",
        upload_sr: bool = False,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, p in self.named_parameters():
            # ``global_snapshot`` was built by ``snapshot()``, which
            # clones live params — so its dtype tracks the live model
            # dtype, not the wire dtype. The subtraction lands in fp32
            # when the live model is fp32 (AMP) and in bf16 when the
            # live model is bf16 (true-bf16). ``_cast_for_upload``
            # then either preserves (fp32 → fp32, bf16 → bf16) or
            # truncates (fp32 → bf16, optionally with SR).
            pg = global_snapshot[name] - p.data.cpu()
            out[name] = _cast_for_upload(pg, upload_dtype, upload_sr)
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
                # The wire dtype is server-side ``download_dtype`` —
                # when it doesn't match the live model dtype, ``.to()``
                # casts round-to-nearest. There is currently no
                # apply-side SR (the experiment matrix exercises this
                # via ``download_sr`` on the server, which casts before
                # the wire so the RNE here only widens the value back).
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
        upload_dtype: str = "bf16",
        upload_sr: bool = False,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, p in self.named_parameters():
            # Same dtype rules as ``SimpleModelParamView`` — snapshot
            # dtype tracks live model dtype, the subtraction inherits,
            # ``_cast_for_upload`` is the wire boundary.
            pg = global_snapshot[name] - p.data.cpu()
            out[name] = _cast_for_upload(pg, upload_dtype, upload_sr)
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
