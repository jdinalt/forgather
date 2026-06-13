"""
Fragment management for Streaming DiLoCo.

Splits model parameters into N fragments for staggered synchronization,
enabling communication-computation overlap. Instead of one large sync every
H steps, each fragment syncs every H/N steps, with communication happening
in the background while training continues on other fragments.

Usage:
    fm = FragmentManager(model, num_fragments=4)

    # In the training loop post-step hook:
    frag_id = fm.get_fragment_schedule(local_step, sync_every=600)
    if frag_id is not None:
        pseudograds = fm.compute_fragment_pseudogradients(frag_id, global_params, model)
        # Submit in background thread...
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NoBlockPlanError(Exception):
    """Raised when a model exposes no usable transformer-block structure, so
    block-faithful fragmentation isn't possible and we fall back to a contiguous
    split."""


def discover_block_boundaries(
    model,
) -> Tuple[List[List[str]], List[str], List[str]]:
    """Derive transformer-block parameter groups from a model's pipeline-parallel
    metadata, following Streaming DiLoCo (arXiv:2501.18512), which fragments the
    model on transformer-block boundaries rather than by raw parameter count.

    Blocks are identified by ``_no_split_modules`` (the HF convention for the
    atomic transformer-block classes that must never be split across a pipeline
    boundary — the same signal vLLM uses, see docs/inference/vllm_integration.md).
    Each such submodule's fully-qualified name is an indexed child of a single
    container (e.g. ``causal_lm.layer_stack.layers.{i}``); its parameters form
    one atomic group, ordered by the integer index.

    Returns ``(block_param_groups, pre_block_params, post_block_params)`` where
    ``block_param_groups[i]`` is the ordered list of parameter names in block
    ``i``, and the pre/post lists hold the non-block parameters (embeddings;
    final norm + LM head) that appear before / after the blocks in the model's
    parameter order.

    Raises :class:`NoBlockPlanError` if the model defines no ``_no_split_modules``,
    has no matching block submodules, or the blocks don't sit under a single
    integer-indexed container (so the caller can fall back to a contiguous split).
    """
    no_split = getattr(model, "_no_split_modules", None) or getattr(
        type(model), "_no_split_modules", None
    )
    if not no_split:
        raise NoBlockPlanError("model defines no _no_split_modules")
    no_split = set(no_split)

    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise NoBlockPlanError("model has no named_modules() (not an nn.Module)")

    # Block submodules, identified by class name. Dedupe by FQN (a proxy
    # ModuleList over a ModuleDict can register the same child twice).
    block_fqns = []
    seen = set()
    for name, mod in named_modules():
        if type(mod).__name__ in no_split and name and name not in seen:
            seen.add(name)
            block_fqns.append(name)
    if not block_fqns:
        raise NoBlockPlanError(
            f"no submodules match _no_split_modules classes {sorted(no_split)}"
        )

    # All blocks must be the integer-indexed children of one container, e.g.
    # ``<prefix>.0``, ``<prefix>.1``, ... — that ordering is the block order.
    def _index(fqn: str) -> Optional[int]:
        tail = fqn.rsplit(".", 1)[-1]
        return int(tail) if tail.isdigit() else None

    containers = {fqn.rsplit(".", 1)[0] for fqn in block_fqns}
    if len(containers) != 1 or any(_index(f) is None for f in block_fqns):
        raise NoBlockPlanError(
            f"blocks are not the indexed children of a single container "
            f"(containers={sorted(containers)})"
        )
    block_fqns.sort(key=_index)

    all_params = [name for name, _ in model.named_parameters()]
    block_prefixes = [f"{fqn}." for fqn in block_fqns]
    block_groups: List[List[str]] = []
    block_param_set = set()
    for bp in block_prefixes:
        names = [n for n in all_params if n.startswith(bp)]
        block_groups.append(names)
        block_param_set.update(names)
    if not block_param_set:
        raise NoBlockPlanError("block submodules carry no parameters")

    # Non-block params: classify by position relative to the block params in the
    # model's parameter order — before the first block => pre (embeddings),
    # otherwise => post (final norm, LM head). This assumes the non-block params
    # bracket the blocks (true for every transformer in-repo: embeddings first,
    # norm + head last). A param interleaved *between* blocks would be attached
    # to the last fragment; the partition stays valid (every name in exactly one
    # fragment), just slightly less balanced.
    first_block_idx = min(i for i, n in enumerate(all_params) if n in block_param_set)
    pre_block_params, post_block_params = [], []
    for i, n in enumerate(all_params):
        if n in block_param_set:
            continue
        (pre_block_params if i < first_block_idx else post_block_params).append(n)

    return block_groups, pre_block_params, post_block_params


class FragmentManager:
    """
    Manages splitting model parameters into fragments for streaming sync.

    Following Streaming DiLoCo (arXiv:2501.18512), parameters are split on
    **transformer-block boundaries**: each fragment is a set of whole blocks
    (plus the non-block params attached to the first/last fragment), assigned
    either ``sequential`` (contiguous runs of blocks) or ``strided`` (block
    ``i`` -> fragment ``i % N``, the paper's mild preference). When the model
    exposes no block plan (no ``_no_split_modules`` / not a transformer), it
    falls back to an equal-param-count contiguous split with a warning.

    Args:
        model: The model (or ``ParamView``) whose parameters will be fragmented.
        num_fragments: Number of fragments. Must be >= 1.
        assignment: ``"strided"`` (default) or ``"sequential"`` block-to-fragment
            assignment.
        boundary_source: The full model to derive block boundaries from (it must
            expose ``_no_split_modules`` + ``named_modules()``). Defaults to
            ``model``; under pipeline parallel the worker passes the real model
            here while ``model`` is the rank's ``ParamView``.
    """

    def __init__(
        self,
        model,
        num_fragments: int,
        assignment: str = "strided",
        boundary_source=None,
    ):
        """``model`` may be an ``nn.Module`` or a ``ParamView`` (duck-typed via
        ``.named_parameters()``). Under pipeline parallel the view exposes only
        the rank's slice; block boundaries are discovered globally from
        ``boundary_source`` and then filtered to the slice, so the logical
        fragment ids stay consistent across ranks."""
        if num_fragments < 1:
            raise ValueError(f"num_fragments must be >= 1, got {num_fragments}")
        if assignment not in ("strided", "sequential"):
            raise ValueError(
                f"assignment must be 'strided' or 'sequential', got {assignment!r}"
            )

        param_names = [name for name, _ in model.named_parameters()]
        if num_fragments > len(param_names):
            raise ValueError(
                f"num_fragments ({num_fragments}) exceeds number of "
                f"parameters ({len(param_names)})"
            )

        self.num_fragments = num_fragments
        self.assignment = assignment

        src = boundary_source if boundary_source is not None else model
        try:
            self.fragments: List[List[str]] = self._split_blocks(
                param_names, num_fragments, assignment, src
            )
            self.block_faithful = True
        except NoBlockPlanError as exc:
            logger.warning(
                "FragmentManager: %s; falling back to equal-param-count "
                "contiguous fragments (NOT paper-faithful).",
                exc,
            )
            self.fragments = self._split_contiguous(param_names, num_fragments)
            self.block_faithful = False

        # Build reverse mapping: param_name -> fragment_id
        self.param_to_fragment: Dict[str, int] = {}
        for frag_id, names in enumerate(self.fragments):
            for name in names:
                self.param_to_fragment[name] = frag_id

        kind = "block" if self.block_faithful else "contiguous(fallback)"
        logger.info(
            f"FragmentManager: {len(param_names)} parameters split into "
            f"{num_fragments} {assignment} {kind} fragments: "
            + ", ".join(
                f"frag {i}: {len(f)} params" for i, f in enumerate(self.fragments)
            )
        )

    @staticmethod
    def _split_blocks(
        param_names: List[str],
        num_fragments: int,
        assignment: str,
        boundary_source,
    ) -> List[List[str]]:
        """Split on transformer-block boundaries. Discovers blocks globally from
        ``boundary_source``, assigns whole blocks to fragments (sequential runs
        or strided), attaches the non-block params (embeddings -> first fragment,
        final norm + LM head -> last fragment), then filters each fragment to the
        names actually present in ``param_names`` (the rank's slice)."""
        block_groups, pre_params, post_params = discover_block_boundaries(
            boundary_source
        )
        num_blocks = len(block_groups)
        if num_fragments > num_blocks:
            raise NoBlockPlanError(
                f"num_fragments ({num_fragments}) exceeds transformer blocks "
                f"({num_blocks}); cannot split faithfully on block boundaries"
            )

        # Assign global block indices to logical fragments.
        if assignment == "strided":
            frag_block_idxs: List[List[int]] = [[] for _ in range(num_fragments)]
            for i in range(num_blocks):
                frag_block_idxs[i % num_fragments].append(i)
        else:  # sequential: contiguous runs, remainder distributed to the front
            frag_block_idxs = FragmentManager._split_contiguous(
                list(range(num_blocks)), num_fragments
            )

        fragments: List[List[str]] = [[] for _ in range(num_fragments)]
        for frag_id, block_idxs in enumerate(frag_block_idxs):
            for bi in block_idxs:
                fragments[frag_id].extend(block_groups[bi])

        # Non-block params: embeddings -> first fragment, final norm + head ->
        # last fragment (keeps every param in exactly one fragment).
        fragments[0] = pre_params + fragments[0]
        fragments[-1] = fragments[-1] + post_params

        # Filter to this rank's slice (a no-op when model == boundary_source).
        slice_set = set(param_names)
        return [[n for n in frag if n in slice_set] for frag in fragments]

    @staticmethod
    def _split_contiguous(items: List[str], n: int) -> List[List[str]]:
        """Split a list into n roughly equal contiguous chunks."""
        total = len(items)
        base_size = total // n
        remainder = total % n

        fragments = []
        start = 0
        for i in range(n):
            # First 'remainder' chunks get one extra item
            size = base_size + (1 if i < remainder else 0)
            fragments.append(items[start : start + size])
            start += size
        return fragments

    def get_fragment_param_names(self, fragment_id: int) -> List[str]:
        """Get parameter names belonging to a fragment."""
        return self.fragments[fragment_id]

    def get_fragment_schedule(self, local_step: int, sync_every: int) -> Optional[int]:
        """
        Determine which fragment should sync at this step, if any.

        Fragments are synced at evenly spaced intervals within the sync_every
        window. With sync_every=600 and 3 fragments:
        - Step 200: fragment 0
        - Step 400: fragment 1
        - Step 600: fragment 2

        Args:
            local_step: Current local step count (1-based, incremented before check).
            sync_every: Total steps between full model syncs.

        Returns:
            Fragment ID to sync, or None if no sync needed at this step.
        """
        fragment_interval = sync_every // self.num_fragments
        if fragment_interval <= 0:
            fragment_interval = 1

        if local_step <= 0 or local_step % fragment_interval != 0:
            return None

        fragment_idx = (local_step // fragment_interval - 1) % self.num_fragments
        return fragment_idx

    def is_last_fragment(self, local_step: int, sync_every: int) -> bool:
        """Check if the current step triggers the last fragment in a round."""
        frag_id = self.get_fragment_schedule(local_step, sync_every)
        return frag_id == self.num_fragments - 1

    def compute_fragment_pseudogradients(
        self,
        fragment_id: int,
        global_params: Dict[str, torch.Tensor],
        model,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute raw pseudo-gradients for a single fragment.

        pseudo_grad = global_params - local_params for each parameter in the
        fragment, in the live model dtype. Mirrors
        :meth:`ParamView.compute_pseudograds`; the wire-precision cast is the
        backend's job (applied in ``HttpStarBackend.synchronize_fragment``),
        not here.

        Args:
            fragment_id: Which fragment to compute pseudo-gradients for.
            global_params: CPU snapshot of global parameters.
            model: The model being trained (with current local params).

        Returns:
            Dict mapping parameter names to raw pseudo-gradient tensors.
        """
        param_names = set(self.fragments[fragment_id])
        pseudograds = {}

        for name, p in model.named_parameters():
            if name in param_names:
                pseudograds[name] = global_params[name] - p.data.cpu()

        return pseudograds

    def apply_fragment_global_params(
        self,
        fragment_id: int,
        new_params: Dict[str, torch.Tensor],
        model,
        global_params: Dict[str, torch.Tensor],
    ):
        """
        Apply updated global parameters for a single fragment.

        Updates both the model's live parameters and the CPU snapshot of
        global parameters used for pseudo-gradient computation.

        Args:
            fragment_id: Which fragment to update.
            new_params: Updated parameters from the server.
            model: The model to update.
            global_params: CPU snapshot dict to update in place.
        """
        param_names = set(self.fragments[fragment_id])

        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in param_names and name in new_params:
                    p.data.copy_(new_params[name].to(dtype=p.dtype, device=p.device))

        # Update global snapshot
        for name in new_params:
            if name in param_names:
                global_params[name] = new_params[name].detach().clone().cpu()
