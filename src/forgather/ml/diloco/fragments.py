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
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FragmentManager:
    """
    Manages splitting model parameters into fragments for streaming sync.

    Parameters are split into roughly equal contiguous groups. This keeps
    adjacent layers together, which is a natural fit for pipeline parallelism
    where each pipeline stage maps to one or more fragments.

    Args:
        model: The model whose parameters will be fragmented.
        num_fragments: Number of fragments to split the model into.
            Must be >= 1 and <= number of parameters.
    """

    def __init__(self, model: nn.Module, num_fragments: int):
        if num_fragments < 1:
            raise ValueError(f"num_fragments must be >= 1, got {num_fragments}")

        param_names = [name for name, _ in model.named_parameters()]
        if num_fragments > len(param_names):
            raise ValueError(
                f"num_fragments ({num_fragments}) exceeds number of "
                f"parameters ({len(param_names)})"
            )

        self.num_fragments = num_fragments

        # Split parameters into contiguous groups of roughly equal size
        self.fragments: List[List[str]] = self._split_contiguous(
            param_names, num_fragments
        )

        # Build reverse mapping: param_name -> fragment_id
        self.param_to_fragment: Dict[str, int] = {}
        for frag_id, names in enumerate(self.fragments):
            for name in names:
                self.param_to_fragment[name] = frag_id

        logger.info(
            f"FragmentManager: {len(param_names)} parameters split into "
            f"{num_fragments} fragments: "
            + ", ".join(f"frag {i}: {len(f)} params" for i, f in enumerate(self.fragments))
        )

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
        model: nn.Module,
        bf16_comm: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pseudo-gradients for a single fragment.

        pseudo_grad = global_params - local_params for each parameter in the
        fragment. Optionally cast to bfloat16 for bandwidth reduction.

        Args:
            fragment_id: Which fragment to compute pseudo-gradients for.
            global_params: CPU snapshot of global parameters.
            model: The model being trained (with current local params).
            bf16_comm: If True, cast pseudo-gradients to bfloat16.

        Returns:
            Dict mapping parameter names to pseudo-gradient tensors.
        """
        param_names = set(self.fragments[fragment_id])
        pseudograds = {}

        for name, p in model.named_parameters():
            if name in param_names:
                pg = global_params[name] - p.data.cpu()
                if bf16_comm:
                    pg = pg.to(torch.bfloat16)
                pseudograds[name] = pg

        return pseudograds

    def apply_fragment_global_params(
        self,
        fragment_id: int,
        new_params: Dict[str, torch.Tensor],
        model: nn.Module,
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
