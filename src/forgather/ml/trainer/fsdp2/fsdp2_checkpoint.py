"""
HuggingFace-compatible checkpoint save/load for FSDP2-sharded models.

FSDP2 replaces parameters with DTensors on a device mesh. The plain
``sharded_checkpoint.save_sharded_checkpoint`` / ``.load_checkpoint`` helpers
assume plain tensors and won't work directly: saving needs to gather shards to
rank 0, loading needs to broadcast-and-reshard from rank 0. PyTorch's
distributed checkpoint state_dict utilities provide exactly these primitives
(``get_model_state_dict`` with ``full_state_dict=True, cpu_offload=True`` for
save; ``set_model_state_dict`` with ``broadcast_from_rank0=True`` for load).

These helpers plug into ``CheckpointManager`` via its ``model_save_fn`` /
``model_load_fn`` hooks so that FSDP2 checkpoints are written in the standard
HuggingFace safetensors layout (``model-00001-of-000N.safetensors`` +
``model.safetensors.index.json``) and can be loaded by ``from_pretrained`` in
the transformers library — and symmetrically, the FSDP2 trainer can now
resume from any plain HF checkpoint, not just ones it produced itself.
"""

import logging
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from forgather.ml.distributed import DistributedEnvInterface
from forgather.ml.sharded_checkpoint import (
    create_sharing_metadata,
    index_file_name,
)
from forgather.ml.sharded_checkpoint import load_checkpoint as sharded_load_checkpoint
from forgather.ml.sharded_checkpoint import (
    make_shard_index,
    save_shard_index,
    save_sharded_checkpoint,
)

logger = logging.getLogger(__name__)


def save_fsdp2_model_as_hf(
    model: torch.nn.Module,
    output_dir: str,
    dist: DistributedEnvInterface,
    safetensors: bool = True,
    max_shard_size: int = 2**32,
) -> None:
    """
    Save an FSDP2-sharded model as a HuggingFace-compatible sharded checkpoint.

    All ranks must call this: ``get_model_state_dict(full_state_dict=True)``
    is a collective that gathers the sharded DTensors onto rank 0 (with CPU
    offload). Only rank 0 writes files; other ranks return after the gather.

    Parameters
    ----------
    model : torch.nn.Module
        FSDP2-wrapped model (after ``fully_shard`` has been applied).
    output_dir : str
        Target directory for shard files and index.
    dist : DistributedEnvInterface
        Distributed environment (used for rank gating).
    safetensors : bool, optional
        Emit safetensors (default) vs pytorch_model.bin format.
    max_shard_size : int, optional
        Maximum shard file size in bytes.
    """
    # Collective: gather full state dict to rank 0. Non-zero ranks receive
    # an empty dict; CPU offload keeps GPU memory from spiking on rank 0.
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_state_dict: Dict[str, Tensor] = get_model_state_dict(  # type: ignore[assignment]
        model, options=opts
    )

    if dist.rank == 0:
        # Preserve tied-weight metadata so the HF-side load can re-tie.
        sharing_metadata = create_sharing_metadata(model) or None

        shard_index = make_shard_index(
            [full_state_dict],
            safetensors=safetensors,
            max_shard_size=max_shard_size,
            param_sharing_metadata=sharing_metadata,
        )
        save_shard_index(shard_index, output_dir, index_file_name(safetensors))
        save_sharded_checkpoint(
            output_dir,
            shard_index,
            full_state_dict,
            safetensors=safetensors,
        )
        logger.info(
            f"Saved FSDP2 model as HF-format checkpoint at {output_dir} "
            f"({len(full_state_dict)} tensors)"
        )

    # Free the gathered dict on every rank as soon as rank 0's write is
    # queued; the set_model_state_dict path on load will re-allocate.
    del full_state_dict


def load_fsdp2_model_from_hf(
    model: torch.nn.Module,
    checkpoint_path: str,
    dist: DistributedEnvInterface,
    strict: bool = False,
) -> None:
    """
    Load a HuggingFace-format checkpoint into an FSDP2-sharded model.

    All ranks must call this. Rank 0 reads the full state dict from disk
    (auto-detecting safetensors / pytorch_model.bin, sharded or single-file);
    other ranks pass an empty dict. ``set_model_state_dict`` with
    ``broadcast_from_rank0=True`` then broadcasts and reshards into each
    rank's local DTensor slices.

    Re-ties weights after load so that safetensors' deduplicated storage is
    re-tied on the live module.

    Parameters
    ----------
    model : torch.nn.Module
        FSDP2-wrapped model to load weights into.
    checkpoint_path : str
        Path to HF checkpoint directory.
    dist : DistributedEnvInterface
        Distributed environment (used for rank gating).
    strict : bool, optional
        If ``True``, raise if the checkpoint misses keys the model needs.
        Default ``False`` so a plain HF checkpoint whose tensor names are a
        superset / subset of the model's can still load.
    """
    full_state_dict: Optional[dict] = None
    if dist.rank == 0:
        # Dict mode: return a raw state dict rather than assigning to a
        # module, since the DTensor model needs the broadcast path below.
        full_state_dict = sharded_load_checkpoint(
            checkpoint_path,
            module=None,
            device="cpu",
        )
        logger.info(
            f"Rank 0 read {len(full_state_dict)} tensors from HF-format "
            f"checkpoint at {checkpoint_path}"
        )
    else:
        full_state_dict = {}

    opts = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        strict=strict,
    )
    set_model_state_dict(model, full_state_dict, options=opts)

    # Safetensors can't represent shared storage, so tied weights are written
    # as separate tensors and must be re-tied on load. tie_weights() is
    # idempotent when the storage is already shared.
    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
        tie_weights()

    del full_state_dict
