"""
Utility functions for distributed checkpoint coordination.

This module provides utilities for working with process groups in distributed
checkpoint coordination, including:
- Process group rank introspection
- Group membership detection
- Rank selection for group-based saving
- Metadata collection via all-gather
- Replication validation with tensor-level checksums
"""

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """
    Validation thoroughness levels for replication checking.

    NONE: No validation (fastest, no overhead)

    QUICK: Hash-based validation (fast, catches most structural issues)
           - Minimal overhead
           - May miss subtle numerical differences
           - Device strings normalized for cross-rank comparison

    TENSOR: Tensor-level checksums (moderate speed, accurate)
            - Reasonable overhead for most models
            - Detects shape, dtype, device, and value differences

    FULL: Full tensor comparison (slow, comprehensive)
          - WARNING: High memory usage! Gathers all tensors from all ranks.
          - Memory required: ~world_size * model_size
          - Use only for debugging or small models
          - Can cause OOM on large models with many ranks
    """
    NONE = "none"
    QUICK = "quick"
    TENSOR = "tensor"
    FULL = "full"


@dataclass
class TensorChecksum:
    """
    Checksum information for a single tensor.

    Includes metadata and checksums that can detect:
    - Shape mismatches
    - Dtype differences
    - Numerical differences (within tolerance)
    """
    name: str
    shape: Tuple[int, ...]
    dtype: str
    device: str
    numel: int
    checksum: str  # Hex checksum of tensor data
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0


@dataclass
class StateChecksum:
    """
    Complete checksum information for a state_dict.

    Includes individual tensor checksums plus overall hash.
    """
    overall_hash: str
    tensor_checksums: Dict[str, TensorChecksum]
    num_tensors: int
    total_elements: int


def get_group_rank(process_group: Optional[ProcessGroup] = None) -> int:
    """
    Get the rank of the current process within a process group.

    Args:
        process_group: The process group to query. If None, uses default group.

    Returns:
        Rank within the process group (0 to group_size-1)

    Note:
        PyTorch's dist.get_rank() returns global rank by default, but
        can take a group parameter to get rank within that group.
    """
    if process_group is None:
        return dist.get_rank()

    try:
        return dist.get_rank(process_group)
    except Exception:
        # Fallback: try to determine from global rank and group membership
        # This is a workaround for older PyTorch versions
        logger.warning(
            "Could not get group rank directly, falling back to heuristic"
        )
        return _estimate_group_rank(process_group)


def get_group_size(process_group: Optional[ProcessGroup] = None) -> int:
    """
    Get the size of a process group.

    Args:
        process_group: The process group to query. If None, uses default group.

    Returns:
        Number of ranks in the process group
    """
    if process_group is None:
        return dist.get_world_size()

    try:
        return dist.get_world_size(process_group)
    except Exception:
        logger.warning(
            "Could not get group size directly, falling back to heuristic"
        )
        return _estimate_group_size(process_group)


def _estimate_group_rank(process_group: ProcessGroup) -> int:
    """
    Estimate group rank using group membership.

    This is a fallback for when dist.get_rank(group) is not available.
    Not guaranteed to be correct for arbitrary group configurations.
    """
    # Try to use torch.distributed's internal APIs if available
    if hasattr(process_group, "_get_backend_name"):
        # New PyTorch versions may have better introspection
        pass

    # Fallback: return 0 and warn
    # This is safe but not optimal - only rank 0 in group will save
    logger.warning(
        "Could not determine group rank, defaulting to 0. "
        "This may cause duplicate saves in complex group configurations."
    )
    return 0


def _estimate_group_size(process_group: ProcessGroup) -> int:
    """
    Estimate group size.

    Fallback for when dist.get_world_size(group) is not available.
    """
    # Fallback: return global world size and warn
    global_size = dist.get_world_size()
    logger.warning(
        f"Could not determine group size, defaulting to global world size ({global_size})"
    )
    return global_size


def is_group_leader(process_group: Optional[ProcessGroup] = None) -> bool:
    """
    Check if the current rank is the leader (rank 0) of a process group.

    Args:
        process_group: The process group to check. If None, checks global rank.

    Returns:
        True if this is rank 0 in the group, False otherwise

    This is used to determine which rank should save for PER_GROUP patterns.
    """
    return get_group_rank(process_group) == 0


def get_node_rank() -> int:
    """
    Get the rank of the current node.

    Nodes are numbered 0 to num_nodes-1. All ranks on the same node
    have the same node rank.

    Returns:
        Node rank (0-indexed)

    Note:
        Computed as global_rank // local_world_size
    """
    try:
        import os
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))

        if local_world_size == 0:
            return 0

        return rank // local_world_size
    except Exception as e:
        logger.warning(f"Could not determine node rank: {e}")
        return 0


def get_num_nodes() -> int:
    """
    Get the total number of nodes in the distributed job.

    Returns:
        Number of nodes

    Note:
        Computed as world_size // local_world_size
    """
    try:
        import os
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

        if local_world_size == 0:
            return 1

        # Number of nodes is world_size divided by processes per node
        num_nodes = world_size // local_world_size

        # Handle case where world_size is not evenly divisible
        if world_size % local_world_size != 0:
            num_nodes += 1

        return max(1, num_nodes)
    except Exception as e:
        logger.warning(f"Could not determine number of nodes: {e}")
        return 1


def is_node_leader() -> bool:
    """
    Check if the current rank is the leader (local rank 0) on its node.

    Returns:
        True if this is local rank 0, False otherwise

    This is used to determine which rank should save for PER_NODE patterns.
    """
    try:
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return local_rank == 0
    except Exception:
        return True  # Conservative: assume leader if can't determine


def all_gather_scalar(value: int, group: Optional[ProcessGroup] = None) -> List[int]:
    """
    All-gather a scalar integer value from all ranks.

    Args:
        value: The integer to gather from this rank
        group: Process group to use. If None, uses default group.

    Returns:
        List of values from all ranks (length = group_size)

    This is used to collect metadata like file sizes, ranks that saved, etc.
    """
    if not dist.is_initialized():
        return [value]

    world_size = get_group_size(group)

    # Convert scalar to tensor
    value_tensor = torch.tensor([value], dtype=torch.long)

    # Move to appropriate device
    if torch.cuda.is_available():
        value_tensor = value_tensor.cuda()

    # Prepare output tensor list
    gathered_tensors = [
        torch.zeros_like(value_tensor) for _ in range(world_size)
    ]

    # All-gather
    try:
        dist.all_gather(gathered_tensors, value_tensor, group=group)
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.warning(f"all_gather failed on rank {rank}: {e}, returning single value")
        return [value]

    # Convert back to list of ints
    return [int(t.item()) for t in gathered_tensors]


def all_gather_object_list(
    obj: any, group: Optional[ProcessGroup] = None
) -> List[any]:
    """
    All-gather arbitrary Python objects from all ranks.

    Args:
        obj: The Python object to gather from this rank
        group: Process group to use. If None, uses default group.

    Returns:
        List of objects from all ranks (length = group_size)

    This is used to collect complex metadata like component manifests.
    """
    if not dist.is_initialized():
        return [obj]

    world_size = get_group_size(group)

    # Prepare output list
    gathered_objects = [None] * world_size

    # All-gather objects
    try:
        dist.all_gather_object(gathered_objects, obj, group=group)
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.warning(f"all_gather_object failed on rank {rank}: {e}, returning single object")
        return [obj]

    return gathered_objects


def collect_group_savers(
    process_groups: Dict[str, ProcessGroup]
) -> Dict[str, List[int]]:
    """
    Collect which ranks are leaders for each process group.

    Args:
        process_groups: Dictionary mapping group names to ProcessGroup objects

    Returns:
        Dictionary mapping group names to list of global ranks that are
        leaders (rank 0) within that group

    Example:
        process_groups = {"dp_group": dp_pg, "pp_group": pp_pg}
        savers = collect_group_savers(process_groups)
        # savers = {"dp_group": [0, 4, 8, 12], "pp_group": [0, 1, 2, 3]}

    This is used to populate manifest with accurate information about
    which ranks saved for PER_GROUP components.
    """
    savers = {}

    for group_name, pg in process_groups.items():
        # Check if this rank is leader in this group
        is_leader = is_group_leader(pg)

        # Gather leader status from all ranks
        global_rank = dist.get_rank() if dist.is_initialized() else 0

        # Use object all-gather to collect (rank, is_leader) pairs
        rank_info = (global_rank, is_leader)
        all_rank_info = all_gather_object_list(rank_info, group=None)

        # Extract ranks that are leaders
        leader_ranks = [rank for rank, is_lead in all_rank_info if is_lead]
        savers[group_name] = sorted(leader_ranks)

    return savers


def collect_node_savers() -> List[int]:
    """
    Collect which ranks are node leaders (local rank 0).

    Returns:
        List of global ranks that are local rank 0 on their node,
        sorted in ascending order

    This is used to populate manifest with accurate information about
    which ranks saved for PER_NODE components.
    """
    if not dist.is_initialized():
        return [0]

    # Check if this rank is node leader
    is_leader = is_node_leader()
    global_rank = dist.get_rank()

    # Gather (rank, is_leader) pairs from all ranks
    rank_info = (global_rank, is_leader)
    all_rank_info = all_gather_object_list(rank_info, group=None)

    # Extract ranks that are node leaders
    leader_ranks = [rank for rank, is_lead in all_rank_info if is_lead]
    return sorted(leader_ranks)


def get_group_file_suffix(
    group_name: str, process_group: ProcessGroup
) -> str:
    """
    Get a unique file suffix for this rank's position in a group.

    Args:
        group_name: Name of the process group
        process_group: The ProcessGroup object

    Returns:
        String suffix like "group_dp_group_grank_0_rank_3"
        where grank is rank within group, rank is global rank

    This creates unique, predictable filenames for PER_GROUP saves.
    """
    group_rank = get_group_rank(process_group)
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    return f"group_{group_name}_grank_{group_rank}_rank_{global_rank}"


def get_node_file_suffix() -> str:
    """
    Get a unique file suffix for this rank's node.

    Returns:
        String suffix like "node_2_rank_8"
        where node is node rank, rank is global rank

    This creates unique, predictable filenames for PER_NODE saves.
    """
    node_rank = get_node_rank()
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    return f"node_{node_rank}_rank_{global_rank}"


def find_group_checkpoint_file(
    checkpoint_path: str,
    component_key: str,
    group_name: str,
    process_group: ProcessGroup,
) -> Optional[str]:
    """
    Find the checkpoint file for this rank's group.

    Args:
        checkpoint_path: Directory containing checkpoint files
        component_key: Component key (e.g., "model", "optimizer")
        group_name: Name of the process group
        process_group: The ProcessGroup object

    Returns:
        Path to checkpoint file, or None if not found

    This handles loading PER_GROUP components by finding the file
    saved by the group leader that matches this rank's group membership.
    """
    import os
    import glob

    # Try to find file using group rank
    group_rank = get_group_rank(process_group)

    # Pattern: {key}_state_group_{group_name}_grank_{group_rank}_rank_*.pt
    pattern = os.path.join(
        checkpoint_path,
        f"{component_key}_state_group_{group_name}_grank_{group_rank}_rank_*.pt"
    )

    matches = glob.glob(pattern)

    if matches:
        return matches[0]  # Return first match

    # Fallback: try without group rank (legacy format)
    pattern = os.path.join(
        checkpoint_path,
        f"{component_key}_state_group_{group_name}_rank_*.pt"
    )

    matches = glob.glob(pattern)

    if matches:
        logger.warning(
            f"Found legacy checkpoint file for {component_key} in group {group_name}: {matches[0]}"
        )
        return matches[0]

    return None


def find_node_checkpoint_file(
    checkpoint_path: str,
    component_key: str,
) -> Optional[str]:
    """
    Find the checkpoint file for this rank's node.

    Args:
        checkpoint_path: Directory containing checkpoint files
        component_key: Component key (e.g., "cache", "node_data")

    Returns:
        Path to checkpoint file, or None if not found

    This handles loading PER_NODE components by finding the file
    saved by the node leader for this node.
    """
    import os
    import glob

    node_rank = get_node_rank()

    # Pattern: {key}_state_node_{node_rank}_rank_*.pt
    pattern = os.path.join(
        checkpoint_path,
        f"{component_key}_state_node_{node_rank}_rank_*.pt"
    )

    matches = glob.glob(pattern)

    if matches:
        return matches[0]  # Return first match

    return None


def compute_tensor_checksum(name: str, tensor: torch.Tensor) -> TensorChecksum:
    """
    Compute checksum for a single tensor.

    Args:
        name: Name of the tensor (key in state_dict)
        tensor: The tensor to checksum

    Returns:
        TensorChecksum with metadata and checksum

    The checksum includes:
    - Shape, dtype, device metadata
    - Hash of tensor bytes
    - Statistical summary (mean, std, min, max)
    """
    # Basic metadata
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype)
    device = str(tensor.device)
    numel = tensor.numel()

    # Move to CPU for checksumming
    cpu_tensor = tensor.detach().cpu()

    # Compute hash of tensor bytes
    tensor_bytes = cpu_tensor.numpy().tobytes()
    checksum = hashlib.sha256(tensor_bytes).hexdigest()[:16]  # First 16 hex chars

    # Compute statistics for floating point tensors
    mean = std = tensor_min = tensor_max = 0.0
    if tensor.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        try:
            # Convert to float32 for stats (bfloat16 doesn't support some ops)
            stats_tensor = cpu_tensor.float()
            mean = float(stats_tensor.mean().item())
            std = float(stats_tensor.std().item())
            tensor_min = float(stats_tensor.min().item())
            tensor_max = float(stats_tensor.max().item())
        except Exception as e:
            logger.debug(f"Could not compute stats for tensor {name}: {e}")

    return TensorChecksum(
        name=name,
        shape=shape,
        dtype=dtype,
        device=device,
        numel=numel,
        checksum=checksum,
        mean=mean,
        std=std,
        min=tensor_min,
        max=tensor_max,
    )


def compute_state_checksum(
    state_dict: Dict[str, Any],
    validation_level: ValidationLevel = ValidationLevel.TENSOR,
) -> StateChecksum:
    """
    Compute checksums for an entire state_dict.

    Args:
        state_dict: State dictionary to checksum
        validation_level: Level of validation detail

    Returns:
        StateChecksum with per-tensor and overall checksums

    The level controls thoroughness:
    - QUICK: Only overall hash (fast)
    - TENSOR: Per-tensor checksums (moderate)
    - FULL: Detailed per-tensor checksums with stats (slow)
    """
    tensor_checksums = {}
    total_elements = 0

    if validation_level in (ValidationLevel.TENSOR, ValidationLevel.FULL):
        # Compute per-tensor checksums
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                tensor_checksums[key] = compute_tensor_checksum(key, value)
                total_elements += value.numel()
            elif isinstance(value, dict):
                # Handle nested dicts (e.g., optimizer state)
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        full_key = f"{key}.{subkey}"
                        tensor_checksums[full_key] = compute_tensor_checksum(
                            full_key, subvalue
                        )
                        total_elements += subvalue.numel()

    # Compute overall hash
    # For QUICK mode, just hash the serialized state_dict
    # For TENSOR/FULL modes, hash the per-tensor checksums for efficiency
    if validation_level == ValidationLevel.QUICK:
        from forgather.ml.trainer.checkpoint_types import compute_state_hash
        overall_hash = compute_state_hash(state_dict)
    else:
        # Hash the concatenated tensor checksums
        checksum_str = "".join(
            f"{tc.name}:{tc.shape}:{tc.dtype}:{tc.checksum}"
            for tc in sorted(tensor_checksums.values(), key=lambda x: x.name)
        )
        overall_hash = hashlib.sha256(checksum_str.encode()).hexdigest()

    return StateChecksum(
        overall_hash=overall_hash,
        tensor_checksums=tensor_checksums,
        num_tensors=len(tensor_checksums),
        total_elements=total_elements,
    )


def validate_replication(
    state_dict: Dict[str, Any],
    validation_level: ValidationLevel = ValidationLevel.TENSOR,
    group: Optional[ProcessGroup] = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[bool, List[str]]:
    """
    Validate that state is replicated (identical) across all ranks.

    Args:
        state_dict: State dictionary to validate
        validation_level: Level of validation thoroughness
        group: Process group to validate across (None = all ranks)
        rtol: Relative tolerance for floating point comparison
        atol: Absolute tolerance for floating point comparison

    Returns:
        Tuple of (is_valid, error_messages)
        - is_valid: True if all ranks have identical state
        - error_messages: List of discrepancy descriptions (empty if valid)

    Validation levels:
    - NONE: No validation, always returns (True, [])
    - QUICK: Hash comparison (fast, detects any differences)
    - TENSOR: Per-tensor checksum comparison (moderate, more detailed errors)
    - FULL: Full tensor comparison (slow, exact differences)
    """
    if validation_level == ValidationLevel.NONE:
        return True, []

    if not dist.is_initialized():
        return True, []  # Single rank, nothing to validate

    errors = []

    if validation_level == ValidationLevel.QUICK:
        # Quick hash-based validation
        from forgather.ml.trainer.checkpoint_types import compute_state_hash

        local_hash = compute_state_hash(state_dict)

        # Convert hash to integer for all-gather
        # Use first 15 hex chars (60 bits) to avoid int64 overflow
        # (16 hex chars = 64 bits can exceed signed int64 range of 2^63-1)
        hash_int = int(local_hash[:15], 16)
        all_hashes = all_gather_scalar(hash_int, group)

        # Check if all hashes match
        if len(set(all_hashes)) > 1:
            errors.append(
                f"State hash mismatch across ranks. Unique hashes: {len(set(all_hashes))}"
            )
            return False, errors

        return True, []

    elif validation_level == ValidationLevel.TENSOR:
        # Tensor-level checksum validation
        local_checksum = compute_state_checksum(state_dict, ValidationLevel.TENSOR)

        # Gather checksums from all ranks
        all_checksums = all_gather_object_list(local_checksum, group)

        # Compare with rank 0's checksum
        ref_checksum = all_checksums[0]

        # Check number of tensors
        if local_checksum.num_tensors != ref_checksum.num_tensors:
            errors.append(
                f"Tensor count mismatch: rank 0 has {ref_checksum.num_tensors}, "
                f"this rank has {local_checksum.num_tensors}"
            )

        # Check individual tensor checksums
        for tensor_name, local_tc in local_checksum.tensor_checksums.items():
            if tensor_name not in ref_checksum.tensor_checksums:
                errors.append(f"Tensor '{tensor_name}' missing on rank 0")
                continue

            ref_tc = ref_checksum.tensor_checksums[tensor_name]

            # Check shape
            if local_tc.shape != ref_tc.shape:
                errors.append(
                    f"Tensor '{tensor_name}' shape mismatch: "
                    f"rank 0 has {ref_tc.shape}, this rank has {local_tc.shape}"
                )

            # Check dtype
            if local_tc.dtype != ref_tc.dtype:
                errors.append(
                    f"Tensor '{tensor_name}' dtype mismatch: "
                    f"rank 0 has {ref_tc.dtype}, this rank has {local_tc.dtype}"
                )

            # Check checksum
            if local_tc.checksum != ref_tc.checksum:
                errors.append(
                    f"Tensor '{tensor_name}' data mismatch (checksum differs)"
                )

        return len(errors) == 0, errors

    elif validation_level == ValidationLevel.FULL:
        # Full tensor comparison (expensive!)
        # This compares actual tensor values across ranks

        # Get list of tensor keys
        tensor_keys = []
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                tensor_keys.append(key)

        # For each tensor, gather from all ranks and compare
        for key in tensor_keys:
            tensor = state_dict[key]

            # Gather tensors from all ranks
            world_size = get_group_size(group)
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

            try:
                dist.all_gather(gathered_tensors, tensor, group=group)
            except Exception as e:
                logger.warning(f"Could not all-gather tensor '{key}': {e}")
                continue

            # Compare with rank 0's tensor
            ref_tensor = gathered_tensors[0]

            # Check if tensors are close (within tolerance)
            if tensor.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                if not torch.allclose(tensor, ref_tensor, rtol=rtol, atol=atol):
                    # Compute max difference
                    max_diff = (tensor - ref_tensor).abs().max().item()
                    errors.append(
                        f"Tensor '{key}' values differ (max diff: {max_diff:.2e})"
                    )
            else:
                # Integer or other types: exact comparison
                if not torch.equal(tensor, ref_tensor):
                    errors.append(f"Tensor '{key}' values differ (not equal)")

        return len(errors) == 0, errors

    else:
        logger.warning(f"Unknown validation level: {validation_level}")
        return True, []
