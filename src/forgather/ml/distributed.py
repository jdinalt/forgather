"""
Distributed training environment utilities.

This module provides utilities for initializing and managing PyTorch distributed
training environments. It handles:

- Environment variable management for distributed training (RANK, WORLD_SIZE, etc.)
- Process group creation and caching (local per-node and global gloo groups)
- Barrier synchronization that works with both GPU (nccl) and CPU (gloo) backends
- Coordination primitives like main_local_process_first for dataset preprocessing

The module supports both GPU-accelerated training (typically using nccl backend)
and CPU-only distributed training (using gloo backend), which is useful for testing
distributed configurations without requiring GPUs.

Key components:
    - DistributedEnvironment: Main class for initializing distributed training
    - get_barrier_fn: Returns a barrier function appropriate for the current backend
    - main_local_process_first: Context manager for node-local coordination
    - get_local_process_group / get_global_process_group: Gloo-based process groups
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Protocol

import torch
from torch import accelerator
from torch import distributed as dist
from torch._C._distributed_c10d import Work
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# Tracks recursion depth of main_local_process_first to prevent nested barriers
mpf_recursion_level = 0

# Cached gloo process group containing only ranks on the local node.
# Created lazily by get_local_process_group().
_local_process_group: ProcessGroup | None = None

# Cached gloo process group containing all ranks across all nodes.
# Created lazily by get_global_process_group().
_global_process_group: ProcessGroup | None = None


def get_world_size() -> int:
    """
    Get the total number of processes in the distributed group.

    Returns the WORLD_SIZE environment variable, or 1 if not set (single process).
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_world_size() -> int:
    """
    Get the number of processes on the local node.

    Returns the LOCAL_WORLD_SIZE environment variable, or 1 if not set.
    This is typically the number of GPUs per node or the number of processes
    launched per node by torchrun.
    """
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def get_rank() -> int:
    """
    Get the global rank of the current process.

    Returns the RANK environment variable, or 0 if not set.
    Global rank is unique across all nodes (0 to world_size-1).
    """
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """
    Get the local rank of the current process within its node.

    Returns the LOCAL_RANK environment variable, or 0 if not set.
    Local rank is unique within a node (0 to local_world_size-1).
    Used for device assignment (e.g., which GPU to use on this node).
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_local_process_group() -> ProcessGroup | None:
    """
    Get or create a gloo process group containing only ranks on this node.

    This creates one process group per node, where each group contains only
    the ranks running on that node. Uses gloo backend for CPU compatibility.

    The groups are created lazily on first call and cached. Because dist.new_group()
    is a collective operation, ALL ranks across ALL nodes must participate when
    the groups are first created, even though each rank only joins its local group.

    This is useful for node-local coordination, such as ensuring only one process
    per node performs dataset preprocessing while others wait.

    Returns:
        The local (per-node) gloo process group, or None if:
        - torch.distributed is not available
        - torch.distributed is not initialized
        - local_world_size is 1 (single process per node, no group needed)

    Note:
        Assumes ranks are assigned sequentially per node by the launcher
        (e.g., node0: ranks 0-3, node1: ranks 4-7). This is the default
        behavior of torchrun.
    """
    global _local_process_group

    if (
        not dist.is_available()
        or not dist.is_initialized()
        or get_local_world_size() == 1
    ):
        return None

    if _local_process_group is not None:
        return _local_process_group

    # Create local process groups (one per node)
    world_size = get_world_size()
    rank = get_rank()
    local_world_size = get_local_world_size()

    # Group ranks by node - assumes ranks are assigned sequentially per node
    # e.g., node0: ranks 0-3, node1: ranks 4-7, etc.
    num_nodes = world_size // local_world_size

    # IMPORTANT: dist.new_group() is collective - ALL ranks must participate
    # in creating ALL groups, even if they're not members of that group.
    for node_id in range(num_nodes):
        # Ranks for this node
        node_ranks = list(
            range(node_id * local_world_size, (node_id + 1) * local_world_size)
        )
        group = dist.new_group(
            ranks=node_ranks, backend="gloo", group_desc="local-gloo"
        )

        # Cache the group if this rank belongs to it
        if rank in node_ranks:
            _local_process_group = group
            # Don't break - must continue creating all groups (collective operation)

    return _local_process_group


def get_global_process_group() -> ProcessGroup | None:
    """
    Get or create a gloo process group containing all ranks.

    This creates a gloo-backend process group that spans all ranks across all nodes.
    Unlike the default process group (which may use nccl), this group uses gloo,
    making it suitable for CPU-based collective operations like barriers.

    The group is created lazily on first call and cached for subsequent calls.
    All ranks must call this function collectively when the group is first created.

    Returns:
        The global gloo process group, or None if:
        - torch.distributed is not available
        - torch.distributed is not initialized
        - world_size is 1 (single process, no group needed)
    """
    global _global_process_group

    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        logger.debug(
            f"[Rank {get_rank()}] get_global_process_group: returning None "
            f"(available={dist.is_available()}, "
            f"initialized={dist.is_initialized()}, "
            f"world_size={get_world_size()})"
        )
        return None

    if _global_process_group is not None:
        logger.debug(
            f"[Rank {get_rank()}] get_global_process_group: returning cached group"
        )
        return _global_process_group

    logger.debug(
        f"[Rank {get_rank()}] get_global_process_group: creating new gloo group"
    )
    _global_process_group = dist.new_group(backend="gloo", group_desc="global-gloo")
    logger.debug(
        f"[Rank {get_rank()}] get_global_process_group: group created successfully"
    )

    return _global_process_group


@contextmanager
def main_process_first(group: Optional[ProcessGroup] = None):
    """
    Context manager for ensuring that the main process (rank0) runs first

    By default, this is applied per-node, with local-rank-0 being the 'main_process'
    and only synchronizing with other ranks on the same node.

    Optionally, a process group can be passed in, in which case 'rank-0' of that
    process group will run first. For example, to apply this to global-rank-0, use:
        main_process_first(get_global_process_group())

    An example use-case is tokenizing a dataset, where it's not a good use of
    resources to perform this action on all processes, as the work is redundant.

    When the first process on each node completes, the tokenized dataset is
    automatically cached. Then, when the remaining processes on that node try to
    tokenize the dataset, the cached dataset is loaded instead.

    If torch.distributed is not available or not initialized, and the world-size is not
    zero, a warning will be emitted, as this can corrupt data.

    Args:
        group: The process-group to use. Default is get_local_process_group()

    ```
    @main_process_first()
    def tokenize_dataset(dataset):
        ...
    ```
    """
    # No-op on recursion or single process
    global mpf_recursion_level
    if mpf_recursion_level or get_world_size() == 1:
        yield
        return

    if not dist.is_available() or not dist.is_initialized():
        logger.warning(
            f"main_process_first() was called with world-size of {get_world_size()}, "
            "but torch.distributed was not initialized. This can potentially corrupt data!"
        )
        yield
        return

    # Get barrier function - use local process group for node-local coordination
    if group is None:
        group = get_local_process_group()

    # If no group, return the null_barrier (a noop function)
    if group is None:
        barrier = null_barrier
    else:
        barrier = get_barrier_fn(group=group)

    local_rank = torch.distributed.get_group_rank(group, get_rank())
    mpf_recursion_level += 1
    try:
        if local_rank != 0:
            barrier()
        yield
    finally:
        barrier()
        if local_rank == 0:
            barrier()
        mpf_recursion_level -= 1


class DistributedEnvInterface(Protocol):
    """
    Protocol defining the interface for distributed environment objects.

    This protocol specifies the attributes that a distributed environment
    must provide. It is used for type checking and to allow different
    implementations (DistributedEnvironment, StaticDistributedEnvironment)
    to be used interchangeably.

    Attributes:
        rank: Global rank of this process (0 to world_size-1)
        local_rank: Rank within the local node (0 to local_world_size-1)
        world_size: Total number of processes across all nodes
        local_world_size: Number of processes on this node
        master_addr: Address of the rank 0 process (for rendezvous)
        master_port: Port for distributed communication setup
        device: Device string for this rank (e.g., "cuda:0", "cpu")
        device_type: Device type string (e.g., "cuda", "cpu")
    """

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    master_addr: str
    master_port: int
    device: str
    device_type: str


def init_from_env(dist: DistributedEnvInterface):
    """
    Synchronize a distributed environment object with environment variables.

    This function provides bidirectional synchronization:
    - If an environment variable is not set, it sets it from the dist object's attribute
    - If an environment variable is set, it updates the dist object's attribute from it

    This allows the distributed environment to be configured either programmatically
    (by setting attributes on the dist object) or via environment variables (as set
    by torchrun or other launchers).

    Args:
        dist: A distributed environment object implementing DistributedEnvInterface

    Note:
        See https://pytorch.org/docs/stable/elastic/run.html for the standard
        environment variables set by torchrun.
    """
    # Environment variable names and their types for proper conversion
    ENVIRON_VARS = (
        ("LOCAL_RANK", "int"),
        ("RANK", "int"),
        ("WORLD_SIZE", "int"),
        ("LOCAL_WORLD_SIZE", "int"),
        ("MASTER_ADDR", "str"),
        ("MASTER_PORT", "int"),
    )

    for var_name, value_type in ENVIRON_VARS:
        if var_name not in os.environ:
            # Environment not set - export from dist object
            os.environ[var_name] = str(getattr(dist, var_name.lower()))
        else:
            # Environment is set - import into dist object
            value = os.environ[var_name]
            match value_type:
                case "int":
                    value = int(value)
                case _:
                    pass
            setattr(dist, var_name.lower(), value)


def null_barrier(*args, **kwargs) -> None | Work:
    """
    A no-op barrier function for single-process execution.

    This is returned by get_barrier_fn() when world_size is 1, allowing code
    to call barrier() without checking whether distributed is active.

    Returns:
        Always returns None (no Work object since no operation is performed)
    """
    return None


def get_barrier_fn(group: Optional[ProcessGroup]) -> Callable[[], None | Work]:
    """
    Get a barrier function appropriate for the current distributed configuration.

    This factory function returns a callable that performs a distributed barrier
    when called. It handles the complexity of different backends:

    - For gloo backend: No device_ids argument needed (CPU-based)
    - For nccl/other GPU backends: Requires device_ids to specify the GPU
    - For single-process: Returns a no-op function

    This is the recommended way to perform barriers in forgather, as it works
    correctly with both CPU (gloo) and GPU (nccl) distributed training.

    Args:
        group: Process group for the barrier. If None, uses the default group
               (all ranks). Pass a specific group to synchronize a subset of
               ranks, such as the local node group from get_local_process_group().

    Returns:
        A callable that performs a barrier when invoked. The callable accepts
        no arguments and returns None or a Work object.

    Example:
        >>> barrier = get_barrier_fn(get_local_process_group())
        >>> barrier()  # Synchronize all ranks on this node
    """
    if dist.is_available() and dist.is_initialized() and get_world_size() != 1:
        if group is None:
            group = dist.distributed_c10d._get_default_group()

        barrier_kwargs = dict(group=group)

        # Non-gloo backends (nccl, etc.) require device_ids for GPU synchronization
        if dist.get_backend(group) != "gloo":
            barrier_kwargs["device_ids"] = [torch.accelerator.current_device_index()]

        return partial(
            dist.barrier,
            **barrier_kwargs,
        )
    else:
        assert get_world_size() == 1
        return null_barrier


@dataclass(kw_only=True)
class StaticDistributedEnvironment(DistributedEnvInterface):
    """
    A static, manually configured distributed environment data object.

    Unlike DistributedEnvironment, this class does NOT initialize torch.distributed
    or perform any device setup. It simply holds the distributed environment values
    as data.

    This is useful for:
    - Testing and mocking distributed scenarios without actually running distributed
    - Reading environment variables into a structured object (via from_env())
    - Manually specifying distributed configuration values

    Defaults are set for single-process CPU execution.

    Attributes:
        rank: Global rank (default: 0)
        local_rank: Local rank within node (default: 0)
        world_size: Total number of processes (default: 1)
        local_world_size: Processes per node (default: 1)
        master_addr: Rendezvous address (default: "localhost")
        master_port: Rendezvous port (default: 29501)
        device: Device string (default: "cpu")
        device_type: Device type (default: "cpu")
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    local_world_size: int = 1
    master_addr: str = "localhost"
    master_port: int = 29501
    device: str = "cpu"
    device_type: str = "cpu"


def from_env(**kwargs):
    """
    Create a StaticDistributedEnvironment populated from environment variables.

    This is a convenience function that creates a StaticDistributedEnvironment
    and synchronizes it with the standard torch distributed environment variables
    (RANK, WORLD_SIZE, LOCAL_RANK, etc.).

    Any keyword arguments are passed to StaticDistributedEnvironment as initial
    values, which may be overridden by environment variables if they are set.

    Args:
        **kwargs: Initial values for StaticDistributedEnvironment attributes

    Returns:
        StaticDistributedEnvironment with values from environment variables

    Example:
        >>> env = from_env()
        >>> print(f"Running as rank {env.rank} of {env.world_size}")
    """
    dist = StaticDistributedEnvironment(**kwargs)
    init_from_env(dist)
    return dist


class DistributedEnvironment(DistributedEnvInterface):
    """
    Initialize and manage the PyTorch distributed training environment.

    This class handles the complete setup of distributed training, including:
    - Synchronizing with environment variables set by launchers (torchrun, etc.)
    - Setting up the appropriate device (GPU or CPU)
    - Initializing the torch.distributed process group

    The distributed environment must be initialized before any torch.distributed
    calls can be made. In forgather configurations, this is typically included
    as an early dependency to ensure proper initialization order.

    Environment Variable Behavior:
        - If environment variables are set (e.g., by torchrun), they override
          the values passed to __init__
        - If environment variables are not set, this class exports the __init__
          values to the environment for consistency

    Device Selection:
        - With GPU available and no_accelerator=False: Uses GPU with nccl backend
        - With no_accelerator=True or no GPU: Uses CPU with gloo backend
        - Device is automatically assigned based on local_rank (or device_map)

    Attributes:
        rank: Global rank of this process
        local_rank: Rank within the local node
        world_size: Total number of processes
        local_world_size: Number of processes on this node
        master_addr: Address of rank 0 for rendezvous
        master_port: Port for rendezvous
        backend: Distributed backend ("nccl", "gloo", etc.)
        device: Device string for this rank (e.g., "cuda:0", "cpu")
        device_type: Device type string (e.g., "cuda", "cpu")
        use_accelerator: Whether to use GPU acceleration

    Example:
        In a forgather YAML configuration::

            distributed_env: &distributed_env !singleton:forgather.ml.distributed:DistributedEnvironment
                backend: "nccl"

        For CPU-only testing::

            distributed_env: &distributed_env !singleton:forgather.ml.distributed:DistributedEnvironment
                no_accelerator: True
    """

    def __init__(
        self,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        local_world_size: int = 1,
        master_addr: str = "localhost",
        master_port: int = 29501,
        backend: str | None = None,
        log_level="INFO",
        device_map=None,
        always: bool = True,
        no_accelerator: bool = False,
    ):
        """
        Initialize the distributed environment.

        Args:
            rank: Global rank (default 0, typically overridden by environment)
            local_rank: Local rank within node (default 0)
            world_size: Total processes (default 1)
            local_world_size: Processes per node (default 1)
            master_addr: Rendezvous address (default "localhost")
            master_port: Rendezvous port (default 29501)
            backend: Distributed backend. If None, auto-selected based on device
                     (nccl for GPU, gloo for CPU)
            log_level: Logging level for distributed module (default "INFO")
            device_map: Optional dict mapping rank -> device index for custom
                        device assignment. If None, uses local_rank.
            always: If True, initialize distributed even for single process.
                    Useful for consistent behavior across configurations.
            no_accelerator: If True, force CPU execution even if GPU is available.
                           Useful for testing distributed configurations.
        """
        logger.setLevel(log_level)
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.local_world_size = local_world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = backend
        self.always = always
        self.device_map = device_map
        self.use_accelerator = not no_accelerator
        self._init_distributed()

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"rank={self.rank}, "
            f"local_rank={self.local_rank}, "
            f"world_size={self.world_size}, "
            f"local_world_size={self.local_world_size}, "
            f"master_addr={self.master_addr}, "
            f"master_port={self.master_port}, "
            f"backend={self.backend})"
        )

    def _init_distributed(self):
        """
        Initialize distributed training: sync env vars, setup device, init process group.

        This method is called automatically during __init__ and performs:
        1. Bidirectional sync with environment variables
        2. Device selection and configuration
        3. Process group initialization (if not already done)
        """
        init_from_env(self)
        logger.info(str(self))

        # Device setup: use accelerator if available and not disabled
        if self.use_accelerator and accelerator.is_available():
            if self.device_map:
                accelerator.set_device_index(self.device_map[self.rank])
            else:
                accelerator.set_device_index(self.local_rank)
            acc = accelerator.current_accelerator()
            if self.backend is None:
                self.backend = dist.get_default_backend_for_device(acc)
            idx = accelerator.current_device_index()
            self.device_type = acc.type
            self.device = f"{self.device_type}:{idx}"
        else:
            # CPU fallback
            self.device = "cpu"
            self.device_type = "cpu"

        # Process group initialization
        if dist.is_available() and (self.world_size > 1 or self.always):
            if not dist.is_initialized():
                self._init_process_group()
            else:
                logger.warning("torch distributed has already been initialized")
        else:
            assert (
                self.world_size == 1
            ), "World size is larger than 1 and torch distributed is not available."

    def _init_process_group(self):
        """Initialize the torch.distributed process group with the configured backend."""
        logger.info(f"RANK{self.rank}: init_process_group({self.backend, self.device})")
        dist.init_process_group(backend=self.backend)
