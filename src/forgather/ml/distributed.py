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
logger.setLevel(logging.INFO)

# Tracks recursion of main_process_first
mpf_recursion_level = 0

# A "gloo" process group, with all local ranks
_local_process_group: ProcessGroup | None = None

# A "gloo" process group, with all ranks
_global_process_group: ProcessGroup | None = None


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_local_process_group() -> ProcessGroup | None:
    """
    Get or create a process group containing only ranks on this node.

    Returns None if distributed is not initialized or not available.
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
    # in creating ALL groups, even if they're not members
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
            # Don't break! Must continue to create all groups

    return _local_process_group


def get_global_process_group() -> ProcessGroup | None:
    """
    Get or create a process group containing all ranks

    Returns None if distributed is not initialized or not available.
    """
    global _global_process_group

    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        return None

    if _local_process_group is not None:
        return _global_process_group

    _global_process_group = dist.new_group(backend="gloo", group_desc="global-gloo")

    return _global_process_group


@contextmanager
def main_local_process_first():
    """
    Context manager for ensuring that the main process on each node runs first

    When running with multiple torch-distributed processes, this context manager
    will execute the context on the main process (local_rank 0) of each node first.

    An example use-case is tokenizing a dataset, where it's not a good use of
    resources to perform this action on all processes, as the work is redundant.

    When the first process on each node completes, the tokenized dataset is
    automatically cached. Then, when the remaining processes on that node try to
    tokenize the dataset, the cached dataset is loaded instead.

    In multi-node scenarios, each node's rank 0 process will run first, followed
    by other ranks on the same node. The barrier is local to each node, so nodes
    can process independently.

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

    # Use LOCAL_RANK for per-node coordination
    local_rank = get_local_rank()

    # Get barrier function - use local process group for node-local coordination
    local_group = get_local_process_group()
    if local_group is None:
        barrier = null_barrier
    else:
        barrier = get_barrier_fn(group=local_group)

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
    Init a distributed environment from environment variables
    """
    # Envrionment variables names and types
    ENVIRON_VARS = (
        ("LOCAL_RANK", "int"),
        ("RANK", "int"),
        ("WORLD_SIZE", "int"),
        ("LOCAL_WORLD_SIZE", "int"),
        ("MASTER_ADDR", "str"),
        ("MASTER_PORT", "int"),
    )

    # See: https://pytorch.org/docs/stable/elastic/run.html
    # Is the distributed environment defined?
    for var_name, value_type in ENVIRON_VARS:
        if var_name not in os.environ:
            os.environ[var_name] = str(getattr(dist, var_name.lower()))
        else:
            value = os.environ[var_name]
            match value_type:
                case "int":
                    value = int(value)
                case _:
                    pass
            setattr(dist, var_name.lower(), value)


def null_barrier(*args, **kwargs) -> None | Work:
    """A no-op barrier function, for world-size == 1"""
    return None


def get_barrier_fn(group: Optional[ProcessGroup]) -> Callable[[], None | Work]:
    """
    torch.distributed.barrier() complains about not having specified device_ids, if
    called without this argument. It's also not always available, so this returns either
    a barrier function, bound to the current device, or a no-op lambda function.

    Args:
        group: Optional process group for the barrier. If None, uses the default group (all ranks).
               Pass a specific group to create a barrier for a subset of ranks (e.g., local node only).
    """
    if dist.is_available() and dist.is_initialized() and get_world_size() != 1:
        if group is None:
            group = dist.distributed_c10d._get_default_group()

        barrier_kwargs = dict(group=group)

        # Non-gloo backends expect a device index
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
    A static, manually configured, distributed environment
    This can be useful for mocks -- or just to manually set these values from somewhere else.
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
    Construct a simple distributed environment from env vars.
    """
    dist = StaticDistributedEnvironment(**kwargs)
    init_from_env(dist)
    return dist


class DistributedEnvironment(DistributedEnvInterface):
    """
    This class initializes the distributed envrionment, if not already initialized

    The distributed environment needs to be setup before any torch distributed
    calls may be made. For example, 'main_process_first,' uses
    distributed.barrier(); this will fail, if the envrionment has not been setup.

    If the envrionment variables have already been set (e.g. launching with 'torchrun'),
    the environment will override any values specified here.

    OTOH, if an envrironment variable has not been set, this will set the env variable.

    By including this in a configuration before any other dependencies, the environment
    will be setup.

    This can also be used to provide information to consumers, although this is also
    available from the envrionment.

    It's possible to sub-class this, as to customize how the environment is initialized.

    device_map: rank: int -> device_name: str
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
        init_from_env(self)
        logger.info(str(self))

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
            self.device = "cpu"
            self.device_type = "cpu"

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
        logger.info(f"RANK{self.rank}: init_process_group({self.backend, self.device})")
        dist.init_process_group(backend=self.backend)
