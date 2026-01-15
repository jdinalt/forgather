import logging
import os
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Callable, Protocol

import torch
from torch import accelerator, distributed
from torch._C._distributed_c10d import Work

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tracks recursion of main_process_first
mpf_recursion_level = 0

# Cache for local process group
_local_process_group = None


def get_local_process_group():
    """
    Get or create a process group containing only ranks on this node.

    Returns None if distributed is not initialized or not available.
    """
    global _local_process_group

    if not distributed.is_available() or not distributed.is_initialized():
        return None

    if _local_process_group is not None:
        return _local_process_group

    # Create local process groups (one per node)
    world_size = distributed.get_world_size()
    rank = distributed.get_rank()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    # Group ranks by node - assumes ranks are assigned sequentially per node
    # e.g., node0: ranks 0-3, node1: ranks 4-7, etc.
    num_nodes = world_size // local_world_size

    # IMPORTANT: distributed.new_group() is collective - ALL ranks must participate
    # in creating ALL groups, even if they're not members
    for node_id in range(num_nodes):
        # Ranks for this node
        node_ranks = list(
            range(node_id * local_world_size, (node_id + 1) * local_world_size)
        )
        group = distributed.new_group(ranks=node_ranks)

        # Cache the group if this rank belongs to it
        if rank in node_ranks:
            _local_process_group = group
            # Don't break! Must continue to create all groups

    return _local_process_group


@contextmanager
def main_process_first(dist=None):
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
    if mpf_recursion_level or int(os.environ.get("WORLD_SIZE", "1")) == 1:
        yield
        return

    # Use LOCAL_RANK for per-node coordination
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Get barrier function - use local process group for node-local coordination
    local_group = get_local_process_group()
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


def init_from_env(dist: DistributedEnvInterface):
    """
    Init a distributed environment from envrionment variables
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


def get_barrier_fn(group=None) -> Callable[[], None | Work]:
    """
    torch.distributed.barrier() complains about not having specified device_ids, if
    called without this argument. It's also not always available, so this returns either
    a barrier function, bound to the current device, or a no-op lambda function.

    Args:
        group: Optional process group for the barrier. If None, uses the default group (all ranks).
               Pass a specific group to create a barrier for a subset of ranks (e.g., local node only).
    """
    if distributed.is_available() and accelerator.is_available():
        return partial(
            distributed.barrier,
            device_ids=[torch.accelerator.current_device_index()],
            group=group,
        )
    else:
        return lambda *args, **kwargs: None


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
        always: bool = False,
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

        if torch.accelerator.is_available():
            if self.device_map:
                torch.accelerator.set_device_index(self.device_map[self.rank])
            else:
                torch.accelerator.set_device_index(self.local_rank)
            acc = torch.accelerator.current_accelerator()
            if self.backend is None:
                self.backend = torch.distributed.get_default_backend_for_device(acc)
            idx = torch.accelerator.current_device_index()
            self.device = f"{acc.type}:{idx}"
        else:
            self.device = "cpu"

        if distributed.is_available() and (self.world_size > 1 or self.always):
            if not distributed.is_initialized():
                self._init_process_group()
            else:
                logger.warning("torch distributed has already been initialized")
        else:
            assert (
                self.world_size == 1
            ), "World size is larger than 1 and torch distributed is not available."
        self.barrier_fn = get_barrier_fn()

    def _init_process_group(self):
        logger.info(f"RANK{self.rank}: init_process_group({self.backend, self.device})")
        distributed.init_process_group(backend=self.backend)

    def barrier(self):
        self.barrier_fn()
