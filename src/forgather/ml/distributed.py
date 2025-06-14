from contextlib import contextmanager
import os
import sys
import torch
from torch import distributed
from loguru import logger

# Should we use thread-local storage for this?
# It seems unlikely, as distributed.barrier() probably does not
# play nice with threads. TBD
mpf_recursion_level = 0


@contextmanager
def main_process_first():
    """
    Context manager for ensuring that the main process runs first

    When running with multiple torch-distributed processes, this context manager
    will execute the context on the main process first.

    An example use-case is tokenizing a dataset, where it's not a good use of
    resources to perform this action on all processes, as the work is redundant.

    When the first process completets, the tokenized dataset is automatically cached,
    thens, when the remaining processes try to tokenized the dataset, the cached
    dataset is loaded instead.

    ```
    @main_process_first()
    def tokenize_dataset(dataset):
        ...
    ```
    """
    global mpf_recursion_level
    if mpf_recursion_level or int(os.environ.get("WORLD_SIZE", "1")) == 1:
        yield
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    mpf_recursion_level += 1
    try:
        if local_rank != 0:
            distributed.barrier()
        yield
    finally:
        distributed.barrier()
        if local_rank == 0:
            distributed.barrier()
        mpf_recursion_level -= 1


class DistributedEnvironment:
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

    # Envrionment variables names and types
    ENVIRON_VARS = (
        ("LOCAL_RANK", "int"),
        ("RANK", "int"),
        ("WORLD_SIZE", "int"),
        ("LOCAL_WORLD_SIZE", "int"),
        ("MASTER_ADDR", "str"),
        ("MASTER_PORT", "int"),
    )

    def __init__(
        self,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        local_world_size: int = 1,
        master_addr: str = "localhost",
        master_port: int = 29501,
        backend: str = None,
        log_level="INFO",
        device_map = None,
        always: bool = False,
    ):
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
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
        # See: https://pytorch.org/docs/stable/elastic/run.html
        # Is the distributed environment defined?
        for var_name, value_type in self.ENVIRON_VARS:
            if var_name not in os.environ:
                os.environ[var_name] = str(getattr(self, var_name.lower()))
            else:
                value = os.environ[var_name]
                match value_type:
                    case "int":
                        value = int(value)
                    case _:
                        pass
                setattr(self, var_name.lower(), value)

        logger.info(str(self))

        if torch.cuda.is_available():
            if self.device_map:
                self.device = self.device_map[self.rank]
            else:
                self.device = f"cuda:{self.local_rank}"
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
        self._init_cuda()

    def _init_cuda(self):
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

    def _init_process_group(self):
        distributed.init_process_group(backend=self.backend, device_id=torch.device(self.device))
