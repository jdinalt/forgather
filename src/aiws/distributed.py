from contextlib import contextmanager
import os
from torch.distributed import barrier


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
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        yield
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    try:
        if local_rank != 0:
            barrier()
        yield
    finally:
        barrier()
        if local_rank == 0:
            barrier()
