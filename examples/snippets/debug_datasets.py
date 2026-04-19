import logging
import os
from functools import partial

import torch
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torchdata.stateful_dataloader import StatefulDataLoader

from forgather import Project
from forgather.ml.data_collator import DataCollatorForCausalLM
from forgather.ml.datasets import sync_dataset_state_from_dataloader
from forgather.ml.distributed import DistributedEnvironment, get_barrier_fn, get_rank
from forgather.ml.trainer.synchronized_dataloader import SynchronizedDataLoader


class RankFilter(logging.Filter):
    """Filter that adds the distributed rank to log records."""

    def filter(self, record):
        record.rank = get_rank()
        return True


def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.propagate = False
    handler = logging.StreamHandler()
    handler.addFilter(RankFilter())
    formatter = logging.Formatter(
        "[Rank %(rank)s] %(asctime)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logging()

# Get path to dataset project
script_path = os.path.dirname(os.path.abspath(__file__))

# Set paths to test projects
dataset_project_dir = os.path.join(script_path, "..", "datasets", "HuggingFaceTB")
tokenizer_project_path = os.path.join(script_path, "..", "tokenizers", "wikitext")

# Setup torch distributed
denv = DistributedEnvironment()
barrier = get_barrier_fn()

# Get tokenizer from project definition
tokenizer_project = Project(
    "8k.yaml",
    tokenizer_project_path,
)

# This configuration is composed of two datasets, "fineweb-edu-dedup" and "cosmopedia-v2"
# These two datasets are then wrapped in an InterleavedDataset, where the draw probabilities
# are computed by balance_remaining_examples(), which selects examples based upon the estimated
# remaining size of each dataset.

# Get dataset from project definition
dataset_project = Project(
    "smollm-corpus/interleaved-packed.yaml",
    dataset_project_dir,
)

# Materialize tokenizer instance
tokenizer = tokenizer_project()

# Materialize train and eval datasets
train_dataset, eval_dataset = dataset_project(
    "train_dataset",
    "eval_dataset",
    preprocess_args=dict(truncation=True, max_length=4096),
    tokenizer=tokenizer,
    shard_dataset=True,
)

# Show dataset info
logger.info(f"{train_dataset=}\n{eval_dataset=}")

# Make a data-collator
data_collator = DataCollatorForCausalLM(
    tokenizer=tokenizer,
    packed_sequences=True,
    return_tensors="pt",
)

# Create dataloader partial
make_dataloader = partial(
    StatefulDataLoader,
    collate_fn=data_collator,
    drop_last=True,
    num_workers=1,
    pin_memory=True,
    prefetch_factor=32,
    persistent_workers=True,
)

# Construct dataloaders
train_dataloader = make_dataloader(train_dataset, batch_size=16)
eval_dataloader = make_dataloader(eval_dataset, batch_size=32)

# Create device mesh, same as DDP uses
device_mesh = init_device_mesh(
    denv.device_type,
    (denv.world_size,),
    mesh_dim_names=("data_parallel",),
)

ddp_group = device_mesh.get_group(0)

# Wrap datasets in SynchronizedDataLoader
# This will prevent hanging, if dataset lengths are uneven.
wrap_dataloader = partial(
    SynchronizedDataLoader,
    device=denv.device,
    process_group=ddp_group,
)

train_dataloader = wrap_dataloader(train_dataloader)
eval_dataloader = wrap_dataloader(eval_dataloader)


def get_length_info(dataset):
    ratio = (
        dataset._output_count / dataset._input_count
        if dataset._input_count > 0
        else 0.0
    )
    return (
        f"(len={len(dataset)},"
        f"_original_length={dataset._original_length}, "
        f"_input_count={dataset._input_count}, "
        f"_output_count={dataset._output_count}, "
        f"ratio={ratio:.3f}, "
        f"_cached_exact_length={dataset._cached_exact_length})"
    )


# Get length info for all sub-datasets
def get_dataset_lengths(dataloader):
    lengths = []
    for ds in dataloader.dataset.datasets:
        lengths.append(get_length_info(ds))
    return lengths


# Define our "train" and "eval" loops
def eval_loop(eval_dataloader):
    i = -1
    batches = 0
    for i, batch in enumerate(eval_dataloader):
        batches += len(batch["input_ids"])
    i += 1
    total_batches = torch.tensor(batches, dtype=torch.int, device=denv.device)
    dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

    # Sync dataloader length with workers
    sync_dataset_state_from_dataloader(eval_dataloader)
    lengths = get_dataset_lengths(eval_dataloader)
    logger.info(
        f"Eval yielded {i} batches [total={total_batches.item()}]; Dataloader len = {len(eval_dataloader)}, {lengths}"
    )


def train_loop(train_dataloader, eval_dataloader, max_steps):
    ds_iter = iter(train_dataloader)
    try:
        for step in range(max_steps):
            batch = next(ds_iter)
            barrier()
            if step % 10 == 0:
                # Sync dataloader length with workers
                sync_dataset_state_from_dataloader(train_dataloader)
                lengths = []
                lengths = get_dataset_lengths(train_dataloader)
                logger.info(
                    f"Train Step: {step}; Dataloader length={len(train_dataloader)}, {lengths}"
                )
            if step % 50 == 0:
                eval_loop(eval_dataloader)
    except StopIteration:
        logger.info("StopIteration")


logger.info("Starting train loop")
train_loop(train_dataloader, eval_dataloader, 1000)
logger.info("Train loop complete")


# Shutdown torch distributed
barrier()
dist.destroy_process_group()
