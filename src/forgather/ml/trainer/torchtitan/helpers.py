from logging import Logger
from typing import Callable, cast

import torch
from datasets.distributed import split_dataset_by_node
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import IterableDataset
from torchtitan.components.dataloader import (
    ParallelAwareDataloader,  # type: ignore[import-untyped]  # torchtitan is an optional dependency
)
from torchtitan.components.ft import (
    FTManager,  # type: ignore[import-untyped]  # torchtitan is an optional dependency
)
from torchtitan.components.lr_scheduler import (
    LRSchedulersContainer,  # type: ignore[import-untyped]  # torchtitan is an optional dependency
)
from torchtitan.components.optimizer import (
    OptimizersContainer,  # type: ignore[import-untyped]  # torchtitan is an optional dependency
)
from torchtitan.components.tokenizer import (
    BaseTokenizer,  # type: ignore[import-untyped]  # torchtitan is an optional dependency
)
from torchtitan.config import (
    JobConfig,  # type: ignore[import-untyped]  # torchtitan is an optional dependency
)
from torchtitan.config import (
    LRScheduler as LRSchedulerConfig,  # type: ignore[import-untyped]  # torchtitan is an optional dependency
)
from torchtitan.config import (
    Optimizer as OptimizerConfig,  # type: ignore[import-untyped]  # torchtitan is an optional dependency
)
from torchtitan.distributed import (
    ParallelDims,  # type: ignore[import-untyped]  # torchtitan is an optional dependency
)

LossFunction = Callable[..., torch.Tensor]

logger = Logger(__name__)


def build_optimizers(
    model_parts: list[Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager,
    container_factory: Callable[..., OptimizersContainer],
    **kwargs,
) -> OptimizersContainer:
    return container_factory(
        model_parts=model_parts,
    )


def build_lr_schedulers(
    optimizers: OptimizersContainer,
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
    container_factory: Callable[..., LRSchedulersContainer],
    **kwargs,
) -> LRSchedulersContainer:
    return container_factory(
        optimizers=optimizers,
    )


def build_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    dataset: IterableDataset,
    batch_size: int,
    data_collator,
    infinite: bool = True,
    **kwargs,
) -> ParallelAwareDataloader:
    return ParallelAwareDataloader(
        dataset=split_dataset_by_node(cast(object, dataset), dp_rank, dp_world_size),  # type: ignore[arg-type]  # IterableDataset satisfies DatasetType at runtime
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=data_collator,
    )


def build_tokenizer(
    job_config: JobConfig,
    tokenizer: BaseTokenizer,
    **kwargs,
) -> BaseTokenizer:
    return tokenizer


def build_loss_fn(
    job_config: JobConfig, loss_fn: LossFunction, **kwargs
) -> LossFunction:
    if job_config.compile.enable and "loss" in job_config.compile.components:
        loss_fn = torch.compile(loss_fn)
    return loss_fn
