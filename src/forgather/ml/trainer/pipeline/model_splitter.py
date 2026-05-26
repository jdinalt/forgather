"""
Model splitter type definition for pipeline parallel training.

A ModelSplitter is a callable that splits a model into pipeline stages.
It can be implemented as either a function or a class with __call__.
"""

from typing import Callable, List, Optional, Tuple, TypeAlias

import torch
from torch.distributed.pipelining.stage import _PipelineStageBase
from torch.nn import Module

# Type alias for the model splitter callable signature
#
# A splitter takes a model on the meta device and returns:
# 1. all_pipeline_modules: All stage modules across all ranks (on meta device)
# 2. pipeline_modules: Stage modules for the current rank (on meta device)
# 3. pipeline_stages: PipelineStage objects for the current rank
# 4. attention_mask_creator: Optional function to create attention masks externally
#
# The splitter should NOT materialize modules - that's handled by the trainer.
#
# Historical note: earlier revisions threaded an ``(example_args, example_kwargs)``
# tuple through this signature for a torch.export()-based splitter that has
# since been removed. The current manual splitter doesn't trace; the
# data-shape example was dead weight (and a real cost under DiLoCo work-unit
# dispatch, where pulling one batch just to read input_ids.shape would burn
# one work unit per pipeline rank for nothing). Dropped.
ModelSplitter: TypeAlias = Callable[
    [
        Module,  # model: Model on meta device
        List[Tuple[int, ...]],  # stage_indices: Stage indices per rank
        bool,  # train: Whether in train mode
        torch.device,  # device: Target device for this rank
        int,  # rank: Current rank
        "torch.distributed.ProcessGroup",  # pp_group: Pipeline parallel process group
    ],
    Tuple[
        List[Module],  # all_pipeline_modules (meta device)
        List[Module],  # pipeline_modules (meta device)
        List[_PipelineStageBase],  # pipeline_stages
        Optional[Callable],  # attention_mask_creator or None
    ],
]
