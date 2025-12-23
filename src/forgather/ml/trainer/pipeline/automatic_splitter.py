"""
Automatic model splitter using torch.export tracing.

This is the original splitting method that uses PyTorch's build_pipeline
to automatically trace the model and split at specified points.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.distributed.pipelining import SplitPoint
from torch.distributed.pipelining import pipeline as build_pipeline
from torch.distributed.pipelining.stage import _PipelineStageBase
from torch.nn import Module

from .model_splitter import ModelSplitter
from .pipeline_fixes import apply_pipeline_buffer_fix, remove_vestigial_modules

logger = logging.getLogger(__name__)


def create_automatic_splitter(split_spec: dict) -> ModelSplitter:
    """
    Factory for automatic splitter using torch.export tracing.

    This splitter uses PyTorch's build_pipeline to automatically trace
    the model execution graph and split it at specified layer boundaries.

    Args:
        split_spec: Dictionary mapping layer names to SplitPoint (BEGINNING/END)
                   Example: {"layer_stack.layers.2": SplitPoint.BEGINNING}

    Returns:
        A ModelSplitter function that performs automatic splitting

    Note:
        The automatic splitter does not currently support external attention
        mask creation, so it returns None for the mask_creator.
    """
    # Convert string split points to enums if needed
    processed_split_spec = {}
    for key, value in split_spec.items():
        if isinstance(value, str):
            match value:
                case "beginning":
                    processed_split_spec[key] = SplitPoint.BEGINNING
                case "end":
                    processed_split_spec[key] = SplitPoint.END
                case _:
                    raise ValueError(f"Unknown split-point type {value} for {key}")
        else:
            processed_split_spec[key] = value

    def automatic_splitter(
        model: Module,
        example_args: tuple,
        example_kwargs: dict,
        stage_indices: List[Tuple[int, ...]],
        train: bool,
        device: torch.device,
        rank: int,
        pp_group: "torch.distributed.ProcessGroup",
    ) -> Tuple[
        List[Module], List[Module], List[_PipelineStageBase], Optional[Callable]
    ]:
        """
        Split model using automatic tracing.

        Returns modules on meta device - caller is responsible for materialization.
        """
        # Set model mode
        if train:
            model.train()
        else:
            model.eval()

        # Build pipeline using torch's automatic tracing
        # This uses fake tensors, so no actual computation happens
        kwargs: Dict[str, Any] = dict(split_spec=processed_split_spec)

        if train:
            pipe = build_pipeline(model, example_args, example_kwargs, **kwargs)
        else:
            with torch.no_grad():
                pipe = build_pipeline(model, example_args, example_kwargs, **kwargs)

        if rank == 0:
            logger.debug(f"Pipeline split:\n{pipe}")

        # Calculate total number of stages
        num_stages = sum(len(indices) for indices in stage_indices)

        # Get all stage modules (still on meta device)
        all_pipeline_modules = [pipe.get_stage_module(i) for i in range(num_stages)]

        # Get modules for this rank
        pipeline_modules = [all_pipeline_modules[i] for i in stage_indices[rank]]

        # Build pipeline stages
        # Note: PipelineStage should accept meta modules
        pipeline_stages = [
            pipe.build_stage(stage_index=i, device=device) for i in stage_indices[rank]
        ]

        # Apply fixes after pipeline split but before materialization
        # This fixes zombie buffers and shared buffer accessibility
        apply_pipeline_buffer_fix(all_pipeline_modules, model)

        # Remove vestigial modules to prevent duplicate FQN conflicts
        remove_vestigial_modules(all_pipeline_modules)

        # Automatic splitting doesn't support external attention masks yet
        # The model computes masks internally during forward pass
        attention_mask_creator = None

        return (
            all_pipeline_modules,
            pipeline_modules,
            pipeline_stages,
            attention_mask_creator,
        )

    return automatic_splitter
