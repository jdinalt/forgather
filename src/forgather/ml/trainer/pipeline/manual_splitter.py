"""
Manual model splitter for CasualLM-based models using layer deletion.

This splitter creates pipeline stages by:
1. Deep copying the model (on meta device)
2. Deleting layers that don't belong to each stage
3. Creating PipelineStage objects with the split modules

This approach works for models with deletable layers (like CasualLM)
and supports external attention mask creation.
"""

from typing import List, Tuple, Optional, Callable
import copy
import logging
import torch
from torch.nn import Module
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.stage import _PipelineStageBase

from .pipeline_split_utils import (
    generate_llm_fqn_per_model_part,
    split_model as split_model_fn,
)
from .model_splitter import ModelSplitter

logger = logging.getLogger(__name__)


def create_manual_causal_lm_splitter(
    num_layers: Optional[int] = None,
    input_weight: int = 1,
    output_weight: int = 1,
) -> ModelSplitter:
    """
    Factory for manual CasualLM splitter using layer deletion.

    This splitter is designed for models following the CasualLM architecture
    (input_encoder, layer_stack with layers, output_decoder). It splits the
    model by deleting layers that don't belong to each stage.

    Args:
        num_layers: Number of transformer layers. If None, auto-detected from
                   model.config.num_hidden_layers at split time.
        input_weight: Computational weight for input_encoder when distributing
                     layers across stages. Higher weight means input_encoder
                     is counted as equivalent to more layers.
        output_weight: Computational weight for output modules (layer_norm +
                      output_decoder) when distributing layers.

    Returns:
        A ModelSplitter function that performs manual splitting

    Example:
        >>> # Auto-detect layers, equal weights
        >>> splitter = create_manual_causal_lm_splitter()
        >>>
        >>> # Fixed 32 layers, heavier weight on input/output
        >>> splitter = create_manual_causal_lm_splitter(
        ...     num_layers=32,
        ...     input_weight=2,
        ...     output_weight=2
        ... )
    """

    def manual_splitter(
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
        Split model using manual layer deletion.

        Returns modules on meta device - caller is responsible for materialization.
        """
        # Auto-detect num_layers if not provided
        actual_num_layers = num_layers
        if actual_num_layers is None:
            if not hasattr(model, "config") or not hasattr(
                model.config, "num_hidden_layers"
            ):
                raise ValueError(
                    "Cannot auto-detect num_layers: model.config.num_hidden_layers not found. "
                    "Please provide num_layers explicitly to create_manual_causal_lm_splitter()"
                )
            actual_num_layers = model.config.num_hidden_layers
            if rank == 0:
                logger.info(
                    f"Auto-detected {actual_num_layers} layers from model config"
                )

        # Calculate total number of stages
        num_stages = sum(len(indices) for indices in stage_indices)

        # Generate module distribution using pipeline_split_utils
        # This determines which modules (input_encoder, layers, output_decoder) go to each stage
        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_stages=num_stages,
            num_layers=actual_num_layers,
            input_weight=input_weight,
            output_weight=output_weight,
        )

        if rank == 0:
            logger.debug("Module distribution per stage:")
            for stage_idx, module_names in enumerate(module_names_per_stage):
                logger.debug(f"  Stage {stage_idx}: {module_names}")

        # Create all stage models via deep copy + split
        # IMPORTANT: Keep on meta device - materialization handled by trainer
        all_pipeline_modules = []
        for stage_idx in range(num_stages):
            # Deep copy the meta model
            stage_model = copy.deepcopy(model)

            # Delete modules not belonging to this stage
            split_model_fn(stage_model, module_names_per_stage[stage_idx])

            all_pipeline_modules.append(stage_model)

        # Get modules for this rank (still on meta device)
        pipeline_modules = [all_pipeline_modules[i] for i in stage_indices[rank]]

        # Create PipelineStage objects
        # These should accept meta modules - materialization happens later
        pipeline_stages = [
            PipelineStage(
                submodule=all_pipeline_modules[i],
                stage_index=i,
                num_stages=num_stages,
                device=device,
                group=pp_group,
            )
            for i in stage_indices[rank]
        ]

        # Get attention mask creator from model
        # This allows external mask creation to avoid pipeline transport issues
        attention_mask_creator = _get_mask_creator(model, rank)

        return (
            all_pipeline_modules,
            pipeline_modules,
            pipeline_stages,
            attention_mask_creator,
        )

    return manual_splitter


def _get_mask_creator(model: Module, rank: int) -> Optional[Callable]:
    """
    Extract create_attention_mask method from model if available.

    For models that support external attention mask creation (e.g., those with
    create_attention_mask method), returns that method. This allows attention
    masks to be computed independently on each pipeline stage rather than being
    forwarded through the pipeline (which causes gradient issues).

    Args:
        model: The model to extract the mask creator from
        rank: Current rank (for logging)

    Returns:
        The create_attention_mask method if found, None otherwise
    """
    # Try direct access
    if hasattr(model, "get_attn_mask_fn"):
        if rank == 0:
            logger.info(
                "Using external attention mask creation (model.get_attn_mask_fn())"
            )
        return model.get_attn_mask_fn()

    # No external mask creator found - model will handle masks internally
    if rank == 0:
        logger.info("No external attention mask creator found")
    return None
