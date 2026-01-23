#!/usr/bin/env python3
"""
Distributed pipeline parallel test using PyTorch's PipelineStage and torchrun.

This script tests whether the model can work with PyTorch's actual pipeline
infrastructure, particularly testing if attention_mask (which could be a BlockMask)
can be transported between stages.

Usage:
    torchrun --nnodes 1 --nproc_per_node 2 test_distributed_pipeline.py
    torchrun --nnodes 1 --nproc_per_node 4 test_distributed_pipeline.py
"""
import argparse
import copy
import logging
import os
import sys
from argparse import RawTextHelpFormatter
from contextlib import ExitStack
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import ScheduleGPipe
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from forgather import from_project
from forgather.ml.construct import torch_dtype
from forgather.ml.data_collator import DataCollatorForCausalLM
from forgather.ml.loss import CausalLoss
from forgather.ml.sharded_checkpoint import (
    create_sharing_metadata,
    load_checkpoint,
    retie_parameters,
)
from forgather.ml.trainer.pipeline.pipeline_split_utils import (
    generate_llm_fqn_per_model_part,
    split_model,
)
from forgather.ml.utils import default_dtype


def init_distributed():
    """Initialize distributed process group."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize process group
    dist.init_process_group(backend="nccl")

    # Set device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    return rank, local_rank, world_size, device


def load_model_and_config(
    model_path: str,
    attn_implementation: str,
):
    """Load config, model_ctor, and tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model_ctor = partial(
        AutoModelForCausalLM.from_config,
        config,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )

    return model_ctor, config, tokenizer


def load_dataset(
    dataset_project,
    dataset_config,
    tokenizer,
    batch_size,
    sequence_length,
):
    """Load dataset with configurable batch size."""
    dataset = from_project(
        project_dir=dataset_project,
        config_template=dataset_config,
        targets="train_dataset",
        preprocess_args={},
        tokenizer=tokenizer,
    )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        return_tensors="pt",
        truncation=True,
        max_length=sequence_length,
        padding="max_length",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return dataloader


def create_pipeline_stage(
    model_ctor,
    stage_index,
    num_stages,
    num_layers,
    checkpoint_path,
    device,
    pp_group,
    dtype=None,
    gradient_checkpointing=False,
):
    """
    Create a single pipeline stage using manual splitting.

    This follows the pattern from test_manual_pipeline_split.py:
    1. Construct on meta device
    2. Get sharing metadata
    3. Deep copy
    4. Split
    5. Materialize
    6. Retie parameters
    7. Load checkpoint
    """
    # Construct on meta device
    with ExitStack() as exit_stack:
        exit_stack.enter_context(torch.device("meta"))
        if args.dtype:
            exit_stack.enter_context(default_dtype(dtype))
        meta_model = model_ctor()

    # Get sharing metadata
    sharing_metadata = create_sharing_metadata(meta_model)

    # Generate module distribution
    module_names_per_stage = generate_llm_fqn_per_model_part(num_stages, num_layers)
    module_names = module_names_per_stage[stage_index]

    # Deep copy and split
    stage_model = copy.deepcopy(meta_model)
    split_model(stage_model, module_names)

    # Materialize
    stage_model.to_empty(device=device)

    # Retie parameters
    retie_parameters(stage_model, sharing_metadata)

    # Load checkpoint
    load_checkpoint(checkpoint_path, stage_model, device=device, strict=False)

    # Keep in train mode for gradient computation
    stage_model.train()

    if gradient_checkpointing:
        stage_model.gradient_checkpointing_enable()

    print(f"Model for stage index {stage_index}\n{stage_model}")

    # Create PipelineStage
    pipeline_stage = PipelineStage(
        stage_model, stage_index, num_stages, device, group=pp_group
    )

    return pipeline_stage, module_names


@record
def main(args):
    """Main distributed pipeline test."""
    # Initialize distributed
    rank, local_rank, world_size, device = init_distributed()

    # Create process group for pipeline parallel
    pp_group = dist.new_group()

    # Configuration
    num_stages = world_size

    if rank == 0:
        print("=" * 80)
        print(f"Distributed Pipeline Parallel Test")
        print(f"World size: {world_size}")
        print(f"Num stages: {num_stages}")
        print("=" * 80)

    # Load model and data
    model_ctor, config, tokenizer = load_model_and_config(
        model_path=args.model_path, attn_implementation=args.attn_implementation
    )
    num_layers = config.num_hidden_layers

    if rank == 0:
        print(f"Model: {num_layers} layers, {config.hidden_size} hidden size")

    # Create pipeline stage for this rank
    if rank == 0:
        print(f"\nRank {rank}: Creating pipeline stage {rank}...")

    dtype = torch_dtype(args.dtype) if args.dtype is not None else None

    stage, module_names = create_pipeline_stage(
        model_ctor=model_ctor,
        stage_index=rank,  # Each rank gets one stage
        num_stages=num_stages,
        num_layers=num_layers,
        checkpoint_path=args.checkpoint_path,
        device=device,
        pp_group=pp_group,
        dtype=dtype,
    )

    if rank == 0:
        print(f"Rank {rank}: Stage modules: {module_names}")

    # All ranks need to load the dataset; it uses a barrier, which will
    # otherwise hang a multi-rank process.
    # batch_size must be >= num_microbatches (which equals world_size)
    dataloader = load_dataset(
        dataset_project=args.dataset_project,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )

    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    if rank == 0:
        print(f"\nTest batch shape: {input_ids.shape}")

    # Create pipeline schedule
    # num_microbatches must be >= num_stages for GPipe
    num_microbatches = args.num_microbatches
    schedule = ScheduleGPipe(
        stage,
        n_microbatches=num_microbatches,
        loss_fn=CausalLoss(),
    )

    if rank == 0:
        print(f"\nRunning pipeline with {num_microbatches} microbatches...")

    # Compute attention mask on ALL stages (following TorchTitan pattern)
    # This is passed as extra_kwargs to all stages but NOT forwarded through pipeline
    # All stages need input_ids to compute the mask, even though only rank 0 uses it for embeddings
    if rank == 0:
        print(f"Rank {rank}: Creating attention mask from input_ids...")

    # Get the stage model (unwrap from PipelineStage)
    stage_model = stage.submod

    # Create attention mask using the model's method
    # All stages compute from the same input_ids
    attention_mask = stage_model.create_attention_mask(
        input_ids=input_ids,
    )

    if rank == 0:
        if isinstance(attention_mask, BlockMask):
            print(attention_mask)
            print(repr(attention_mask))
        elif isinstance(attention_mask, torch.Tensor):
            print(f"Attention mask shape: {attention_mask.shape}")
        else:
            print(f"Attention mask type: {type(attention_mask)}")

    # Run pipeline (following TorchTitan pattern exactly)
    try:
        # Prepare targets and losses for last stage
        targets, losses = (labels, []) if rank == num_stages - 1 else (None, None)

        if rank == 0:
            # First stage: pass input_ids plus extra_kwargs, target, losses
            print(f"Rank {rank}: Running schedule.step with input_ids...")
            schedule.step(
                input_ids,
                attention_mask=attention_mask,
                target=targets,
                losses=losses,
            )
            print(f"Rank {rank}: ✓ Step completed")
        else:
            # All other stages (middle and last): pass only extra_kwargs, target, losses
            print(f"Rank {rank}: Running schedule.step...")
            schedule.step(
                attention_mask=attention_mask,
                target=targets,
                losses=losses,
            )
            print(f"Rank {rank}: ✓ Step completed")

        # Print loss if this is the last stage
        if rank == num_stages - 1:
            loss = torch.stack(losses).mean()
            print(f"Rank {rank}: Loss = {loss.item():.6f}")

    except Exception as e:
        print(f"Rank {rank}: ✗ Error during pipeline execution:")
        print(f"Rank {rank}: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        dist.barrier()
        dist.destroy_process_group()
        return 1

    # Synchronize
    dist.barrier(device_ids=[rank])

    if rank == 0:
        print("\n" + "=" * 80)
        print("Pipeline execution completed successfully!")
        print("=" * 80)

    # Cleanup
    dist.destroy_process_group()
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Test pp model splitting",
        epilog=(
            "This script should be started with torchrun.\n"
            "  torchrun --standalone --nproc_per_node N_STAGES test_distributed_pipeline.py ARGS"
        ),
    )

    parser.add_argument(
        "-m",
        "--model-path",
        type=os.path.expanduser,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=os.path.expanduser,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "-d",
        "--dataset-project",
        type=os.path.expanduser,
        required=True,
        help="Path to dataset project",
    )
    parser.add_argument(
        "-t",
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset config name",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Construct with default torch dtype",
    )
    parser.add_argument(
        "-g",
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "-a",
        "--attn-implementation",
        help="Attention implementation: eager, sdpa, flex_attention, ...",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "-s",
        "--sequence-length",
        type=int,
        default=512,
        help="Sequence length",
    )
    parser.add_argument(
        "-n",
        "--num-microbatches",
        type=int,
        default=4,
        help="Number of microbatches",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
