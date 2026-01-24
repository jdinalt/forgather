#!/usr/bin/env python3
"""
Example usage of distributed checkpoint abstraction.

This example demonstrates how to use the new state-centric checkpoint API
for different parallelism scenarios.
"""

from typing import Dict, List

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.checkpoint.stateful import Stateful

# Import new checkpoint types
from forgather.ml.trainer.checkpoint_types import SharingPattern, StateComponent
from forgather.ml.trainer.checkpoint_coordinator import CheckpointCoordinator
from forgather.ml.trainer.checkpoint_manager import RNGState
from forgather.ml.distributed import StaticDistributedEnvironment


# Example 1: Simple single-GPU trainer
class SimpleTrainer:
    """Single-GPU trainer - all state is GLOBAL."""

    def __init__(self, model, optimizer, lr_scheduler, train_dataloader):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.dist = StaticDistributedEnvironment(
            world_size=1, rank=0, local_rank=0, device=torch.device("cuda:0")
        )

    def get_state_components(self) -> List[StateComponent]:
        """All state is globally shared in single-GPU setting."""
        return [
            StateComponent(
                key="model",
                stateful=self.model,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
            ),
        ]

    def get_process_groups(self) -> Dict[str, ProcessGroup]:
        return {}

    def save_checkpoint(self, output_dir: str):
        """Save checkpoint using CheckpointCoordinator."""
        coordinator = CheckpointCoordinator(
            state_components=self.get_state_components(),
            process_groups=self.get_process_groups(),
            dist=self.dist,
            output_dir=output_dir,
        )
        return coordinator.save_checkpoint(checkpoint_id="step-1000")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint using CheckpointCoordinator."""
        coordinator = CheckpointCoordinator(
            state_components=self.get_state_components(),
            process_groups=self.get_process_groups(),
            dist=self.dist,
            output_dir=checkpoint_path,
        )
        coordinator.load_checkpoint(checkpoint_path)


# Example 2: DDP trainer
class DDPTrainer:
    """DDP trainer - model/optimizer are REPLICATED."""

    def __init__(self, model, optimizer, lr_scheduler, train_dataloader, dist):
        self.model = model  # DDP-wrapped model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.dist = dist

    def unwrapped_model(self):
        """Get unwrapped model for checkpointing."""
        return self.model.module if hasattr(self.model, "module") else self.model

    def get_state_components(self) -> List[StateComponent]:
        """DDP synchronizes weights - use REPLICATED pattern."""
        return [
            StateComponent(
                key="model",
                stateful=self.unwrapped_model(),
                sharing_pattern=SharingPattern.REPLICATED,
                validate_replication=True,  # Verify DDP synchronization
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.REPLICATED,
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.REPLICATED,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=SharingPattern.GLOBAL,  # Centralized dispatch
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
            ),
        ]

    def get_process_groups(self) -> Dict[str, ProcessGroup]:
        return {}


# Example 3: Pipeline parallel trainer
class PipelineTrainer:
    """Pipeline parallel trainer - each rank has different stage."""

    def __init__(self, pipeline_modules, optimizer, lr_scheduler, train_dataloader, dist):
        self.pipeline_modules = pipeline_modules  # List of modules for this rank's stage
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.dist = dist

    def get_state_components(self) -> List[StateComponent]:
        """Each rank has different pipeline stage - use PER_RANK."""
        return [
            StateComponent(
                key="model",
                stateful=self.pipeline_modules,
                sharing_pattern=SharingPattern.PER_RANK,
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.PER_RANK,
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.REPLICATED,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
            ),
        ]

    def get_process_groups(self) -> Dict[str, ProcessGroup]:
        return {}


# Example 4: Hybrid DDP x Pipeline trainer
class HybridTrainer:
    """Hybrid DDP x Pipeline - uses PER_GROUP pattern."""

    def __init__(
        self,
        pipeline_modules,
        optimizer,
        lr_scheduler,
        train_dataloader,
        dist,
        dp_group,
        pp_group,
    ):
        self.pipeline_modules = pipeline_modules
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.dist = dist
        self.dp_group = dp_group
        self.pp_group = pp_group

    def get_state_components(self) -> List[StateComponent]:
        """Hybrid parallelism - model shard per PP group, shared within DP."""
        return [
            StateComponent(
                key="model",
                stateful=self.pipeline_modules,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="pp_group",
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="pp_group",
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.REPLICATED,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=SharingPattern.PER_GROUP,
                process_group_name="dp_group",
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
            ),
        ]

    def get_process_groups(self) -> Dict[str, ProcessGroup]:
        return {
            "dp_group": self.dp_group,
            "pp_group": self.pp_group,
        }


# Example 5: Dynamic pattern determination
class SmartTrainer:
    """Trainer that determines sharing patterns at runtime."""

    def __init__(self, model, optimizer, lr_scheduler, train_dataloader, dist):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.dist = dist

    def _get_dataset_sharing_pattern(self) -> SharingPattern:
        """Determine dataset state sharing pattern based on dataloader type."""
        from forgather.ml.trainer.dataloader_dispatcher import DataloaderDispatcher

        if isinstance(self.train_dataloader, DataloaderDispatcher):
            # Dispatcher handles coordination
            if self.train_dataloader._dp_size == 1:
                # Pure MP mode: all ranks get same batch, rank 0 loads
                return SharingPattern.GLOBAL
            elif self.train_dataloader._mp_size == 1:
                # Pure DP mode: rank 0 loads and dispatches
                return SharingPattern.GLOBAL
            else:
                # Hybrid: each DP group has one loader
                return SharingPattern.PER_GROUP
        else:
            # Each rank has independent dataloader
            return SharingPattern.PER_RANK

    def get_state_components(self) -> List[StateComponent]:
        """Components with dynamic pattern resolution."""
        return [
            StateComponent(
                key="model",
                stateful=self.model,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.GLOBAL,
            ),
            StateComponent(
                key="dataset",
                stateful=self.train_dataloader,
                sharing_pattern=self._get_dataset_sharing_pattern(),  # Dynamic!
            ),
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
            ),
        ]

    def get_process_groups(self) -> Dict[str, ProcessGroup]:
        return {}


# Example usage
def main():
    """Example of using the checkpoint API."""

    # Create a simple model
    model = nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    # Mock dataloader (in real code, this would be a proper DataLoader)
    class MockDataLoader(Stateful):
        def state_dict(self):
            return {"iteration": 0}

        def load_state_dict(self, state_dict):
            pass

    train_dataloader = MockDataLoader()

    # Create trainer
    trainer = SimpleTrainer(model, optimizer, lr_scheduler, train_dataloader)

    # Save checkpoint
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = trainer.save_checkpoint(tmpdir)
        print(f"Saved checkpoint to: {checkpoint_path}")

        # Verify manifest exists
        import os

        manifest_path = os.path.join(checkpoint_path, "checkpoint_manifest.json")
        if os.path.exists(manifest_path):
            print(f"Manifest created: {manifest_path}")

            # Load and print manifest
            from forgather.ml.trainer.checkpoint_types import CheckpointManifest

            manifest = CheckpointManifest.load(manifest_path)
            print(f"Checkpoint components: {list(manifest.components.keys())}")
            print(f"World size: {manifest.world_size}")

        # Load checkpoint
        trainer2 = SimpleTrainer(
            nn.Linear(10, 1),  # New model
            torch.optim.AdamW(model.parameters(), lr=1e-4),
            torch.optim.lr_scheduler.LinearLR(optimizer),
            MockDataLoader(),
        )
        trainer2.load_checkpoint(checkpoint_path)
        print("Successfully loaded checkpoint")


if __name__ == "__main__":
    main()
