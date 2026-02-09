#!/usr/bin/env python3
"""
Unit tests for checkpoint functionality in Forgather trainers.

Tests the new optimizer/scheduler state saving and loading, automatic checkpoint
discovery, and checkpoint validation functionality.
"""

import os
import shutil
import tempfile
import time
import unittest
from dataclasses import dataclass
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
import transformers
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import DataLoader, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from forgather.ml.distributed import (
    DistributedEnvInterface,
    StaticDistributedEnvironment,
)
from forgather.ml.sharded_checkpoint import (
    find_latest_checkpoint,
    validate_checkpoint,
)
from forgather.ml.trainer.base_trainer import BaseTrainer
from forgather.ml.trainer.checkpoint_manager import (
    CheckpointConfig,
    CheckpointManager,
    RNGState,
)
from forgather.ml.trainer.trainer import Trainer, TrainingArguments
from forgather.ml.trainer.trainer_types import TrainerState


class SimpleMockModel(nn.Module):
    """Simple model for testing"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class MockDataset(IterableDataset, Stateful):
    def __init__(self, examples: int = 10):
        self.examples = examples
        self.i = 0

    def load_state_dict(self, state_dict):
        self.i = state_dict["index"]

    def state_dict(self):
        return dict(index=self.i)

    def __len__(self):
        return self.examples

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.examples:
            raise StopIteration

        i = self.i
        self.i += 1
        return i


class NonStatefulMockDataset(IterableDataset):
    """Non-stateful dataset for testing fallback behavior"""

    def __init__(self, examples: int = 10):
        self.examples = examples

    def __len__(self):
        return self.examples

    def __iter__(self):
        return iter(range(self.examples))


class MockTrainer(BaseTrainer, Stateful):
    """Mock trainer that implements abstract methods for testing"""

    def __init__(self, args: TrainingArguments, **kwargs):
        self.optimizer = None
        self.lr_scheduler = None
        super().__init__(args=args, model=SimpleMockModel(), **kwargs)

    def _post_init(self):
        self.args.device = "cpu"

    def _prepare(self, train_dataset, eval_dataset):
        assert self.model

        # Initialize state first
        self.state = TrainerState(
            logging_steps=10,
            eval_steps=10,
            train_batch_size=8,
            max_steps=100,
            max_eval_steps=-1,
            save_steps=20,
            best_metric=None,
            best_model_checkpoint=None,
            num_train_epochs=1,
        )

        if train_dataset is not None:
            # Mock optimizer and scheduler
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10
            )

            cp_config = CheckpointConfig(
                output_dir=self.args.output_dir,
                save_total_limit=self.args.save_total_limit,
            )

            if isinstance(train_dataset, IterableDataset):
                self.train_dataloader = StatefulDataLoader(dataset=train_dataset)
            else:
                self.train_dataloader = StatefulDataLoader(dataset=MockDataset())

            self.checkpoint_manager = CheckpointManager(
                config=cp_config,
                dist=StaticDistributedEnvironment(),
                stateful_provider=self,
                model=self.model,
            )

            # Restore from checkpoint if specified (after state is initialized)
            if self.args.resume_from_checkpoint:
                checkpoint_path = self.args.resume_from_checkpoint
                if isinstance(checkpoint_path, bool):
                    checkpoint_path = None
                self.load_checkpoint(checkpoint_path)

    def _resolve_checkpoint_path(self):
        assert self.checkpoint_manager
        if self.args.resume_from_checkpoint:
            checkpoint_path = self.args.resume_from_checkpoint
            if isinstance(checkpoint_path, bool):
                checkpoint_path = None
            return self.checkpoint_manager.resolve_checkpoint_path(checkpoint_path)
        else:
            return None

    def _train_loop(self):
        return Mock()

    def _eval_loop(self):
        return {}

    def load_state_dict(self, state_dict):
        self.state.global_step = state_dict["global_step"]

    def state_dict(self):
        return {"global_step": self.state.global_step}


class TestCheckpointFunctionality(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoints_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Create training arguments
        self.args = TrainingArguments(
            output_dir=self.test_dir,
            save_total_limit=3,
        )

        self.trainer = MockTrainer(self.args)
        self.trainer._prepare(train_dataset=Mock(), eval_dataset=None)

    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_mock_checkpoint(self, step: int, delay: float = 0):
        """Create a mock checkpoint directory with model files."""
        if delay > 0:
            time.sleep(delay)

        checkpoint_path = os.path.join(self.checkpoints_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Create a mock model file to make it valid
        # Save a proper torch tensor dict that can be loaded
        mock_state_dict = {
            "linear.weight": torch.randn(1, 10),
            "linear.bias": torch.randn(1),
        }
        torch.save(
            mock_state_dict,
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            _use_new_zipfile_serialization=True,
        )

        return checkpoint_path

    def test_checkpoint_validation_valid(self):
        """Test that valid checkpoints are correctly identified."""
        checkpoint_path = self.create_mock_checkpoint(100)

        self.assertTrue(validate_checkpoint(checkpoint_path))

    def test_checkpoint_validation_invalid_no_directory(self):
        """Test that non-existent directories are invalid."""
        fake_path = os.path.join(self.checkpoints_dir, "nonexistent")

        self.assertFalse(validate_checkpoint(fake_path))

    def test_checkpoint_validation_invalid_no_model_files(self):
        """Test that directories without model files are invalid."""
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-empty")
        os.makedirs(checkpoint_path, exist_ok=True)
        # Don't create any model files

        self.assertFalse(validate_checkpoint(checkpoint_path))

    def test_find_latest_checkpoint_by_modification_time(self):
        """Test that latest checkpoint is found by modification time, not step number."""
        # Create checkpoints with intentionally out-of-order step numbers
        # but in chronological order by creation time
        self.create_mock_checkpoint(300, delay=0.1)  # Created first
        self.create_mock_checkpoint(100, delay=0.1)  # Created second
        checkpoint_200 = self.create_mock_checkpoint(
            200, delay=0.1
        )  # Created third (latest)

        latest = find_latest_checkpoint(self.test_dir)

        # Should return checkpoint-200 as it was created last
        self.assertEqual(latest, checkpoint_200)
        self.assertTrue("checkpoint-200" in latest)

    def test_find_latest_checkpoint_no_checkpoints(self):
        """Test behavior when no checkpoints exist."""
        # Empty checkpoints directory
        latest = find_latest_checkpoint(self.test_dir)
        self.assertIsNone(latest)

    def test_find_latest_checkpoint_only_invalid_checkpoints(self):
        """Test behavior when only invalid checkpoints exist."""
        # Create checkpoint without model files
        invalid_path = os.path.join(self.checkpoints_dir, "checkpoint-invalid")
        os.makedirs(invalid_path, exist_ok=True)

        latest = find_latest_checkpoint(self.test_dir)
        self.assertIsNone(latest)

    def test_resolve_checkpoint_path_auto_discovery(self):
        """Test automatic checkpoint discovery when resume_from_checkpoint=True."""
        self.create_mock_checkpoint(100, delay=0.1)
        latest_checkpoint = self.create_mock_checkpoint(200, delay=0.1)

        self.args.resume_from_checkpoint = True

        resolved = self.trainer._resolve_checkpoint_path()
        self.assertEqual(resolved, latest_checkpoint)

    def test_resolve_checkpoint_path_explicit_path(self):
        """Test explicit checkpoint path when resume_from_checkpoint is a string."""
        checkpoint_path = self.create_mock_checkpoint(150)

        self.args.resume_from_checkpoint = checkpoint_path

        resolved = self.trainer._resolve_checkpoint_path()
        self.assertEqual(resolved, checkpoint_path)

    def test_resolve_checkpoint_path_explicit_invalid_path(self):
        """Test explicit checkpoint path that doesn't exist."""
        fake_path = os.path.join(self.checkpoints_dir, "nonexistent")

        self.args.resume_from_checkpoint = fake_path

        resolved = self.trainer._resolve_checkpoint_path()
        self.assertIsNone(resolved)

    def test_resolve_checkpoint_path_disabled(self):
        """Test when checkpoint resuming is disabled."""
        self.create_mock_checkpoint(100)

        self.args.resume_from_checkpoint = False

        resolved = self.trainer._resolve_checkpoint_path()
        self.assertIsNone(resolved)

    def test_save_training_state_optimizer_and_scheduler(self):
        """Test saving optimizer and scheduler state."""
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Modify optimizer and scheduler state
        self.trainer.optimizer.param_groups[0]["lr"] = 0.005
        self.trainer.lr_scheduler.step()  # Advance scheduler state

        self.trainer.checkpoint_manager._save_training_state(checkpoint_path)

        # Check that separate state files were created
        optimizer_state_path = os.path.join(checkpoint_path, "optimizer_state.pt")
        scheduler_state_path = os.path.join(checkpoint_path, "scheduler_state.pt")
        dataset_state_path = os.path.join(checkpoint_path, "dataset_state.pt")
        rng_state_path = os.path.join(checkpoint_path, "rng_state_rank_0.pt")

        self.assertTrue(os.path.exists(optimizer_state_path))
        self.assertTrue(os.path.exists(scheduler_state_path))
        self.assertTrue(os.path.exists(dataset_state_path))
        self.assertTrue(os.path.exists(rng_state_path))

        # Load and verify contents
        optimizer_state = torch.load(optimizer_state_path, map_location="cpu")
        scheduler_state = torch.load(scheduler_state_path, map_location="cpu")

        # Verify optimizer state contains the modified learning rate
        self.assertEqual(optimizer_state["param_groups"][0]["lr"], 0.005)

        # Verify scheduler state was saved
        self.assertIn("last_epoch", scheduler_state)

    def test_all_state_always_saved(self):
        """Test that all state is always saved to checkpoints."""
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        self.trainer.checkpoint_manager._save_training_state(checkpoint_path)

        # All components should be saved
        optimizer_state_path = os.path.join(checkpoint_path, "optimizer_state.pt")
        scheduler_state_path = os.path.join(checkpoint_path, "scheduler_state.pt")
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        rng_state_path = os.path.join(checkpoint_path, "rng_state_rank_0.pt")

        self.assertTrue(os.path.exists(optimizer_state_path))
        self.assertTrue(os.path.exists(scheduler_state_path))
        self.assertTrue(os.path.exists(trainer_state_path))
        self.assertTrue(os.path.exists(rng_state_path))

    def test_load_training_state_optimizer_and_scheduler(self):
        """Test loading optimizer and scheduler state."""
        # Create and save initial state
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Modify and save state
        self.trainer.optimizer.param_groups[0]["lr"] = 0.005
        self.trainer.lr_scheduler.step()
        initial_scheduler_state = self.trainer.lr_scheduler.state_dict()
        self.trainer.checkpoint_manager._save_training_state(checkpoint_path)

        # Reset to different state (modify in-place to keep coordinator's reference)
        self.trainer.optimizer.param_groups[0]["lr"] = 0.01
        self.trainer.lr_scheduler.last_epoch = 0

        # Load saved state
        self.trainer.checkpoint_manager._load_training_state(checkpoint_path)

        # Verify state was restored
        self.assertEqual(self.trainer.optimizer.param_groups[0]["lr"], 0.005)
        # Scheduler should have the same last_epoch as saved
        self.assertEqual(
            self.trainer.lr_scheduler.state_dict()["last_epoch"],
            initial_scheduler_state["last_epoch"],
        )

    def test_load_training_state_missing_file(self):
        """Test graceful handling when training state file doesn't exist."""
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Should not raise exception
        self.trainer.checkpoint_manager._load_training_state(checkpoint_path)

    def test_partial_checkpoint_load_missing_optimizer(self):
        """Test loading checkpoint with missing optimizer state."""
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save full checkpoint
        self.trainer.optimizer.param_groups[0]["lr"] = 0.005
        self.trainer.checkpoint_manager._save_training_state(checkpoint_path)

        # Delete optimizer state file
        optimizer_state_path = os.path.join(checkpoint_path, "optimizer_state.pt")
        os.remove(optimizer_state_path)

        # Reset optimizer
        self.trainer.optimizer.param_groups[0]["lr"] = 0.01

        # Should load successfully with warning (optimizer not restored)
        self.trainer.checkpoint_manager._load_training_state(checkpoint_path)

        # Optimizer should keep current state (not restored)
        self.assertEqual(self.trainer.optimizer.param_groups[0]["lr"], 0.01)

    def test_partial_checkpoint_load_missing_scheduler(self):
        """Test loading checkpoint with missing scheduler state."""
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save full checkpoint
        self.trainer.lr_scheduler.step()
        initial_epoch = self.trainer.lr_scheduler.state_dict()["last_epoch"]
        self.trainer.checkpoint_manager._save_training_state(checkpoint_path)

        # Delete scheduler state file
        scheduler_state_path = os.path.join(checkpoint_path, "scheduler_state.pt")
        os.remove(scheduler_state_path)

        # Reset scheduler
        self.trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.trainer.optimizer, step_size=10
        )

        # Should load successfully with warning (scheduler not restored)
        self.trainer.checkpoint_manager._load_training_state(checkpoint_path)

        # Scheduler should be fresh (not restored)
        self.assertEqual(self.trainer.lr_scheduler.state_dict()["last_epoch"], 0)

    def test_partial_checkpoint_load_missing_dataset(self):
        """Test loading checkpoint with missing dataset state."""
        # This test verifies that missing dataset state is handled gracefully
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save full checkpoint
        self.trainer.checkpoint_manager._save_training_state(checkpoint_path)

        # Delete dataset state file if it exists
        dataset_state_path = os.path.join(checkpoint_path, "dataset_state.pt")
        if os.path.exists(dataset_state_path):
            os.remove(dataset_state_path)

        # Should load successfully with warning
        self.trainer.checkpoint_manager._load_training_state(checkpoint_path)

    def test_save_checkpoint_cleanup_by_modification_time(self):
        """Test that checkpoint cleanup uses modification time, not step numbers."""
        # Set save_total_limit to 2 so we can test cleanup
        self.args.save_total_limit = 2
        self.trainer.state = TrainerState(
            logging_steps=10,
            eval_steps=10,
            train_batch_size=8,
            max_steps=100,
            max_eval_steps=-1,
            save_steps=20,
            global_step=400,
            best_metric=None,
            best_model_checkpoint=None,
            num_train_epochs=1,
        )

        # Create several checkpoints with out-of-order step numbers
        old_checkpoint = self.create_mock_checkpoint(300, delay=0.1)  # Oldest by time
        self.create_mock_checkpoint(100, delay=0.1)  # Middle
        self.create_mock_checkpoint(200, delay=0.1)  # Newest by time

        # Mock the _save_model method to avoid the directory creation issue
        with patch.object(self.trainer.checkpoint_manager, "_save_model") as mock_save:

            def mock_save_with_model_file(path):
                # Create a mock model file to make checkpoint valid
                # Save a proper torch tensor dict that can be loaded
                mock_state_dict = {
                    "linear.weight": torch.randn(1, 10),
                    "linear.bias": torch.randn(1),
                }
                torch.save(
                    mock_state_dict,
                    os.path.join(path, "pytorch_model.bin"),
                    _use_new_zipfile_serialization=True,
                )

            mock_save.side_effect = mock_save_with_model_file
            # This should trigger cleanup (limit is 3, we have 3, adding 1 more)
            self.trainer.save_checkpoint()

        # The checkpoint with earliest modification time should be deleted
        self.assertFalse(os.path.exists(old_checkpoint))

    def test_integration_checkpoint_save_and_restore_cycle(self):
        """Integration test: save checkpoint, create new trainer, restore state."""
        # Step 1: Train and save checkpoint
        self.trainer.state.global_step = 50
        self.trainer.optimizer.param_groups[0]["lr"] = 0.003
        self.trainer.lr_scheduler.step()

        # Mock the _save_model method to avoid directory issues, but still save training state
        with patch.object(self.trainer.checkpoint_manager, "_save_model") as mock_save:

            def mock_save_with_model_file(path):
                # Create a mock model file to make checkpoint valid
                mock_state_dict = {
                    "linear.weight": torch.randn(1, 10),
                    "linear.bias": torch.randn(1),
                }
                torch.save(
                    mock_state_dict,
                    os.path.join(path, "pytorch_model.bin"),
                    _use_new_zipfile_serialization=True,
                )

            mock_save.side_effect = mock_save_with_model_file
            self.trainer.save_checkpoint()

        # Step 2: Create new trainer instance with resume enabled
        args_with_resume = TrainingArguments(
            output_dir=self.test_dir,
            resume_from_checkpoint=True,
        )

        new_trainer = MockTrainer(args_with_resume)
        new_trainer._prepare(train_dataset=Mock(), eval_dataset=None)

        # Step 3: Verify state was restored
        self.assertEqual(new_trainer.optimizer.param_groups[0]["lr"], 0.003)
        # Scheduler should have stepped once
        self.assertEqual(new_trainer.lr_scheduler.last_epoch, 1)


class TestDoubleCheckpointDeletion(unittest.TestCase):
    """Test that checkpoint deletion only happens once (in CheckpointManager, not CheckpointCoordinator)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

        self.args = TrainingArguments(
            output_dir=self.test_dir,
            save_total_limit=4,
        )

        self.trainer = MockTrainer(self.args)
        self.trainer._prepare(train_dataset=Mock(), eval_dataset=None)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_mock_checkpoint(self, step: int, delay: float = 0):
        """Create a mock checkpoint directory with model files."""
        if delay > 0:
            time.sleep(delay)

        checkpoints_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_path, exist_ok=True)

        mock_state_dict = {
            "linear.weight": torch.randn(1, 10),
            "linear.bias": torch.randn(1),
        }
        torch.save(
            mock_state_dict,
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            _use_new_zipfile_serialization=True,
        )
        return checkpoint_path

    def test_save_total_limit_respected_with_best_checkpoints(self):
        """Test that save_total_limit and best checkpoint preservation work correctly.

        Regression test for double-deletion bug where CheckpointCoordinator
        deleted checkpoints with default save_total_limit=2 and no preservation,
        before CheckpointManager could apply the correct limits.
        """
        self.trainer.state = TrainerState(
            logging_steps=10,
            eval_steps=10,
            train_batch_size=8,
            max_steps=1000,
            max_eval_steps=-1,
            save_steps=20,
            global_step=600,
            best_metric=None,
            best_model_checkpoint=None,
            num_train_epochs=1,
        )

        # Simulate saving 5 checkpoints with decreasing loss (lower = better).
        # Checkpoints 100 and 200 have the best (lowest) loss values.
        steps_and_losses = [
            (100, 0.10),  # Best
            (200, 0.15),  # Second best
            (300, 0.50),
            (400, 0.40),
            (500, 0.30),
        ]

        with patch.object(self.trainer.checkpoint_manager, "_save_model") as mock_save:

            def mock_save_with_model_file(path):
                mock_state_dict = {
                    "linear.weight": torch.randn(1, 10),
                    "linear.bias": torch.randn(1),
                }
                torch.save(
                    mock_state_dict,
                    os.path.join(path, "pytorch_model.bin"),
                    _use_new_zipfile_serialization=True,
                )

            mock_save.side_effect = mock_save_with_model_file

            for step, loss in steps_and_losses:
                self.trainer.state.global_step = step
                checkpoint_path = self.trainer.checkpoint_manager.save_checkpoint(
                    checkpoint_id=str(step),
                )

                # Update best checkpoints (replicating real trainer flow)
                self.trainer.checkpoint_manager.update_best_checkpoints(
                    checkpoint_path=checkpoint_path,
                    metrics={"loss": loss},
                    metric_key="loss",
                    greater_is_better=False,
                    preserve_n_best=2,
                )

                time.sleep(0.05)  # Ensure distinct modification times

        # Verify: exactly 4 checkpoints remain (save_total_limit=4)
        checkpoints_dir = os.path.join(self.test_dir, "checkpoints")
        remaining = sorted(os.listdir(checkpoints_dir))
        self.assertEqual(
            len(remaining),
            4,
            f"Expected 4 checkpoints (save_total_limit=4), got {len(remaining)}: {remaining}",
        )

        # Verify: both best checkpoints still exist
        best_paths = [cp[0] for cp in self.trainer.checkpoint_manager.best_checkpoints]
        for best_path in best_paths:
            self.assertTrue(
                os.path.exists(best_path),
                f"Best checkpoint was deleted: {best_path}",
            )

        # Verify: the most recent checkpoint exists
        latest_checkpoint = os.path.join(checkpoints_dir, "checkpoint-500")
        self.assertTrue(
            os.path.exists(latest_checkpoint),
            f"Most recent checkpoint was deleted: {latest_checkpoint}",
        )


class TestTrainerIntegration(unittest.TestCase):
    """Test integration with the actual Trainer class."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.args = TrainingArguments(
            output_dir=self.test_dir,
            resume_from_checkpoint=True,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            save_steps=10,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("forgather.ml.trainer.checkpoint_manager.logger")
    def test_trainer_calls_checkpoint_restoration(self, mock_logger):
        """Test that Trainer._prepare() calls checkpoint restoration."""
        # Create a checkpoint first
        checkpoints_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoints_dir, "checkpoint-50")
        os.makedirs(checkpoint_path)

        # Create mock model file
        mock_state_dict = {
            "linear.weight": torch.randn(1, 10),
            "linear.bias": torch.randn(1),
        }
        torch.save(
            mock_state_dict,
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            _use_new_zipfile_serialization=True,
        )

        # Create training state file with proper optimizer and scheduler state dict structure
        # Create a temporary model and optimizers to get proper state dict structure
        temp_model = SimpleMockModel()
        temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.001)
        temp_scheduler = transformers.get_scheduler(
            "linear", temp_optimizer, num_warmup_steps=0, num_training_steps=100
        )

        # Save separate state files
        torch.save(
            temp_optimizer.state_dict(),
            os.path.join(checkpoint_path, "optimizer_state.pt"),
        )
        torch.save(
            temp_scheduler.state_dict(),
            os.path.join(checkpoint_path, "scheduler_state.pt"),
        )
        # Note: dataset_state.pt intentionally omitted - a fake state dict
        # would cause StatefulDataLoader to crash during iterator creation.
        # The coordinator handles missing optional components gracefully.
        torch.save(
            {"torch_rng_state": torch.get_rng_state()},
            os.path.join(checkpoint_path, "rng_state_rank_0.pt"),
        )

        # Create trainer
        model = SimpleMockModel()
        trainer = Trainer(
            model=model,
            args=self.args,
            optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.005),
            distributed_env=StaticDistributedEnvironment(),
        )

        # Mock train dataset to trigger prepare
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)

        # This should trigger checkpoint restoration
        trainer._prepare(train_dataset=mock_dataset, eval_dataset=None)

        # Verify that restoration was attempted (check log calls)
        mock_logger.info.assert_any_call(
            "Resuming training from checkpoint: " + checkpoint_path
        )


class TestDataloaderStateHandling(unittest.TestCase):
    """Test dataloader state saving and restoration with different dataloader types."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoints_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Create training arguments
        self.args = TrainingArguments(
            output_dir=self.test_dir,
            save_total_limit=3,
        )

    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_stateful_dataloader_save_and_restore(self):
        """Test that StatefulDataLoader state is correctly saved and restored."""
        # Create trainer with StatefulDataLoader
        trainer = MockTrainer(self.args)

        # Create a stateful dataset
        dataset = MockDataset(examples=20)

        # Prepare trainer - _prepare creates StatefulDataLoader with the provided dataset,
        # which is properly referenced by the coordinator's state components
        trainer._prepare(train_dataset=dataset, eval_dataset=None)

        # Advance the dataset state by creating an iterator and consuming items
        dataloader_iter = iter(trainer.train_dataloader)
        next(dataloader_iter)  # Consume first item
        next(dataloader_iter)  # Consume second item

        # The dataset index should be at 2
        self.assertEqual(dataset.i, 2)

        # Save state
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-stateful")
        os.makedirs(checkpoint_path, exist_ok=True)
        components = trainer.get_state_components()

        # Should include dataset in components
        component_keys = [c.key for c in components]
        self.assertIn("dataset", component_keys)
        dataset_component = next(c for c in components if c.key == "dataset")
        self.assertEqual(dataset_component.stateful, trainer.train_dataloader)

        # Save the checkpoint
        trainer.checkpoint_manager._save_training_state(checkpoint_path)

        # Verify dataset state file was created
        dataset_state_path = os.path.join(checkpoint_path, "dataset_state.pt")
        self.assertTrue(os.path.exists(dataset_state_path))

        # Load the saved state and verify contents
        saved_state = torch.load(dataset_state_path, map_location="cpu")
        # StatefulDataLoader stores dataset state in fetcher_state.dataset_iter_state
        self.assertEqual(saved_state["fetcher_state"]["dataset_iter_state"]["index"], 2)

        # Create new trainer and dataset for restoration test
        new_trainer = MockTrainer(self.args)
        new_dataset = MockDataset(examples=20)
        new_trainer._prepare(train_dataset=new_dataset, eval_dataset=None)

        # Verify new dataset starts at 0
        self.assertEqual(new_dataset.i, 0)

        # Restore state
        new_trainer.checkpoint_manager._load_training_state(checkpoint_path)

        # Verify dataloader state was restored (the StatefulDataLoader tracks its own state)
        restored_state = new_trainer.train_dataloader.state_dict()
        self.assertEqual(
            restored_state["fetcher_state"]["dataset_iter_state"]["index"], 2
        )
        self.assertEqual(restored_state["_num_yielded"], 2)

        # Test that iteration continues from correct position
        # When we iterate again, it should pick up from where we left off
        remaining_items = []
        dataloader_iter = iter(new_trainer.train_dataloader)
        for i, item in enumerate(dataloader_iter):
            remaining_items.append(item.item())  # Convert tensor to int
            if i >= 2:  # Get next 3 items (2, 3, 4)
                break

        # Should continue from index 2
        expected_items = [2, 3, 4]
        self.assertEqual(remaining_items, expected_items)

    def test_non_stateful_dataloader_graceful_handling(self):
        """Test that non-stateful dataloaders are handled gracefully."""
        # Create trainer with regular DataLoader (non-stateful)
        trainer = MockTrainer(self.args)

        # Create a non-stateful dataset and wrap in regular DataLoader
        dataset = NonStatefulMockDataset(examples=20)
        # Set after _prepare to avoid it being overridden
        trainer._prepare(train_dataset=dataset, eval_dataset=None)
        trainer.train_dataloader = DataLoader(dataset=dataset, batch_size=1)

        # Dataset component should not be included if dataloader is not stateful
        components = trainer.get_state_components()
        component_keys = [c.key for c in components]

        # Should not include dataset in components
        self.assertNotIn("dataset", component_keys)
        # Trainer should still be included
        self.assertIn("trainer", component_keys)

    def test_dataset_always_saved_if_stateful(self):
        """Test that dataset state is always saved if dataloader is stateful."""
        trainer = MockTrainer(self.args)
        dataset = MockDataset(examples=20)
        trainer.train_dataloader = StatefulDataLoader(dataset=dataset, batch_size=1)

        trainer._prepare(train_dataset=dataset, eval_dataset=None)
        components = trainer.get_state_components()

        # Should include dataset component if dataloader is stateful
        component_keys = [c.key for c in components]
        self.assertIn("dataset", component_keys)
        self.assertIn("trainer", component_keys)

    def test_dataset_component_missing_if_not_stateful(self):
        """Test that dataset component is not included if dataloader is not stateful."""
        # Create custom dataloader without state_dict method
        class NonStatefulDataLoader:
            pass

        trainer = MockTrainer(self.args)
        trainer._prepare(train_dataset=MockDataset(), eval_dataset=None)
        # Set after _prepare to avoid it being overridden
        trainer.train_dataloader = NonStatefulDataLoader()

        components = trainer.get_state_components()
        component_keys = [c.key for c in components]

        # Should not include dataset component
        self.assertNotIn("dataset", component_keys)
        # But trainer should still be included
        self.assertIn("trainer", component_keys)

    def test_stateful_dataloader_state_consistency(self):
        """Test that StatefulDataLoader state properly tracks dataset iteration."""
        dataset = MockDataset(examples=5)
        dataloader = StatefulDataLoader(dataset=dataset, batch_size=1)

        # Initial state
        initial_state = dataloader.state_dict()

        # Consume some items
        items = []
        for i, item in enumerate(dataloader):
            items.append(item)
            if i >= 2:  # Stop after 3 items
                break

        # Get state after consuming items
        mid_state = dataloader.state_dict()

        # State should be different
        self.assertNotEqual(initial_state, mid_state)

        # Create new dataloader and restore state
        new_dataset = MockDataset(examples=5)
        new_dataloader = StatefulDataLoader(dataset=new_dataset, batch_size=1)
        new_dataloader.load_state_dict(mid_state)

        # Continue iteration - should pick up where we left off
        remaining_items = list(new_dataloader)

        # Should get the remaining items (3, 4)
        expected_remaining = [3, 4]
        self.assertEqual(remaining_items, expected_remaining)

        # Total items should match original dataset
        all_items = items + remaining_items
        self.assertEqual(all_items, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
