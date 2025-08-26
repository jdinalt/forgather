#!/usr/bin/env python3
"""
Unit tests for checkpoint functionality in Forgather trainers.

Tests the new optimizer/scheduler state saving and loading, automatic checkpoint
discovery, and checkpoint validation functionality.
"""

import unittest
import tempfile
import shutil
import os
import time
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import transformers

from forgather.ml.trainer.trainer_types import TrainingArguments, TrainerState
from forgather.ml.trainer.base_trainer import BaseTrainer
from forgather.ml.trainer.trainer import Trainer
from forgather.ml.sharded_checkpoint import (
    validate_checkpoint,
    find_latest_checkpoint,
    maybe_delete_oldest_checkpoint,
    next_checkpoint_path,
)


class SimpleMockModel(nn.Module):
    """Simple model for testing"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class MockTrainer(BaseTrainer):
    """Mock trainer that implements abstract methods for testing"""

    def __init__(self, args, **kwargs):
        self.optimizer = None
        self.lr_scheduler = None
        super().__init__(args=args, model=SimpleMockModel(), **kwargs)

    def _post_init(self):
        self.args.device = "cpu"

    def _prepare(self, train_dataset, eval_dataset):
        # Initialize state first
        self.state = TrainerState(
            logging_steps=10,
            eval_steps=10,
            train_batch_size=8,
            max_steps=100,
            save_steps=20,
            best_metric=None,
            best_model_checkpoint=None,
        )
        
        if train_dataset is not None:
            # Mock optimizer and scheduler
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10
            )

            # Restore from checkpoint if specified (after state is initialized)
            checkpoint_path = self._resolve_checkpoint_path()
            if checkpoint_path:
                self._load_training_state(checkpoint_path)

    def _train_loop(self):
        return Mock()

    def _eval_loop(self):
        return {}


class TestCheckpointFunctionality(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoints_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Create training arguments
        self.args = TrainingArguments(
            output_dir=self.test_dir,
            save_optimizer_state=True,
            save_scheduler_state=True,
            restore_optimizer_state=True,
            restore_scheduler_state=True,
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
        mock_state_dict = {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)}
        torch.save(mock_state_dict, os.path.join(checkpoint_path, "pytorch_model.bin"), _use_new_zipfile_serialization=True)

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

        self.trainer._save_training_state(checkpoint_path)

        # Check that training_state.pt was created
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        self.assertTrue(os.path.exists(training_state_path))

        # Load and verify contents
        training_state = torch.load(training_state_path, map_location="cpu")
        self.assertIn("optimizer", training_state)
        self.assertIn("lr_scheduler", training_state)

        # Verify optimizer state contains the modified learning rate
        self.assertEqual(training_state["optimizer"]["param_groups"][0]["lr"], 0.005)

    def test_save_training_state_optimizer_only(self):
        """Test saving only optimizer state when scheduler saving is disabled."""
        self.args.save_scheduler_state = False
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        self.trainer._save_training_state(checkpoint_path)

        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        self.assertTrue(os.path.exists(training_state_path))

        training_state = torch.load(training_state_path, map_location="cpu")
        self.assertIn("optimizer", training_state)
        self.assertNotIn("lr_scheduler", training_state)

    def test_save_training_state_disabled(self):
        """Test that optimizer and scheduler state are not saved when disabled."""
        self.args.save_optimizer_state = False
        self.args.save_scheduler_state = False
        self.args.save_dataset_state = False
        self.args.save_rng_state = False
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        self.trainer._save_training_state(checkpoint_path)

        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        # With all state saving disabled, no file should be created
        self.assertFalse(os.path.exists(training_state_path))

    def test_load_training_state_optimizer_and_scheduler(self):
        """Test loading optimizer and scheduler state."""
        # Create and save initial state
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Modify and save state
        self.trainer.optimizer.param_groups[0]["lr"] = 0.005
        self.trainer.lr_scheduler.step()
        initial_scheduler_state = self.trainer.lr_scheduler.state_dict()
        self.trainer._save_training_state(checkpoint_path)

        # Reset to different state
        self.trainer.optimizer.param_groups[0]["lr"] = 0.01
        self.trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.trainer.optimizer, step_size=10
        )

        # Load saved state
        self.trainer._load_training_state(checkpoint_path)

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
        self.trainer._load_training_state(checkpoint_path)

    def test_load_training_state_selective_restore(self):
        """Test selective restoration of optimizer vs scheduler."""
        checkpoint_path = os.path.join(self.checkpoints_dir, "checkpoint-test")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save state
        self.trainer.optimizer.param_groups[0]["lr"] = 0.005
        self.trainer._save_training_state(checkpoint_path)

        # Reset and disable scheduler restoration
        self.trainer.optimizer.param_groups[0]["lr"] = 0.01
        self.args.restore_scheduler_state = False

        # Load - should only restore optimizer
        self.trainer._load_training_state(checkpoint_path)

        self.assertEqual(self.trainer.optimizer.param_groups[0]["lr"], 0.005)

    def test_save_checkpoint_cleanup_by_modification_time(self):
        """Test that checkpoint cleanup uses modification time, not step numbers."""
        # Set save_total_limit to 2 so we can test cleanup
        self.args.save_total_limit = 2
        self.trainer.state = TrainerState(
            logging_steps=10,
            eval_steps=10,
            train_batch_size=8,
            max_steps=100,
            save_steps=20,
            global_step=400,
            best_metric=None,
            best_model_checkpoint=None,
        )

        # Create several checkpoints with out-of-order step numbers
        old_checkpoint = self.create_mock_checkpoint(300, delay=0.1)  # Oldest by time
        self.create_mock_checkpoint(100, delay=0.1)  # Middle
        self.create_mock_checkpoint(200, delay=0.1)  # Newest by time

        # Mock the _save_model method to avoid the directory creation issue
        with patch.object(self.trainer, "_save_model") as mock_save:

            def mock_save_with_model_file(path):
                # Create a mock model file to make checkpoint valid
                # Save a proper torch tensor dict that can be loaded
                mock_state_dict = {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)}
                torch.save(mock_state_dict, os.path.join(path, "pytorch_model.bin"), _use_new_zipfile_serialization=True)

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
        with patch.object(self.trainer, "_save_model") as mock_save:

            def mock_save_with_model_file(path):
                # Create a mock model file to make checkpoint valid
                # Save a proper torch tensor dict that can be loaded
                mock_state_dict = {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)}
                torch.save(mock_state_dict, os.path.join(path, "pytorch_model.bin"), _use_new_zipfile_serialization=True)

            mock_save.side_effect = mock_save_with_model_file
            self.trainer.save_checkpoint()

        # Step 2: Create new trainer instance with resume enabled
        args_with_resume = TrainingArguments(
            output_dir=self.test_dir,
            save_optimizer_state=True,
            save_scheduler_state=True,
            restore_optimizer_state=True,
            restore_scheduler_state=True,
            resume_from_checkpoint=True,
        )

        new_trainer = MockTrainer(args_with_resume)
        new_trainer._prepare(train_dataset=Mock(), eval_dataset=None)

        # Step 3: Verify state was restored
        self.assertEqual(new_trainer.optimizer.param_groups[0]["lr"], 0.003)
        # Scheduler should have stepped once
        self.assertEqual(new_trainer.lr_scheduler.last_epoch, 1)


class TestTrainerIntegration(unittest.TestCase):
    """Test integration with the actual Trainer class."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.args = TrainingArguments(
            output_dir=self.test_dir,
            save_optimizer_state=True,
            save_scheduler_state=True,
            resume_from_checkpoint=True,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            save_steps=10,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("forgather.ml.trainer.base_trainer.logger")
    def test_trainer_calls_checkpoint_restoration(self, mock_logger):
        """Test that Trainer._prepare() calls checkpoint restoration."""
        # Create a checkpoint first
        checkpoints_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoints_dir, "checkpoint-50")
        os.makedirs(checkpoint_path)

        # Create mock model file
        mock_state_dict = {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)}
        torch.save(mock_state_dict, os.path.join(checkpoint_path, "pytorch_model.bin"), _use_new_zipfile_serialization=True)

        # Create training state file with proper optimizer and scheduler state dict structure
        # Create a temporary model and optimizers to get proper state dict structure
        temp_model = SimpleMockModel()
        temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.001)
        temp_scheduler = transformers.get_scheduler(
            "linear", temp_optimizer, num_warmup_steps=0, num_training_steps=100
        )
        optimizer_state_dict = temp_optimizer.state_dict()
        scheduler_state_dict = temp_scheduler.state_dict()
        
        training_state = {
            "optimizer": optimizer_state_dict,
            "lr_scheduler": scheduler_state_dict,
        }
        torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"))

        # Create trainer
        model = SimpleMockModel()
        trainer = Trainer(
            model=model,
            args=self.args,
            optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.005),
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


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
