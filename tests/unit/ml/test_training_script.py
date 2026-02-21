#!/usr/bin/env python3
"""
Unit tests for forgather.ml.training_script module.

Tests the TrainingScript dataclass and its run() method.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from forgather.dotdict import DotDict
from forgather.ml.training_script import TrainingScript


class TestTrainingScript(unittest.TestCase):
    """Test the TrainingScript dataclass."""

    def _make_meta(self, **overrides):
        """Create a minimal meta dict for TrainingScript."""
        meta = {
            "config_name": "test_config",
            "config_description": "A test configuration",
            "output_dir": "/tmp/test_output",
        }
        meta.update(overrides)
        return meta

    def test_post_init_converts_meta_to_dotdict(self):
        """Test that __post_init__ converts meta dict to DotDict."""
        meta = self._make_meta()
        trainer = Mock()

        ts = TrainingScript(meta=meta, trainer=trainer)

        self.assertIsInstance(ts.meta, DotDict)
        self.assertEqual(ts.meta.config_name, "test_config")
        self.assertEqual(ts.meta.config_description, "A test configuration")
        self.assertEqual(ts.meta.output_dir, "/tmp/test_output")

    def test_default_flags(self):
        """Test default values for do_train, do_eval, do_save."""
        meta = self._make_meta()
        trainer = Mock()

        ts = TrainingScript(meta=meta, trainer=trainer)

        self.assertTrue(ts.do_train)
        self.assertFalse(ts.do_eval)
        self.assertFalse(ts.do_save)

    def test_run_do_train(self):
        """Test that run() calls trainer.train() when do_train=True."""
        meta = self._make_meta()
        trainer = Mock()
        trainer.train.return_value = "train_output"

        ts = TrainingScript(meta=meta, trainer=trainer, do_train=True, do_eval=False, do_save=False)
        ts.run()

        trainer.train.assert_called_once()
        trainer.evaluate.assert_not_called()
        trainer.save_model.assert_not_called()

    def test_run_do_eval(self):
        """Test that run() calls trainer.evaluate() when do_eval=True."""
        meta = self._make_meta()
        trainer = Mock()
        trainer.evaluate.return_value = {}

        ts = TrainingScript(meta=meta, trainer=trainer, do_train=False, do_eval=True, do_save=False)
        ts.run()

        trainer.train.assert_not_called()
        trainer.evaluate.assert_called_once()
        trainer.save_model.assert_not_called()

    def test_run_do_save(self):
        """Test that run() calls trainer.save_model() when do_save=True."""
        meta = self._make_meta()
        trainer = Mock()

        ts = TrainingScript(meta=meta, trainer=trainer, do_train=False, do_eval=False, do_save=True)
        ts.run()

        trainer.train.assert_not_called()
        trainer.evaluate.assert_not_called()
        trainer.save_model.assert_called_once_with("/tmp/test_output")

    def test_run_all_flags(self):
        """Test that run() calls all trainer methods when all flags are True."""
        meta = self._make_meta()
        trainer = Mock()
        trainer.train.return_value = "train_output"
        trainer.evaluate.return_value = {"loss": 0.5}

        ts = TrainingScript(
            meta=meta, trainer=trainer, do_train=True, do_eval=True, do_save=True
        )
        ts.run()

        trainer.train.assert_called_once()
        trainer.evaluate.assert_called_once()
        trainer.save_model.assert_called_once_with("/tmp/test_output")

    def test_run_no_flags(self):
        """Test that run() calls nothing when all flags are False."""
        meta = self._make_meta()
        trainer = Mock()

        ts = TrainingScript(meta=meta, trainer=trainer, do_train=False, do_eval=False, do_save=False)
        ts.run()

        trainer.train.assert_not_called()
        trainer.evaluate.assert_not_called()
        trainer.save_model.assert_not_called()

    def test_meta_with_logging_dir(self):
        """Test that run() handles meta with logging_dir attribute."""
        meta = self._make_meta(logging_dir="/tmp/logs")
        trainer = Mock()

        ts = TrainingScript(meta=meta, trainer=trainer, do_train=False)
        # Should not raise even though logging_dir is present
        ts.run()

    def test_meta_dot_access(self):
        """Test that meta supports dot notation access after __post_init__."""
        meta = self._make_meta(custom_key="custom_value")
        trainer = Mock()

        ts = TrainingScript(meta=meta, trainer=trainer)

        self.assertEqual(ts.meta.custom_key, "custom_value")

    def test_distributed_env_default(self):
        """Test that distributed_env has a default factory."""
        meta = self._make_meta()
        trainer = Mock()

        ts = TrainingScript(meta=meta, trainer=trainer)

        # distributed_env should be set via default_factory (from_env)
        self.assertIsNotNone(ts.distributed_env)


class TestRescaleLoss(unittest.TestCase):
    """Test the RescaleLoss wrapper class."""

    def test_rescale_applies_factor(self):
        """Test that RescaleLoss multiplies loss by scale_factor."""
        import torch
        from forgather.ml.loss import RescaleLoss

        base_loss_fn = Mock(return_value=torch.tensor(2.0))
        rescaled = RescaleLoss(base_loss_fn, scale_factor=0.5)

        result = rescaled(torch.tensor([1.0]), torch.tensor([1]))

        self.assertAlmostEqual(result.item(), 1.0)
        base_loss_fn.assert_called_once()

    def test_rescale_factor_of_one(self):
        """Test that scale_factor=1 returns unchanged loss."""
        import torch
        from forgather.ml.loss import RescaleLoss

        base_loss_fn = Mock(return_value=torch.tensor(3.0))
        rescaled = RescaleLoss(base_loss_fn, scale_factor=1.0)

        result = rescaled()

        self.assertAlmostEqual(result.item(), 3.0)

    def test_no_rescale_context_manager(self):
        """Test that no_rescale() context manager disables rescaling."""
        import torch
        from forgather.ml.loss import RescaleLoss

        base_loss_fn = Mock(return_value=torch.tensor(4.0))
        rescaled = RescaleLoss(base_loss_fn, scale_factor=0.25)

        # With rescaling
        result_rescaled = rescaled()
        self.assertAlmostEqual(result_rescaled.item(), 1.0)

        # Without rescaling
        with rescaled.no_rescale():
            result_no_rescale = rescaled()
            self.assertAlmostEqual(result_no_rescale.item(), 4.0)

        # Rescaling restored
        result_after = rescaled()
        self.assertAlmostEqual(result_after.item(), 1.0)

    def test_no_rescale_restores_on_exception(self):
        """Test that no_rescale() restores state even after exception."""
        import torch
        from forgather.ml.loss import RescaleLoss

        base_loss_fn = Mock(return_value=torch.tensor(2.0))
        rescaled = RescaleLoss(base_loss_fn, scale_factor=0.5)

        try:
            with rescaled.no_rescale():
                self.assertTrue(rescaled.skip_rescale)
                raise ValueError("test error")
        except ValueError:
            pass

        # Should be restored
        self.assertFalse(rescaled.skip_rescale)

    def test_functools_update_wrapper(self):
        """Test that RescaleLoss wraps the underlying function metadata."""
        import torch
        from forgather.ml.loss import RescaleLoss

        def my_loss_fn(logits, labels):
            """Custom loss docstring."""
            return torch.tensor(1.0)

        rescaled = RescaleLoss(my_loss_fn, scale_factor=2.0)

        # functools.update_wrapper copies __name__, __doc__, etc.
        self.assertEqual(rescaled.__name__, "my_loss_fn")
        self.assertEqual(rescaled.__doc__, "Custom loss docstring.")

    def test_rescale_with_zero_factor(self):
        """Test rescaling with factor 0 returns zero loss."""
        import torch
        from forgather.ml.loss import RescaleLoss

        base_loss_fn = Mock(return_value=torch.tensor(5.0))
        rescaled = RescaleLoss(base_loss_fn, scale_factor=0.0)

        result = rescaled()

        self.assertAlmostEqual(result.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
