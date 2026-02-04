#!/usr/bin/env python
"""
Integration test for checkpoint preservation and divergence detection.

Tests the complete checkpoint management flow with synthetic metrics,
including DDP coordination, without requiring actual model training.

Run with: torchrun --nproc_per_node=2 tests/test_checkpoint_integration.py
Or: python tests/test_checkpoint_integration.py  (single process)
"""
import argparse
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forgather.ml.trainer.base_trainer import BaseTrainingArguments
from forgather.ml.trainer.callbacks.divergence_detector import (
    DualTimeScaleDivergenceDetector,
)
from forgather.ml.trainer.checkpoint_manager import CheckpointManager
from forgather.ml.trainer.trainer_types import TrainerControl, TrainerState
from forgather.ml.sharded_checkpoint import next_checkpoint_path


def find_all_checkpoints(output_dir: str) -> list[str]:
    """Find all checkpoint directories."""
    import glob
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        return []
    return sorted(glob.glob(os.path.join(checkpoints_dir, "checkpoint-*")))

logger = logging.getLogger(__name__)


@dataclass
class SimulatedMetrics:
    """Synthetic training metrics for testing."""

    step: int
    eval_loss: float
    train_loss: float
    is_spike: bool = False


class MockModel(nn.Module):
    """Minimal model for checkpoint testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


class CheckpointTestHarness:
    """
    Test harness for checkpoint preservation and divergence detection.

    Simulates training without actually training a model, allowing us to
    inject synthetic metrics and verify checkpoint management behavior.
    """

    def __init__(
        self,
        output_dir: str,
        preserve_best_model: bool = True,
        preserve_n_best: int = 2,
        save_total_limit: int = 3,
        inject_spike_at_step: int | None = None,
        use_divergence_detector: bool = True,
        num_steps: int = 100,
        eval_interval: int = 25,
        save_interval: int = 25,
    ):
        """
        Initialize test harness.

        Args:
            output_dir: Directory for checkpoints
            preserve_best_model: Enable best checkpoint preservation
            preserve_n_best: Number of best checkpoints to keep
            save_total_limit: Max recent checkpoints (excluding best)
            inject_spike_at_step: Step to inject loss spike (None = no spike)
            use_divergence_detector: Enable divergence detection
            num_steps: Total training steps to simulate
            eval_interval: Eval every N steps
            save_interval: Save every N steps
        """
        self.output_dir = output_dir
        self.inject_spike_at_step = inject_spike_at_step
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        # Setup DDP if available
        self.setup_distributed()

        # Setup logging with rank
        logging.basicConfig(
            level=logging.INFO,
            format=f"[rank{self.rank}] %(levelname)s:%(name)s:%(message)s",
            force=True,
        )

        # Create training arguments
        self.args = BaseTrainingArguments(
            output_dir=output_dir,
            preserve_best_model=preserve_best_model,
            preserve_n_best=preserve_n_best,
            best_model_metric="loss",
            best_model_greater_is_better=False,
            save_total_limit=save_total_limit,
            eval_on_save=True,
            save_strategy="steps",
            save_steps=save_interval,
            eval_strategy="steps",
            eval_steps=eval_interval,
        )

        # Create trainer state
        self.state = TrainerState(
            max_steps=num_steps,
            logging_steps=10,
            eval_steps=eval_interval,
            save_steps=save_interval,
            num_train_epochs=1,
            train_batch_size=8,
            is_local_process_zero=self.rank == 0,
            is_world_process_zero=self.rank == 0,
            max_eval_steps=-1,
        )

        # Create minimal checkpoint manager mock for testing
        # We only need the best_checkpoints tracking, not full checkpoint functionality
        class MockCheckpointManager:
            def __init__(self):
                self.best_checkpoints = []
                self.preserve_n_best = preserve_n_best

            def update_best_checkpoints(
                self,
                checkpoint_path: str,
                metrics: dict[str, float],
                metric_key: str,
                greater_is_better: bool | None,
                preserve_n_best: int,
                is_world_process_zero: bool = True,
            ) -> bool:
                """Simplified version of update_best_checkpoints for testing."""
                metric_value = metrics.get(metric_key) or metrics.get(f"eval_{metric_key}")
                if metric_value is None:
                    return False

                if greater_is_better is None:
                    greater_is_better = metric_key not in ["loss", "eval_loss"]

                is_best = False
                if len(self.best_checkpoints) < preserve_n_best:
                    is_best = True
                else:
                    worst_best = (max if greater_is_better else min)(
                        self.best_checkpoints, key=lambda x: x[1]
                    )
                    is_best = (metric_value > worst_best[1]) if greater_is_better else (metric_value < worst_best[1])

                if is_best:
                    if is_world_process_zero:
                        logger.info(f"New best checkpoint: {checkpoint_path} ({metric_key}={metric_value:.4f})")
                    self.best_checkpoints.append((checkpoint_path, metric_value))
                    self.best_checkpoints.sort(key=lambda x: x[1], reverse=greater_is_better)
                    self.best_checkpoints = self.best_checkpoints[:preserve_n_best]
                    if is_world_process_zero:
                        logger.info("Best checkpoints:")
                        for cp_path, cp_metric in self.best_checkpoints:
                            logger.info(f"  {os.path.basename(cp_path)} ({metric_key}={cp_metric:.4f})")
                return is_best

            def get_best_checkpoints_summary(self, metric_key: str = "loss") -> str:
                if not self.best_checkpoints:
                    return "No best checkpoints tracked"
                lines = [f"Best checkpoints (N={len(self.best_checkpoints)}):"]
                for cp_path, cp_metric in self.best_checkpoints:
                    lines.append(f"  {os.path.basename(cp_path)}: {metric_key}={cp_metric:.4f}")
                return "\n".join(lines)

        self.checkpoint_manager = MockCheckpointManager()

        # Create control
        self.control = TrainerControl()

        # Create divergence detector callback
        self.divergence_detector = None
        if use_divergence_detector:
            self.divergence_detector = DualTimeScaleDivergenceDetector(
                short_alpha=0.2,  # Faster response for testing
                long_alpha=0.01,
                threshold=0.75,  # Lower threshold for testing
                action="abort",
                use_eval_loss=True,
            )

        # Create dummy model for checkpoint state
        self.model = MockModel()

        # Track results
        self.checkpoints_saved = []
        self.best_checkpoints_at_each_save = []
        self.divergence_triggered = False

    def setup_distributed(self):
        """Initialize DDP if running with torchrun."""
        if "RANK" in os.environ:
            dist.init_process_group(backend="gloo")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.is_distributed = True
        else:
            self.rank = 0
            self.world_size = 1
            self.is_distributed = False

        # Add rank to logger
        logging.LoggerAdapter(logger, {"rank": self.rank})

    def generate_metrics(self, step: int) -> SimulatedMetrics:
        """
        Generate synthetic metrics for a given step.

        Simulates normal training with gradual loss decrease, then injects
        a spike if requested.
        """
        # Normal decreasing loss (exponential decay)
        base_loss = 5.0 * (0.98 ** step) + 2.5

        # Inject spike if requested
        is_spike = False
        if self.inject_spike_at_step and step >= self.inject_spike_at_step:
            # Spike loss dramatically (base_loss ~3.6 → 8.5), similar to real scenario
            # Add small increment to make it clearly divergent
            spike_loss = 8.5 + (step - self.inject_spike_at_step) * 0.05
            eval_loss = spike_loss
            train_loss = spike_loss + 0.1 * torch.randn(1).item()
            is_spike = True
        else:
            eval_loss = base_loss
            train_loss = base_loss + 0.1 * torch.randn(1).item()

        return SimulatedMetrics(
            step=step,
            eval_loss=eval_loss,
            train_loss=train_loss,
            is_spike=is_spike,
        )

    def save_checkpoint(self, step: int, metrics: dict[str, float]):
        """
        Save checkpoint with the new flow.

        This mimics the refactored trainer flow:
        1. Update best checkpoints list FIRST
        2. Then save checkpoint (deletion uses updated list)
        """
        checkpoint_path = next_checkpoint_path(self.output_dir, str(step))

        # Update best checkpoints BEFORE saving
        if self.args.preserve_best_model:
            self.checkpoint_manager.update_best_checkpoints(
                checkpoint_path=checkpoint_path,
                metrics=metrics,
                metric_key=self.args.best_model_metric,
                greater_is_better=self.args.best_model_greater_is_better,
                preserve_n_best=self.args.preserve_n_best,
                is_world_process_zero=self.state.is_world_process_zero,
            )

        # Save checkpoint (only rank 0)
        if self.rank == 0:
            # Create checkpoint directory
            os.makedirs(checkpoint_path, exist_ok=True)

            # Save minimal checkpoint (just model state for testing)
            model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            torch.save(self.model.state_dict(), model_path)

            # Save metrics
            metrics_path = os.path.join(checkpoint_path, "metrics.json")
            import json
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            if self.state.is_world_process_zero:
                logger.info(f"Saved checkpoint at step {step}")

        # Barrier to ensure all ranks see the checkpoint
        if self.is_distributed:
            dist.barrier()

        # Delete old checkpoints (only rank 0)
        if self.rank == 0 and self.args.save_total_limit > 0:
            from forgather.ml.sharded_checkpoint import maybe_delete_oldest_checkpoint

            # Build preserved list
            preserved_paths = [cp[0] for cp in self.checkpoint_manager.best_checkpoints]

            maybe_delete_oldest_checkpoint(
                self.output_dir,
                self.args.save_total_limit,
                preserved_checkpoints=preserved_paths,
            )

        # Barrier after deletion
        if self.is_distributed:
            dist.barrier()

        # Track what was saved
        self.checkpoints_saved.append(checkpoint_path)
        self.best_checkpoints_at_each_save.append(
            list(self.checkpoint_manager.best_checkpoints)
        )

    def run(self) -> bool:
        """
        Run simulated training.

        Returns:
            True if all assertions passed, False otherwise
        """
        if self.state.is_world_process_zero:
            logger.info("=" * 60)
            logger.info("Starting checkpoint integration test")
            logger.info(f"Output dir: {self.output_dir}")
            logger.info(f"World size: {self.world_size}")
            logger.info(f"Preserve best: {self.args.preserve_best_model}")
            logger.info(f"Preserve N best: {self.args.preserve_n_best}")
            logger.info(f"Save total limit: {self.args.save_total_limit}")
            logger.info(f"Inject spike at step: {self.inject_spike_at_step}")
            logger.info(f"Divergence detector: {self.divergence_detector is not None}")
            logger.info("=" * 60)

        for step in range(1, self.num_steps + 1):
            self.state.global_step = step

            # Generate synthetic metrics
            sim_metrics = self.generate_metrics(step)

            # Eval
            if step % self.eval_interval == 0:
                eval_metrics = {"eval_loss": sim_metrics.eval_loss}

                # Call divergence detector if enabled
                if self.divergence_detector:
                    self.control = self.divergence_detector.on_evaluate(
                        self.args,
                        self.state,
                        self.control,
                        metrics=eval_metrics,
                    )

                    if self.control.should_training_stop:
                        self.divergence_triggered = True
                        if self.state.is_world_process_zero:
                            logger.info(
                                f"Divergence detected at step {step}! Stopping."
                            )
                        break

                # Save checkpoint
                if step % self.save_interval == 0:
                    self.save_checkpoint(step, eval_metrics)

        # Final summary
        if self.state.is_world_process_zero:
            summary = self.checkpoint_manager.get_best_checkpoints_summary(
                metric_key=self.args.best_model_metric
            )
            logger.info("\n" + "=" * 60)
            logger.info("Simulated training complete!")
            logger.info(summary)
            logger.info("=" * 60)

        return self.verify_results()

    def verify_results(self) -> bool:
        """
        Verify checkpoint preservation worked correctly.

        Returns:
            True if all assertions passed, False otherwise
        """
        if self.rank != 0:
            return True  # Only verify on rank 0

        logger.info("\nVerifying results...")

        # Find all checkpoints on disk
        all_checkpoints = sorted(find_all_checkpoints(self.output_dir))
        logger.info(f"Checkpoints on disk: {len(all_checkpoints)}")
        for cp in all_checkpoints:
            logger.info(f"  {os.path.basename(cp)}")

        # Get best checkpoints from manager
        best_paths = [cp[0] for cp in self.checkpoint_manager.best_checkpoints]
        logger.info(f"\nBest checkpoints tracked: {len(best_paths)}")
        for cp, metric in self.checkpoint_manager.best_checkpoints:
            logger.info(f"  {os.path.basename(cp)}: loss={metric:.4f}")

        # Verify assertions
        assertions_passed = True

        # 1. Number of best checkpoints should match preserve_n_best
        if len(self.checkpoint_manager.best_checkpoints) > self.args.preserve_n_best:
            logger.error(
                f"❌ FAILED: Best checkpoints count {len(self.checkpoint_manager.best_checkpoints)} "
                f"exceeds preserve_n_best={self.args.preserve_n_best}"
            )
            assertions_passed = False
        else:
            logger.info(
                f"✓ Best checkpoints count: {len(self.checkpoint_manager.best_checkpoints)} "
                f"<= preserve_n_best={self.args.preserve_n_best}"
            )

        # 2. All best checkpoints should exist on disk
        for cp_path, _ in self.checkpoint_manager.best_checkpoints:
            if not os.path.exists(cp_path):
                logger.error(f"❌ FAILED: Best checkpoint missing: {cp_path}")
                assertions_passed = False
            else:
                logger.info(f"✓ Best checkpoint exists: {os.path.basename(cp_path)}")

        # 3. Total checkpoints should not exceed save_total_limit + preserve_n_best
        max_expected = self.args.save_total_limit + self.args.preserve_n_best
        if len(all_checkpoints) > max_expected:
            logger.error(
                f"❌ FAILED: Total checkpoints {len(all_checkpoints)} "
                f"exceeds save_total_limit + preserve_n_best = {max_expected}"
            )
            assertions_passed = False
        else:
            logger.info(
                f"✓ Total checkpoints: {len(all_checkpoints)} <= {max_expected}"
            )

        # 4. If spike was injected and detector enabled, should have triggered
        if self.inject_spike_at_step and self.divergence_detector:
            if not self.divergence_triggered:
                logger.error(
                    "❌ FAILED: Loss spike injected but divergence detector did not trigger"
                )
                assertions_passed = False
            else:
                logger.info("✓ Divergence detector triggered as expected")

        # 5. Best checkpoints should be sorted by metric
        if self.checkpoint_manager.best_checkpoints:
            metrics = [m for _, m in self.checkpoint_manager.best_checkpoints]
            is_sorted = all(
                metrics[i] <= metrics[i + 1] for i in range(len(metrics) - 1)
            )
            if not is_sorted:
                logger.error(f"❌ FAILED: Best checkpoints not sorted: {metrics}")
                assertions_passed = False
            else:
                logger.info(f"✓ Best checkpoints sorted correctly: {metrics}")

        # Summary
        if assertions_passed:
            logger.info("\n" + "=" * 60)
            logger.info("✓ ALL ASSERTIONS PASSED")
            logger.info("=" * 60)
        else:
            logger.error("\n" + "=" * 60)
            logger.error("❌ SOME ASSERTIONS FAILED")
            logger.error("=" * 60)

        return assertions_passed


def run_test_scenario(
    scenario_name: str,
    output_dir: str,
    **kwargs,
) -> bool:
    """Run a specific test scenario."""
    logger.info(f"\n{'='*60}")
    logger.info(f"SCENARIO: {scenario_name}")
    logger.info(f"{'='*60}")

    harness = CheckpointTestHarness(output_dir=output_dir, **kwargs)
    success = harness.run()

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Integration test for checkpoint preservation"
    )
    parser.add_argument(
        "--scenario",
        choices=["basic", "spike", "all"],
        default="all",
        help="Test scenario to run",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (temp dir if not specified)",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep output directory after test",
    )
    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_base = args.output_dir
        os.makedirs(output_base, exist_ok=True)
    else:
        output_base = tempfile.mkdtemp(prefix="checkpoint_test_")

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        logger.info(f"Output directory: {output_base}")

    scenarios = []
    if args.scenario in ["basic", "all"]:
        scenarios.append(
            (
                "basic",
                {
                    "preserve_best_model": True,
                    "preserve_n_best": 2,
                    "save_total_limit": 3,
                    "inject_spike_at_step": None,
                    "use_divergence_detector": False,
                    "num_steps": 100,
                    "eval_interval": 25,
                    "save_interval": 25,
                },
            )
        )

    if args.scenario in ["spike", "all"]:
        scenarios.append(
            (
                "spike",
                {
                    "preserve_best_model": True,
                    "preserve_n_best": 2,
                    "save_total_limit": 3,
                    "inject_spike_at_step": 75,  # Spike after 3 checkpoints
                    "use_divergence_detector": True,
                    "num_steps": 150,
                    "eval_interval": 25,
                    "save_interval": 25,
                },
            )
        )

    # Run scenarios
    all_passed = True
    for scenario_name, scenario_kwargs in scenarios:
        scenario_dir = os.path.join(output_base, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        passed = run_test_scenario(scenario_name, scenario_dir, **scenario_kwargs)
        all_passed = all_passed and passed

    # Cleanup
    if not args.keep_outputs and rank == 0:
        shutil.rmtree(output_base)
        logger.info(f"Cleaned up output directory: {output_base}")

    # Exit code
    if dist.is_initialized():
        dist.destroy_process_group()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
