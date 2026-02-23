"""
Torch profiler callback for diagnosing training performance issues.

Profiles a configurable range of training steps and exports Chrome traces.

Usage in config:
    profiler: !singleton:forgather.ml.trainer.callbacks:ProfilerCallback
        start_step: 3
        num_steps: 5
        output_dir: "benchmarks/profiles"

Or pass via callback list in training script.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


class ProfilerCallback:
    """Profiles training steps and exports Chrome traces + summary tables."""

    def __init__(
        self,
        start_step: int = 3,
        num_steps: int = 5,
        output_dir: str = "benchmarks/profiles",
        with_stack: bool = True,
        with_flops: bool = True,
        record_shapes: bool = True,
    ):
        self.start_step = start_step
        self.end_step = start_step + num_steps
        self.output_dir = output_dir
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.record_shapes = record_shapes
        self._profiler = None
        self._rank = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._rank = getattr(args, "local_rank", 0) or 0
        os.makedirs(self.output_dir, exist_ok=True)
        if self._rank == 0:
            logger.info(
                f"ProfilerCallback: will profile steps {self.start_step}-{self.end_step - 1}, "
                f"output to {self.output_dir}"
            )

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step
        if step == self.start_step:
            if self._rank == 0:
                logger.info(f"ProfilerCallback: starting profiler at step {step}")
            self._profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=self.record_shapes,
                with_stack=self.with_stack,
                with_flops=self.with_flops,
            )
            self._profiler.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if self._profiler is not None and step >= self.end_step:
            self._profiler.__exit__(None, None, None)
            prof = self._profiler
            self._profiler = None

            trace_path = os.path.join(
                self.output_dir,
                f"real_training_rank{self._rank}.json",
            )
            prof.export_chrome_trace(trace_path)

            if self._rank == 0:
                logger.info(f"ProfilerCallback: trace saved to {trace_path}")
                print("\n" + "=" * 80)
                print(f"CPU time summary (steps {self.start_step}-{self.end_step - 1}):")
                print("=" * 80)
                print(prof.key_averages().table(
                    sort_by="cpu_time_total",
                    row_limit=30,
                ))
                print("\n" + "=" * 80)
                print(f"CUDA time summary (steps {self.start_step}-{self.end_step - 1}):")
                print("=" * 80)
                print(prof.key_averages().table(
                    sort_by="cuda_time_total",
                    row_limit=30,
                ))
