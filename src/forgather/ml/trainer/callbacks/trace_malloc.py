import logging
import os
import tracemalloc
from pprint import pformat

import psutil
import torch
import torch.distributed as dist

from ...memory_monitor import get_memory_monitor
from ..trainer_types import (
    MinimalTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TraceMalloc(TrainerCallback):
    def __init__(self, top_k=10, monitor_system_memory=True, comprehensive_mode=True):
        """
        :param top_k: Number of top memory allocations to show
        :param monitor_system_memory: Whether to monitor system memory usage
        :param comprehensive_mode: Whether to use comprehensive memory monitoring
        """
        super().__init__()
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.top_k = top_k
        self.prev_snap = None
        self.monitor_system_memory = monitor_system_memory
        self.process = psutil.Process(os.getpid()) if monitor_system_memory else None
        self.initial_memory = None
        self.comprehensive_mode = comprehensive_mode
        self.memory_monitor = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.comprehensive_mode:
            self.memory_monitor = get_memory_monitor(self.rank)
            self.memory_monitor.start_monitoring()
        else:
            tracemalloc.start(100)
            if self.monitor_system_memory and self.process:
                memory_info = self.process.memory_info()
                self.initial_memory = memory_info.rss / 1024 / 1024  # MB
                logger.info(
                    f"rank{self.rank}: Initial memory usage: {self.initial_memory:.1f} MB (RSS)"
                )

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        step = logs.get("step", 0)

        if self.comprehensive_mode and self.memory_monitor:
            # Use comprehensive memory monitoring
            self.memory_monitor.log_step_memory(step)

            # Get detailed tracemalloc info every 10 steps
            if step % 10 == 0:
                top_allocs = self.memory_monitor.get_tracemalloc_top(self.top_k)
                logger.info(f"rank{self.rank}: Top {self.top_k} memory allocations:")
                for i, alloc in enumerate(top_allocs):
                    logger.info(
                        f"rank{self.rank}:   {i+1}. {alloc['size_mb']:.2f} MB, {alloc['count']} objects - {alloc['filename']}"
                    )

            # Analyze growth every 20 steps
            if step % 20 == 0 and step > 0:
                self.memory_monitor.analyze_memory_growth()

        else:
            # Original monitoring approach
            if self.monitor_system_memory and self.process:
                try:
                    memory_info = self.process.memory_info()
                    rss_mb = memory_info.rss / 1024 / 1024  # MB
                    vms_mb = memory_info.vms / 1024 / 1024  # MB

                    if self.initial_memory:
                        memory_growth = rss_mb - self.initial_memory
                        logger.info(
                            f"rank{self.rank}: Step {step} - Memory: {rss_mb:.1f} MB RSS, {vms_mb:.1f} MB VMS, Growth: +{memory_growth:.1f} MB"
                        )
                    else:
                        logger.info(
                            f"rank{self.rank}: Step {step} - Memory: {rss_mb:.1f} MB RSS, {vms_mb:.1f} MB VMS"
                        )

                    # Get GPU memory info if available
                    if torch.cuda.is_available():
                        gpu_allocated = (
                            torch.cuda.memory_allocated() / 1024 / 1024
                        )  # MB
                        gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                        logger.info(
                            f"rank{self.rank}: GPU Memory - Allocated: {gpu_allocated:.1f} MB, Reserved: {gpu_reserved:.1f} MB"
                        )

                except Exception as e:
                    logger.warning(f"rank{self.rank}: Failed to get memory info: {e}")

            # Original tracemalloc functionality
            snap = tracemalloc.take_snapshot()
            if self.prev_snap:
                delta = snap.compare_to(self.prev_snap, "lineno")
                logger.info(
                    f"rank{self.rank}: Top {self.top_k} memory allocations since last step:"
                )
                for stat in delta[: self.top_k]:
                    logger.info(f"rank{self.rank}: {stat}")
            self.prev_snap = snap

    def on_train_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        tracemalloc.stop()
