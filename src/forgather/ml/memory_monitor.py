#!/usr/bin/env python3
"""
Enhanced memory monitor for diagnosing PyTorch pipeline memory leaks.
This script provides comprehensive monitoring of tensor allocations and system memory.
"""

import gc
import logging
import os
import tracemalloc
import weakref
from collections import defaultdict
from typing import Dict, List, Set

import psutil
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TensorTracker:
    """Track all tensor allocations and detect leaks"""

    def __init__(self, max_step_history: int = 100):
        """
        Initialize tensor tracker with bounded step history to prevent memory leaks.

        Args:
            max_step_history: Maximum number of steps to track tensor creation info.
                             Older step data is automatically discarded. Set to 0 to disable step tracking.
        """
        self.tensors: Set[int] = set()  # Track tensor IDs
        self.tensor_info: Dict[int, tuple] = (
            {}
        )  # tensor_id -> (shape, dtype, device, creation_stack)
        self._weak_refs: Dict[int, weakref.ref] = (
            {}
        )  # tensor_id -> weakref, prevents the weakref from being GC'd
        self.step_tensors: Dict[int, Set[int]] = defaultdict(
            set
        )  # step -> tensor_ids created in that step
        self.current_step = 0
        self.max_step_history = max_step_history

    def register_tensor(self, tensor: torch.Tensor, creation_info: str = ""):
        """Register a new tensor"""
        tensor_id = id(tensor)
        if tensor_id not in self.tensors:
            self.tensors.add(tensor_id)
            self.tensor_info[tensor_id] = (
                tuple(tensor.shape),
                tensor.dtype,
                tensor.device,
                creation_info,
            )
            self.step_tensors[self.current_step].add(tensor_id)

    def tensor_finalizer(self, tensor_id: int):
        """Called when tensor is garbage collected"""
        self.tensors.discard(tensor_id)
        self.tensor_info.pop(tensor_id, None)
        self._weak_refs.pop(tensor_id, None)

    def track_tensor(self, tensor: torch.Tensor, creation_info: str = ""):
        """Track a tensor with automatic cleanup detection"""
        self.register_tensor(tensor, creation_info)
        # Use weakref to detect when tensor is garbage collected
        tensor_id = id(tensor)
        self._weak_refs[tensor_id] = weakref.ref(
            tensor, lambda ref: self.tensor_finalizer(tensor_id)
        )

    def step(self):
        """Mark start of new step"""
        self.current_step += 1

        # MEMORY LEAK FIX: Clean up old step data to prevent unbounded growth
        if self.max_step_history > 0:
            # Remove step data that's too old
            steps_to_remove = [
                step
                for step in self.step_tensors.keys()
                if step < self.current_step - self.max_step_history
            ]
            for step in steps_to_remove:
                del self.step_tensors[step]

    def get_stats(self):
        """Get current tensor statistics"""
        device_stats: defaultdict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "memory_mb": 0.0})
        dtype_stats = defaultdict(int)
        shape_stats = defaultdict(int)

        for tensor_id in self.tensors:
            if tensor_id in self.tensor_info:
                shape, dtype, device, _ = self.tensor_info[tensor_id]
                device_str = str(device)

                device_stats[device_str]["count"] += 1
                # Estimate memory usage
                numel = 1
                for dim in shape:
                    numel *= dim
                # Get bytes per element safely
                try:
                    if dtype.is_floating_point:
                        bytes_per_element = torch.finfo(dtype).bits // 8
                    elif dtype.is_complex:
                        bytes_per_element = (
                            torch.finfo(dtype).bits // 8
                        )  # Complex uses finfo
                    else:
                        bytes_per_element = torch.iinfo(dtype).bits // 8
                except:
                    # Fallback for unknown dtypes
                    bytes_per_element = 4
                memory_mb = (numel * bytes_per_element) / (1024 * 1024)
                device_stats[device_str]["memory_mb"] += memory_mb

                dtype_stats[str(dtype)] += 1
                shape_stats[shape] += 1

        return {
            "total_tensors": len(self.tensors),
            "by_device": dict(device_stats),
            "by_dtype": dict(dtype_stats),
            "by_shape": dict(shape_stats),
            "tensors_per_step": {
                step: len(tensor_set) for step, tensor_set in self.step_tensors.items()
            },
        }


class ComprehensiveMemoryMonitor:
    """Comprehensive memory monitoring for pipeline parallel training"""

    def __init__(self, rank: int = 0, max_history_size: int = 100):
        """
        Initialize memory monitor with bounded history to prevent memory leaks.

        Args:
            rank: Process rank for distributed training
            max_history_size: Maximum number of memory snapshots to keep in history.
                             Older snapshots are automatically discarded. Set to 0 to disable history.
                             Default is 100 to prevent unbounded growth that causes memory leaks.
        """
        self.rank = rank
        self.process = psutil.Process(os.getpid())
        self.tensor_tracker = TensorTracker(max_step_history=max_history_size)
        self.step_count = 0
        self.initial_memory = None
        self.memory_history = []
        self.max_history_size = max_history_size

        # Track communication objects
        self.communication_objects = []

    def start_monitoring(self):
        """Start comprehensive memory monitoring"""
        tracemalloc.start(25)  # Deep stack traces

        memory_info = self.process.memory_info()
        self.initial_memory = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
        }

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info(f"Rank {self.rank}: Started memory monitoring")
        logger.info(
            f"Rank {self.rank}: Initial memory - RSS: {self.initial_memory['rss_mb']:.1f} MB, VMS: {self.initial_memory['vms_mb']:.1f} MB"
        )

    def log_step_memory(self, step: int, additional_info: str = ""):
        """Log memory usage for a training step"""
        self.step_count = step
        self.tensor_tracker.step()

        # System memory
        memory_info = self.process.memory_info()
        current_memory = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
        }

        if self.initial_memory is None:
            self.initial_memory = current_memory
        growth = {
            "rss_growth": current_memory["rss_mb"] - self.initial_memory["rss_mb"],
            "vms_growth": current_memory["vms_mb"] - self.initial_memory["vms_mb"],
        }

        # GPU memory
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }

        # Tensor statistics
        tensor_stats = self.tensor_tracker.get_stats()

        # Python object counts
        gc_stats = {"objects": len(gc.get_objects()), "garbage": len(gc.garbage)}

        # Store history with size limit to prevent memory leaks
        snapshot = {
            "step": step,
            "memory": current_memory,
            "growth": growth,
            "gpu": gpu_info,
            "tensors": tensor_stats,
            "gc": gc_stats,
            "additional_info": additional_info,
        }

        # MEMORY LEAK FIX: Limit history size to prevent unbounded growth
        if self.max_history_size > 0:
            self.memory_history.append(snapshot)
            # Remove oldest snapshots if we exceed the limit
            while len(self.memory_history) > self.max_history_size:
                self.memory_history.pop(0)
        # If max_history_size is 0, don't store history at all

        # Log summary
        logger.info(
            f"Rank {self.rank}: Step {step} Memory Report{' (' + additional_info + ')' if additional_info else ''}"
        )
        logger.info(
            f"  System: {current_memory['rss_mb']:.1f} MB RSS (+{growth['rss_growth']:.1f}), {current_memory['vms_mb']:.1f} MB VMS (+{growth['vms_growth']:.1f})"
        )
        if gpu_info:
            logger.info(
                f"  GPU: {gpu_info['allocated_mb']:.1f} MB allocated, {gpu_info['reserved_mb']:.1f} MB reserved"
            )
        logger.info(f"  Tensors: {tensor_stats['total_tensors']} total")
        logger.info(f"  Python Objects: {gc_stats['objects']} total")

        return snapshot

    def get_tracemalloc_top(self, limit: int = 10):
        """Get top memory allocations from tracemalloc"""
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        results = []
        for stat in top_stats[:limit]:
            results.append(
                {
                    "filename": stat.traceback.format()[-1],
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                }
            )
        return results

    def analyze_memory_growth(self):
        """Analyze memory growth patterns"""
        if len(self.memory_history) < 2:
            return None

        first = self.memory_history[0]
        last = self.memory_history[-1]

        growth_per_step = {
            "rss_per_step": (
                last["growth"]["rss_growth"] - first["growth"]["rss_growth"]
            )
            / (last["step"] - first["step"]),
            "tensor_growth": last["tensors"]["total_tensors"]
            - first["tensors"]["total_tensors"],
            "object_growth": last["gc"]["objects"] - first["gc"]["objects"],
        }

        logger.info(f"Rank {self.rank}: Memory Growth Analysis:")
        logger.info(f"  RSS growth per step: {growth_per_step['rss_per_step']:.2f} MB")
        logger.info(
            f"  Total tensor growth: {growth_per_step['tensor_growth']} tensors"
        )
        logger.info(
            f"  Total object growth: {growth_per_step['object_growth']} objects"
        )

        return growth_per_step


# Global instance for easy access
memory_monitor = None


def get_memory_monitor(rank: int = 0) -> ComprehensiveMemoryMonitor:
    """Get or create the global memory monitor"""
    global memory_monitor
    if memory_monitor is None:
        memory_monitor = ComprehensiveMemoryMonitor(rank)
    return memory_monitor
