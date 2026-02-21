#!/usr/bin/env python3
"""
Unit tests for the memory monitoring module.

Tests the classes and functions in forgather.ml.memory_monitor:
- TensorTracker: tensor allocation tracking, step management, statistics
- ComprehensiveMemoryMonitor: full memory monitoring with bounded history
- get_memory_monitor: global singleton accessor
"""

import gc
import unittest
from unittest.mock import patch

import torch

import forgather.ml.memory_monitor as memory_monitor_module
from forgather.ml.memory_monitor import (
    ComprehensiveMemoryMonitor,
    TensorTracker,
    get_memory_monitor,
)


# ---------------------------------------------------------------------------
# TensorTracker
# ---------------------------------------------------------------------------


class TestTensorTrackerRegister(unittest.TestCase):
    """Test TensorTracker.register_tensor."""

    def test_register_adds_to_tensors_set(self):
        """Registering a tensor adds its id to the tensors set."""
        tracker = TensorTracker()
        t = torch.zeros(2, 3)
        tracker.register_tensor(t)
        self.assertIn(id(t), tracker.tensors)

    def test_register_stores_tensor_info(self):
        """Registering a tensor stores shape, dtype, device, and creation_info."""
        tracker = TensorTracker()
        t = torch.ones(4, 5, dtype=torch.float32)
        tracker.register_tensor(t, creation_info="test_origin")
        info = tracker.tensor_info[id(t)]
        self.assertEqual(info[0], (4, 5))
        self.assertEqual(info[1], torch.float32)
        self.assertEqual(info[2], t.device)
        self.assertEqual(info[3], "test_origin")

    def test_register_adds_to_step_tensors(self):
        """Registered tensor is recorded under the current step."""
        tracker = TensorTracker()
        t = torch.zeros(1)
        tracker.register_tensor(t)
        self.assertIn(id(t), tracker.step_tensors[0])

    def test_register_duplicate_is_ignored(self):
        """Registering the same tensor twice does not duplicate entries."""
        tracker = TensorTracker()
        t = torch.zeros(3)
        tracker.register_tensor(t, creation_info="first")
        tracker.register_tensor(t, creation_info="second")
        self.assertEqual(len(tracker.tensors), 1)
        # Original info is preserved, not overwritten
        self.assertEqual(tracker.tensor_info[id(t)][3], "first")

    def test_register_default_creation_info(self):
        """Default creation_info is an empty string."""
        tracker = TensorTracker()
        t = torch.zeros(1)
        tracker.register_tensor(t)
        self.assertEqual(tracker.tensor_info[id(t)][3], "")

    def test_register_multiple_tensors(self):
        """Multiple distinct tensors are tracked independently."""
        tracker = TensorTracker()
        t1 = torch.zeros(2)
        t2 = torch.ones(3)
        t3 = torch.randn(4, 5)
        tracker.register_tensor(t1, "t1")
        tracker.register_tensor(t2, "t2")
        tracker.register_tensor(t3, "t3")
        self.assertEqual(len(tracker.tensors), 3)
        self.assertEqual(len(tracker.tensor_info), 3)

    def test_register_tensor_at_different_steps(self):
        """Tensors registered at different steps go into different step sets."""
        tracker = TensorTracker()
        t1 = torch.zeros(1)
        tracker.register_tensor(t1)
        tracker.step()
        t2 = torch.ones(1)
        tracker.register_tensor(t2)
        self.assertIn(id(t1), tracker.step_tensors[0])
        self.assertIn(id(t2), tracker.step_tensors[1])
        self.assertNotIn(id(t1), tracker.step_tensors[1])


class TestTensorTrackerFinalizer(unittest.TestCase):
    """Test TensorTracker.tensor_finalizer."""

    def test_finalizer_removes_from_tensors_set(self):
        """tensor_finalizer removes the tensor id from the tensors set."""
        tracker = TensorTracker()
        t = torch.zeros(2)
        tid = id(t)
        tracker.register_tensor(t)
        self.assertIn(tid, tracker.tensors)
        tracker.tensor_finalizer(tid)
        self.assertNotIn(tid, tracker.tensors)

    def test_finalizer_removes_tensor_info(self):
        """tensor_finalizer removes the tensor info entry."""
        tracker = TensorTracker()
        t = torch.zeros(2)
        tid = id(t)
        tracker.register_tensor(t)
        self.assertIn(tid, tracker.tensor_info)
        tracker.tensor_finalizer(tid)
        self.assertNotIn(tid, tracker.tensor_info)

    def test_finalizer_with_unknown_id(self):
        """tensor_finalizer does not raise for unknown tensor ids."""
        tracker = TensorTracker()
        # Should not raise
        tracker.tensor_finalizer(99999999)

    def test_finalizer_idempotent(self):
        """Calling tensor_finalizer twice for the same id does not raise."""
        tracker = TensorTracker()
        t = torch.zeros(1)
        tid = id(t)
        tracker.register_tensor(t)
        tracker.tensor_finalizer(tid)
        tracker.tensor_finalizer(tid)  # Second call should be safe
        self.assertNotIn(tid, tracker.tensors)


class TestTensorTrackerTrack(unittest.TestCase):
    """Test TensorTracker.track_tensor with weakref-based cleanup."""

    def test_track_registers_tensor(self):
        """track_tensor registers the tensor just like register_tensor."""
        tracker = TensorTracker()
        t = torch.zeros(3, 4)
        tracker.track_tensor(t, creation_info="tracked")
        self.assertIn(id(t), tracker.tensors)
        self.assertEqual(tracker.tensor_info[id(t)][3], "tracked")

    def test_track_cleanup_on_gc(self):
        """BUG: track_tensor's weakref callback captures `tensor` by reference,
        which keeps the tensor alive and prevents garbage collection.

        The lambda `lambda ref: self.tensor_finalizer(id(tensor))` holds a
        reference to the `tensor` local variable. This means the weakref's
        callback itself prevents the tensor from being collected. The fix
        would be to capture `tensor_id = id(tensor)` before the lambda and
        use that instead.
        """
        tracker = TensorTracker()
        t = torch.zeros(5)
        tid = id(t)
        tracker.track_tensor(t, "gc_test")
        self.assertIn(tid, tracker.tensors)

        # Delete the local reference and force garbage collection
        del t
        gc.collect()

        # BUG: tensor is NOT collected because the weakref callback lambda
        # captures `tensor` by reference, keeping it alive.
        # The tensor remains in tracking despite being "deleted" locally.
        self.assertIn(tid, tracker.tensors)


class TestTensorTrackerStep(unittest.TestCase):
    """Test TensorTracker.step and step history management."""

    def test_step_increments_counter(self):
        """step() increments the current_step counter."""
        tracker = TensorTracker()
        self.assertEqual(tracker.current_step, 0)
        tracker.step()
        self.assertEqual(tracker.current_step, 1)
        tracker.step()
        self.assertEqual(tracker.current_step, 2)

    def test_step_cleans_old_data(self):
        """step() removes step_tensors entries older than max_step_history."""
        tracker = TensorTracker(max_step_history=3)
        tensors = []
        for i in range(5):
            t = torch.zeros(1)
            tensors.append(t)  # Keep references so they are not GC'd
            tracker.register_tensor(t, f"step_{i}")
            tracker.step()

        # Steps 0 and 1 should have been cleaned up (current_step=5, keep 3..5)
        self.assertNotIn(0, tracker.step_tensors)
        self.assertNotIn(1, tracker.step_tensors)
        # Step 2 should still be there (5 - 3 = 2, and condition is strict <)
        self.assertIn(2, tracker.step_tensors)

    def test_max_step_history_zero_disables_cleanup(self):
        """When max_step_history=0, no step data is cleaned up."""
        tracker = TensorTracker(max_step_history=0)
        tensors = []
        for i in range(10):
            t = torch.zeros(1)
            tensors.append(t)
            tracker.register_tensor(t, f"step_{i}")
            tracker.step()

        # All step data should be preserved
        for i in range(10):
            self.assertIn(i, tracker.step_tensors)

    def test_step_cleanup_preserves_recent(self):
        """step() preserves step data within the max_step_history window."""
        tracker = TensorTracker(max_step_history=5)
        tensors = []
        for i in range(10):
            t = torch.zeros(1)
            tensors.append(t)
            tracker.register_tensor(t, f"step_{i}")
            tracker.step()

        # Recent steps should be preserved
        # current_step is 10, so steps >= 10 - 5 = 5 are kept
        for i in range(5, 10):
            self.assertIn(i, tracker.step_tensors)

    def test_initial_state(self):
        """Initial state has current_step=0 and empty collections."""
        tracker = TensorTracker()
        self.assertEqual(tracker.current_step, 0)
        self.assertEqual(len(tracker.tensors), 0)
        self.assertEqual(len(tracker.tensor_info), 0)

    def test_max_step_history_default(self):
        """Default max_step_history is 100."""
        tracker = TensorTracker()
        self.assertEqual(tracker.max_step_history, 100)


class TestTensorTrackerGetStats(unittest.TestCase):
    """Test TensorTracker.get_stats."""

    def test_empty_stats(self):
        """Stats for an empty tracker have zero counts."""
        tracker = TensorTracker()
        stats = tracker.get_stats()
        self.assertEqual(stats["total_tensors"], 0)
        self.assertEqual(stats["by_device"], {})
        self.assertEqual(stats["by_dtype"], {})
        self.assertEqual(stats["by_shape"], {})
        self.assertEqual(stats["tensors_per_step"], {})

    def test_stats_total_tensors(self):
        """total_tensors reflects the number of tracked tensors."""
        tracker = TensorTracker()
        t1 = torch.zeros(2, 3)
        t2 = torch.ones(4)
        tracker.register_tensor(t1)
        tracker.register_tensor(t2)
        stats = tracker.get_stats()
        self.assertEqual(stats["total_tensors"], 2)

    def test_stats_by_device(self):
        """by_device correctly groups tensors by device."""
        tracker = TensorTracker()
        t1 = torch.zeros(2, 3)  # CPU tensor
        t2 = torch.ones(4)  # CPU tensor
        tracker.register_tensor(t1)
        tracker.register_tensor(t2)
        stats = tracker.get_stats()
        cpu_key = str(t1.device)
        self.assertIn(cpu_key, stats["by_device"])
        self.assertEqual(stats["by_device"][cpu_key]["count"], 2)

    def test_stats_by_dtype(self):
        """by_dtype correctly groups tensors by dtype."""
        tracker = TensorTracker()
        t1 = torch.zeros(2, dtype=torch.float32)
        t2 = torch.zeros(3, dtype=torch.int64)
        t3 = torch.zeros(4, dtype=torch.float32)
        tracker.register_tensor(t1)
        tracker.register_tensor(t2)
        tracker.register_tensor(t3)
        stats = tracker.get_stats()
        self.assertEqual(stats["by_dtype"][str(torch.float32)], 2)
        self.assertEqual(stats["by_dtype"][str(torch.int64)], 1)

    def test_stats_by_shape(self):
        """by_shape correctly groups tensors by shape."""
        tracker = TensorTracker()
        t1 = torch.zeros(2, 3)
        t2 = torch.zeros(2, 3)
        t3 = torch.zeros(4, 5)
        tracker.register_tensor(t1)
        tracker.register_tensor(t2)
        tracker.register_tensor(t3)
        stats = tracker.get_stats()
        self.assertEqual(stats["by_shape"][(2, 3)], 2)
        self.assertEqual(stats["by_shape"][(4, 5)], 1)

    def test_stats_tensors_per_step(self):
        """tensors_per_step tracks creation counts per step."""
        tracker = TensorTracker()
        t1 = torch.zeros(1)
        t2 = torch.zeros(1)
        tracker.register_tensor(t1)
        tracker.register_tensor(t2)
        tracker.step()
        t3 = torch.zeros(1)
        tracker.register_tensor(t3)
        stats = tracker.get_stats()
        self.assertEqual(stats["tensors_per_step"][0], 2)
        self.assertEqual(stats["tensors_per_step"][1], 1)

    def test_stats_memory_estimation(self):
        """by_device includes a non-zero memory_mb for non-empty tensors."""
        tracker = TensorTracker()
        # 1000 float32 elements = 1000 * 4 bytes = ~0.0038 MB
        t = torch.zeros(1000, dtype=torch.float32)
        tracker.register_tensor(t)
        stats = tracker.get_stats()
        device_key = str(t.device)
        self.assertGreater(stats["by_device"][device_key]["memory_mb"], 0)

    def test_stats_keys_present(self):
        """get_stats returns all expected top-level keys."""
        tracker = TensorTracker()
        stats = tracker.get_stats()
        expected_keys = {"total_tensors", "by_device", "by_dtype", "by_shape", "tensors_per_step"}
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_stats_after_finalizer(self):
        """Stats reflect tensor removal after tensor_finalizer."""
        tracker = TensorTracker()
        t = torch.zeros(2, 3)
        tid = id(t)
        tracker.register_tensor(t)
        self.assertEqual(tracker.get_stats()["total_tensors"], 1)
        tracker.tensor_finalizer(tid)
        self.assertEqual(tracker.get_stats()["total_tensors"], 0)


# ---------------------------------------------------------------------------
# ComprehensiveMemoryMonitor
# ---------------------------------------------------------------------------


class TestComprehensiveMemoryMonitorInit(unittest.TestCase):
    """Test ComprehensiveMemoryMonitor initialization."""

    def test_default_init(self):
        """Default initialization sets expected values."""
        monitor = ComprehensiveMemoryMonitor()
        self.assertEqual(monitor.rank, 0)
        self.assertEqual(monitor.step_count, 0)
        self.assertIsNone(monitor.initial_memory)
        self.assertEqual(monitor.memory_history, [])
        self.assertEqual(monitor.max_history_size, 100)
        self.assertIsInstance(monitor.tensor_tracker, TensorTracker)

    def test_custom_rank(self):
        """rank parameter is stored correctly."""
        monitor = ComprehensiveMemoryMonitor(rank=3)
        self.assertEqual(monitor.rank, 3)

    def test_custom_max_history_size(self):
        """max_history_size parameter is stored correctly."""
        monitor = ComprehensiveMemoryMonitor(max_history_size=50)
        self.assertEqual(monitor.max_history_size, 50)

    def test_tensor_tracker_max_step_matches(self):
        """TensorTracker's max_step_history matches the monitor's max_history_size."""
        monitor = ComprehensiveMemoryMonitor(max_history_size=42)
        self.assertEqual(monitor.tensor_tracker.max_step_history, 42)


class TestComprehensiveMemoryMonitorStartMonitoring(unittest.TestCase):
    """Test ComprehensiveMemoryMonitor.start_monitoring."""

    def setUp(self):
        self.monitor = ComprehensiveMemoryMonitor(rank=0)

    def tearDown(self):
        # Stop tracemalloc if it was started by the test
        import tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    def test_records_initial_memory(self):
        """start_monitoring records initial RSS and VMS."""
        self.monitor.start_monitoring()
        self.assertIsNotNone(self.monitor.initial_memory)
        self.assertIn("rss_mb", self.monitor.initial_memory)
        self.assertIn("vms_mb", self.monitor.initial_memory)

    def test_initial_memory_positive(self):
        """Initial RSS and VMS are positive numbers."""
        self.monitor.start_monitoring()
        self.assertGreater(self.monitor.initial_memory["rss_mb"], 0)
        self.assertGreater(self.monitor.initial_memory["vms_mb"], 0)

    def test_starts_tracemalloc(self):
        """start_monitoring activates tracemalloc."""
        import tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        self.monitor.start_monitoring()
        self.assertTrue(tracemalloc.is_tracing())


class TestComprehensiveMemoryMonitorLogStep(unittest.TestCase):
    """Test ComprehensiveMemoryMonitor.log_step_memory."""

    def setUp(self):
        self.monitor = ComprehensiveMemoryMonitor(rank=0, max_history_size=10)
        self.monitor.start_monitoring()

    def tearDown(self):
        import tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    def test_returns_snapshot(self):
        """log_step_memory returns a dict with expected keys."""
        snapshot = self.monitor.log_step_memory(step=1)
        self.assertIsInstance(snapshot, dict)
        expected_keys = {"step", "memory", "growth", "gpu", "tensors", "gc", "additional_info"}
        self.assertEqual(set(snapshot.keys()), expected_keys)

    def test_snapshot_step_value(self):
        """Snapshot records the correct step number."""
        snapshot = self.monitor.log_step_memory(step=42)
        self.assertEqual(snapshot["step"], 42)

    def test_snapshot_memory_keys(self):
        """Snapshot memory dict has rss_mb and vms_mb."""
        snapshot = self.monitor.log_step_memory(step=1)
        self.assertIn("rss_mb", snapshot["memory"])
        self.assertIn("vms_mb", snapshot["memory"])

    def test_snapshot_growth_keys(self):
        """Snapshot growth dict has rss_growth and vms_growth."""
        snapshot = self.monitor.log_step_memory(step=1)
        self.assertIn("rss_growth", snapshot["growth"])
        self.assertIn("vms_growth", snapshot["growth"])

    def test_snapshot_gc_keys(self):
        """Snapshot gc dict has objects and garbage counts."""
        snapshot = self.monitor.log_step_memory(step=1)
        self.assertIn("objects", snapshot["gc"])
        self.assertIn("garbage", snapshot["gc"])

    def test_snapshot_additional_info(self):
        """Additional info is stored in the snapshot."""
        snapshot = self.monitor.log_step_memory(step=1, additional_info="after_forward")
        self.assertEqual(snapshot["additional_info"], "after_forward")

    def test_history_appended(self):
        """Snapshots are appended to memory_history."""
        self.assertEqual(len(self.monitor.memory_history), 0)
        self.monitor.log_step_memory(step=1)
        self.assertEqual(len(self.monitor.memory_history), 1)
        self.monitor.log_step_memory(step=2)
        self.assertEqual(len(self.monitor.memory_history), 2)

    def test_updates_step_count(self):
        """log_step_memory updates the step_count attribute."""
        self.monitor.log_step_memory(step=10)
        self.assertEqual(self.monitor.step_count, 10)

    def test_advances_tensor_tracker_step(self):
        """log_step_memory calls tensor_tracker.step()."""
        initial_step = self.monitor.tensor_tracker.current_step
        self.monitor.log_step_memory(step=1)
        self.assertEqual(self.monitor.tensor_tracker.current_step, initial_step + 1)


class TestComprehensiveMemoryMonitorHistoryBound(unittest.TestCase):
    """Test ComprehensiveMemoryMonitor max_history_size bounding."""

    def setUp(self):
        self.monitor = ComprehensiveMemoryMonitor(rank=0, max_history_size=5)
        self.monitor.start_monitoring()

    def tearDown(self):
        import tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    def test_history_bounded_by_max_size(self):
        """memory_history never exceeds max_history_size."""
        for i in range(20):
            self.monitor.log_step_memory(step=i)
        self.assertLessEqual(len(self.monitor.memory_history), 5)

    def test_oldest_entries_removed(self):
        """When history is full, the oldest entries are removed first."""
        for i in range(10):
            self.monitor.log_step_memory(step=i)
        # The history should contain the 5 most recent steps
        steps = [s["step"] for s in self.monitor.memory_history]
        self.assertEqual(steps, [5, 6, 7, 8, 9])

    def test_max_history_zero_disables_history(self):
        """When max_history_size=0, no history is stored."""
        monitor = ComprehensiveMemoryMonitor(rank=0, max_history_size=0)
        monitor.start_monitoring()
        for i in range(10):
            monitor.log_step_memory(step=i)
        self.assertEqual(len(monitor.memory_history), 0)
        import tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()


class TestComprehensiveMemoryMonitorAnalyze(unittest.TestCase):
    """Test ComprehensiveMemoryMonitor.analyze_memory_growth."""

    def setUp(self):
        self.monitor = ComprehensiveMemoryMonitor(rank=0, max_history_size=100)
        self.monitor.start_monitoring()

    def tearDown(self):
        import tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    def test_returns_none_with_empty_history(self):
        """Returns None when no history is available."""
        result = self.monitor.analyze_memory_growth()
        self.assertIsNone(result)

    def test_returns_none_with_single_entry(self):
        """Returns None when only one history entry exists."""
        self.monitor.log_step_memory(step=1)
        result = self.monitor.analyze_memory_growth()
        self.assertIsNone(result)

    def test_returns_dict_with_two_entries(self):
        """Returns a dict when at least two history entries exist."""
        self.monitor.log_step_memory(step=1)
        self.monitor.log_step_memory(step=2)
        result = self.monitor.analyze_memory_growth()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_analysis_keys(self):
        """Analysis dict contains expected keys."""
        self.monitor.log_step_memory(step=1)
        self.monitor.log_step_memory(step=2)
        result = self.monitor.analyze_memory_growth()
        self.assertIn("rss_per_step", result)
        self.assertIn("tensor_growth", result)
        self.assertIn("object_growth", result)

    def test_analysis_with_multiple_entries(self):
        """Analysis works correctly with many history entries."""
        for i in range(1, 11):
            self.monitor.log_step_memory(step=i)
        result = self.monitor.analyze_memory_growth()
        self.assertIsNotNone(result)
        # rss_per_step should be a number
        self.assertIsInstance(result["rss_per_step"], float)
        # tensor_growth and object_growth should be integers
        self.assertIsInstance(result["tensor_growth"], int)
        self.assertIsInstance(result["object_growth"], int)


# ---------------------------------------------------------------------------
# get_memory_monitor singleton
# ---------------------------------------------------------------------------


class TestGetMemoryMonitor(unittest.TestCase):
    """Test get_memory_monitor global singleton accessor."""

    def setUp(self):
        """Reset the global singleton before each test."""
        memory_monitor_module.memory_monitor = None

    def tearDown(self):
        """Reset the global singleton after each test."""
        memory_monitor_module.memory_monitor = None

    def test_returns_instance(self):
        """get_memory_monitor returns a ComprehensiveMemoryMonitor."""
        monitor = get_memory_monitor()
        self.assertIsInstance(monitor, ComprehensiveMemoryMonitor)

    def test_returns_same_instance(self):
        """Repeated calls return the same singleton instance."""
        m1 = get_memory_monitor()
        m2 = get_memory_monitor()
        self.assertIs(m1, m2)

    def test_uses_provided_rank(self):
        """First call uses the provided rank parameter."""
        monitor = get_memory_monitor(rank=7)
        self.assertEqual(monitor.rank, 7)

    def test_subsequent_rank_ignored(self):
        """Subsequent calls ignore the rank parameter (singleton already created)."""
        m1 = get_memory_monitor(rank=3)
        m2 = get_memory_monitor(rank=99)
        self.assertIs(m1, m2)
        self.assertEqual(m2.rank, 3)

    def test_default_rank_is_zero(self):
        """Default rank is 0 when no argument is provided."""
        monitor = get_memory_monitor()
        self.assertEqual(monitor.rank, 0)

    def test_reset_allows_new_instance(self):
        """Resetting the global variable allows a new instance to be created."""
        m1 = get_memory_monitor(rank=1)
        memory_monitor_module.memory_monitor = None
        m2 = get_memory_monitor(rank=2)
        self.assertIsNot(m1, m2)
        self.assertEqual(m1.rank, 1)
        self.assertEqual(m2.rank, 2)


if __name__ == "__main__":
    unittest.main()
