#!/usr/bin/env python3
"""
Unit tests for distributed training environment utilities.

Tests the functions and classes in forgather.ml.distributed that can be
exercised without actual torch.distributed initialization:
- Environment variable reading functions (get_world_size, get_rank, etc.)
- null_barrier no-op function
- prefix_logger_rank logger modification
- init_from_env bidirectional env/object synchronization
- StaticDistributedEnvironment dataclass
- from_env factory function
- get_barrier_fn when world_size==1
"""

import logging
import os
import unittest
from unittest.mock import MagicMock, patch

from forgather.ml.distributed import (
    StaticDistributedEnvironment,
    from_env,
    get_barrier_fn,
    get_local_rank,
    get_local_world_size,
    get_rank,
    get_world_size,
    init_from_env,
    null_barrier,
    prefix_logger_rank,
)


# ---------------------------------------------------------------------------
# Environment variable reading functions
# ---------------------------------------------------------------------------


class TestGetWorldSize(unittest.TestCase):
    """Test get_world_size() reading of WORLD_SIZE env var."""

    def test_default_when_unset(self):
        """Returns 1 when WORLD_SIZE is not set."""
        env = {k: v for k, v in os.environ.items() if k != "WORLD_SIZE"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_world_size(), 1)

    def test_reads_env_var(self):
        """Returns the integer value of WORLD_SIZE."""
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}):
            self.assertEqual(get_world_size(), 8)

    def test_reads_single_process(self):
        """Returns 1 when WORLD_SIZE is explicitly '1'."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            self.assertEqual(get_world_size(), 1)

    def test_large_value(self):
        """Handles large world sizes correctly."""
        with patch.dict(os.environ, {"WORLD_SIZE": "256"}):
            self.assertEqual(get_world_size(), 256)


class TestGetLocalWorldSize(unittest.TestCase):
    """Test get_local_world_size() reading of LOCAL_WORLD_SIZE env var."""

    def test_default_when_unset(self):
        """Returns 1 when LOCAL_WORLD_SIZE is not set."""
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_WORLD_SIZE"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_local_world_size(), 1)

    def test_reads_env_var(self):
        """Returns the integer value of LOCAL_WORLD_SIZE."""
        with patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"}):
            self.assertEqual(get_local_world_size(), 4)

    def test_single_gpu_node(self):
        """Returns 1 when LOCAL_WORLD_SIZE is explicitly '1'."""
        with patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "1"}):
            self.assertEqual(get_local_world_size(), 1)


class TestGetRank(unittest.TestCase):
    """Test get_rank() reading of RANK env var."""

    def test_default_when_unset(self):
        """Returns 0 when RANK is not set."""
        env = {k: v for k, v in os.environ.items() if k != "RANK"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_rank(), 0)

    def test_reads_env_var(self):
        """Returns the integer value of RANK."""
        with patch.dict(os.environ, {"RANK": "3"}):
            self.assertEqual(get_rank(), 3)

    def test_rank_zero(self):
        """Returns 0 when RANK is explicitly '0'."""
        with patch.dict(os.environ, {"RANK": "0"}):
            self.assertEqual(get_rank(), 0)

    def test_high_rank(self):
        """Handles high rank numbers correctly."""
        with patch.dict(os.environ, {"RANK": "127"}):
            self.assertEqual(get_rank(), 127)


class TestGetLocalRank(unittest.TestCase):
    """Test get_local_rank() reading of LOCAL_RANK env var."""

    def test_default_when_unset(self):
        """Returns 0 when LOCAL_RANK is not set."""
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_RANK"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_local_rank(), 0)

    def test_reads_env_var(self):
        """Returns the integer value of LOCAL_RANK."""
        with patch.dict(os.environ, {"LOCAL_RANK": "2"}):
            self.assertEqual(get_local_rank(), 2)

    def test_local_rank_zero(self):
        """Returns 0 when LOCAL_RANK is explicitly '0'."""
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            self.assertEqual(get_local_rank(), 0)


# ---------------------------------------------------------------------------
# null_barrier
# ---------------------------------------------------------------------------


class TestNullBarrier(unittest.TestCase):
    """Test null_barrier no-op function."""

    def test_returns_none(self):
        """null_barrier always returns None."""
        self.assertIsNone(null_barrier())

    def test_accepts_args(self):
        """null_barrier silently accepts arbitrary positional arguments."""
        self.assertIsNone(null_barrier(1, 2, 3))

    def test_accepts_kwargs(self):
        """null_barrier silently accepts arbitrary keyword arguments."""
        self.assertIsNone(null_barrier(group="test", device_ids=[0]))

    def test_accepts_mixed_args(self):
        """null_barrier accepts both positional and keyword arguments."""
        self.assertIsNone(null_barrier("a", "b", key="value"))

    def test_callable(self):
        """null_barrier is callable."""
        self.assertTrue(callable(null_barrier))


# ---------------------------------------------------------------------------
# prefix_logger_rank
# ---------------------------------------------------------------------------


class TestPrefixLoggerRank(unittest.TestCase):
    """Test prefix_logger_rank logger modification."""

    def _make_logger(self, name):
        """Create a fresh logger for testing."""
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.DEBUG)
        return logger

    def test_disables_propagation(self):
        """Logger propagation is disabled after prefix_logger_rank."""
        logger = self._make_logger("test.prefix.propagation")
        self.assertTrue(logger.propagate)
        prefix_logger_rank(logger)
        self.assertFalse(logger.propagate)

    def test_adds_handler(self):
        """A StreamHandler is added to the logger."""
        logger = self._make_logger("test.prefix.handler")
        initial_count = len(logger.handlers)
        prefix_logger_rank(logger)
        self.assertEqual(len(logger.handlers), initial_count + 1)
        self.assertIsInstance(logger.handlers[-1], logging.StreamHandler)

    def test_returns_logger(self):
        """The modified logger is returned."""
        logger = self._make_logger("test.prefix.return")
        result = prefix_logger_rank(logger)
        self.assertIs(result, logger)

    def test_default_filter_passes_rank0(self):
        """Default filter passes records when rank is 0."""
        logger = self._make_logger("test.prefix.rank0")
        with patch("forgather.ml.distributed.get_rank", return_value=0):
            prefix_logger_rank(logger)
            handler = logger.handlers[-1]
            record = logging.LogRecord(
                "test", logging.INFO, "", 0, "msg", (), None
            )
            # The filter function is added via addFilter; check it passes
            self.assertTrue(handler.filter(record))

    def test_default_filter_blocks_nonzero_rank(self):
        """Default filter blocks records when rank is not 0."""
        logger = self._make_logger("test.prefix.nonzero")
        with patch("forgather.ml.distributed.get_rank", return_value=3):
            prefix_logger_rank(logger)
            handler = logger.handlers[-1]
            record = logging.LogRecord(
                "test", logging.INFO, "", 0, "msg", (), None
            )
            self.assertFalse(handler.filter(record))

    def test_custom_filter(self):
        """Custom filter function is applied correctly."""
        logger = self._make_logger("test.prefix.custom_filter")
        # Accept only rank 2
        custom_filter = lambda rank: rank == 2

        with patch("forgather.ml.distributed.get_rank", return_value=2):
            prefix_logger_rank(logger, filter=custom_filter)
            handler = logger.handlers[-1]
            record = logging.LogRecord(
                "test", logging.INFO, "", 0, "msg", (), None
            )
            self.assertTrue(handler.filter(record))

        with patch("forgather.ml.distributed.get_rank", return_value=0):
            # Re-create since the filter references get_rank at call time
            record2 = logging.LogRecord(
                "test", logging.INFO, "", 0, "msg", (), None
            )
            self.assertFalse(handler.filter(record2))

    def test_custom_format(self):
        """Custom format string is applied to the handler."""
        logger = self._make_logger("test.prefix.custom_format")
        custom_format = "CUSTOM: %(message)s"
        prefix_logger_rank(logger, format=custom_format)
        handler = logger.handlers[-1]
        self.assertEqual(handler.formatter._fmt, custom_format)

    def test_default_format_without_filter(self):
        """When no custom filter is given, default format does not include rank."""
        logger = self._make_logger("test.prefix.default_fmt_no_filter")
        prefix_logger_rank(logger)
        handler = logger.handlers[-1]
        self.assertNotIn("%(rank)s", handler.formatter._fmt)

    def test_default_format_with_filter(self):
        """When custom filter is given, default format includes rank prefix."""
        logger = self._make_logger("test.prefix.default_fmt_with_filter")
        prefix_logger_rank(logger, filter=lambda rank: True)
        handler = logger.handlers[-1]
        self.assertIn("%(rank)s", handler.formatter._fmt)

    def test_rank_attribute_set_on_record(self):
        """The rank_filter sets a 'rank' attribute on the log record."""
        logger = self._make_logger("test.prefix.rank_attr")
        with patch("forgather.ml.distributed.get_rank", return_value=5):
            prefix_logger_rank(logger, filter=lambda rank: True)
            handler = logger.handlers[-1]
            record = logging.LogRecord(
                "test", logging.INFO, "", 0, "msg", (), None
            )
            handler.filter(record)
            self.assertEqual(record.rank, 5)


# ---------------------------------------------------------------------------
# init_from_env
# ---------------------------------------------------------------------------


class TestInitFromEnv(unittest.TestCase):
    """Test init_from_env bidirectional synchronization."""

    def _clear_dist_env_vars(self):
        """Return a clean env dict with distributed vars removed."""
        keys = [
            "LOCAL_RANK", "RANK", "WORLD_SIZE",
            "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
        ]
        return {k: v for k, v in os.environ.items() if k not in keys}

    def test_exports_to_env_when_unset(self):
        """When env vars are not set, init_from_env exports from the dist object."""
        dist_obj = StaticDistributedEnvironment(
            rank=2, local_rank=1, world_size=4,
            local_world_size=2, master_addr="10.0.0.1", master_port=12345,
        )
        with patch.dict(os.environ, self._clear_dist_env_vars(), clear=True):
            init_from_env(dist_obj)
            self.assertEqual(os.environ["RANK"], "2")
            self.assertEqual(os.environ["LOCAL_RANK"], "1")
            self.assertEqual(os.environ["WORLD_SIZE"], "4")
            self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "2")
            self.assertEqual(os.environ["MASTER_ADDR"], "10.0.0.1")
            self.assertEqual(os.environ["MASTER_PORT"], "12345")

    def test_imports_from_env_when_set(self):
        """When env vars are set, init_from_env updates the dist object."""
        dist_obj = StaticDistributedEnvironment()
        env = self._clear_dist_env_vars()
        env.update({
            "RANK": "7",
            "LOCAL_RANK": "3",
            "WORLD_SIZE": "16",
            "LOCAL_WORLD_SIZE": "8",
            "MASTER_ADDR": "192.168.1.10",
            "MASTER_PORT": "54321",
        })
        with patch.dict(os.environ, env, clear=True):
            init_from_env(dist_obj)
            self.assertEqual(dist_obj.rank, 7)
            self.assertEqual(dist_obj.local_rank, 3)
            self.assertEqual(dist_obj.world_size, 16)
            self.assertEqual(dist_obj.local_world_size, 8)
            self.assertEqual(dist_obj.master_addr, "192.168.1.10")
            self.assertEqual(dist_obj.master_port, 54321)

    def test_partial_env_set(self):
        """When only some env vars are set, imports those and exports the rest."""
        dist_obj = StaticDistributedEnvironment(
            rank=0, local_rank=0, world_size=1,
            local_world_size=1, master_addr="localhost", master_port=29501,
        )
        env = self._clear_dist_env_vars()
        env.update({
            "RANK": "5",
            "WORLD_SIZE": "8",
        })
        with patch.dict(os.environ, env, clear=True):
            init_from_env(dist_obj)
            # Imported from env
            self.assertEqual(dist_obj.rank, 5)
            self.assertEqual(dist_obj.world_size, 8)
            # Exported to env (were not set)
            self.assertEqual(os.environ["LOCAL_RANK"], "0")
            self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "1")
            self.assertEqual(os.environ["MASTER_ADDR"], "localhost")
            self.assertEqual(os.environ["MASTER_PORT"], "29501")

    def test_int_type_conversion(self):
        """Integer-typed env vars are properly converted when importing."""
        dist_obj = StaticDistributedEnvironment()
        env = self._clear_dist_env_vars()
        env.update({
            "RANK": "3",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "2",
            "MASTER_PORT": "9999",
            "MASTER_ADDR": "myhost",
        })
        with patch.dict(os.environ, env, clear=True):
            init_from_env(dist_obj)
            # Integer fields
            self.assertIsInstance(dist_obj.rank, int)
            self.assertIsInstance(dist_obj.local_rank, int)
            self.assertIsInstance(dist_obj.world_size, int)
            self.assertIsInstance(dist_obj.local_world_size, int)
            self.assertIsInstance(dist_obj.master_port, int)
            # String field
            self.assertIsInstance(dist_obj.master_addr, str)

    def test_master_addr_stays_string(self):
        """MASTER_ADDR is treated as a string, not converted to int."""
        dist_obj = StaticDistributedEnvironment()
        env = self._clear_dist_env_vars()
        env.update({"MASTER_ADDR": "node-0.example.com"})
        with patch.dict(os.environ, env, clear=True):
            init_from_env(dist_obj)
            self.assertEqual(dist_obj.master_addr, "node-0.example.com")


# ---------------------------------------------------------------------------
# StaticDistributedEnvironment
# ---------------------------------------------------------------------------


class TestStaticDistributedEnvironment(unittest.TestCase):
    """Test StaticDistributedEnvironment dataclass."""

    def test_default_values(self):
        """Defaults represent single-process CPU execution."""
        env = StaticDistributedEnvironment()
        self.assertEqual(env.rank, 0)
        self.assertEqual(env.local_rank, 0)
        self.assertEqual(env.world_size, 1)
        self.assertEqual(env.local_world_size, 1)
        self.assertEqual(env.master_addr, "localhost")
        self.assertEqual(env.master_port, 29501)
        self.assertEqual(env.device, "cpu")
        self.assertEqual(env.device_type, "cpu")

    def test_custom_values(self):
        """Custom values are stored correctly."""
        env = StaticDistributedEnvironment(
            rank=3, local_rank=1, world_size=8,
            local_world_size=4, master_addr="10.0.0.5",
            master_port=12345, device="cuda:1", device_type="cuda",
        )
        self.assertEqual(env.rank, 3)
        self.assertEqual(env.local_rank, 1)
        self.assertEqual(env.world_size, 8)
        self.assertEqual(env.local_world_size, 4)
        self.assertEqual(env.master_addr, "10.0.0.5")
        self.assertEqual(env.master_port, 12345)
        self.assertEqual(env.device, "cuda:1")
        self.assertEqual(env.device_type, "cuda")

    def test_keyword_only(self):
        """Arguments must be keyword-only (kw_only=True on the dataclass)."""
        with self.assertRaises(TypeError):
            StaticDistributedEnvironment(0, 0, 1, 1)

    def test_attributes_are_mutable(self):
        """Attributes can be set after creation (needed by init_from_env)."""
        env = StaticDistributedEnvironment()
        env.rank = 5
        env.world_size = 16
        env.master_addr = "remote-host"
        self.assertEqual(env.rank, 5)
        self.assertEqual(env.world_size, 16)
        self.assertEqual(env.master_addr, "remote-host")

    def test_is_dataclass(self):
        """StaticDistributedEnvironment is a dataclass."""
        import dataclasses
        self.assertTrue(dataclasses.is_dataclass(StaticDistributedEnvironment))

    def test_partial_override(self):
        """Only overridden fields change; rest keep defaults."""
        env = StaticDistributedEnvironment(rank=7, device="cuda:7")
        self.assertEqual(env.rank, 7)
        self.assertEqual(env.device, "cuda:7")
        # Defaults preserved
        self.assertEqual(env.local_rank, 0)
        self.assertEqual(env.world_size, 1)
        self.assertEqual(env.master_port, 29501)


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


class TestFromEnv(unittest.TestCase):
    """Test from_env factory function."""

    def _clear_dist_env_vars(self):
        """Return a clean env dict with distributed vars removed."""
        keys = [
            "LOCAL_RANK", "RANK", "WORLD_SIZE",
            "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
        ]
        return {k: v for k, v in os.environ.items() if k not in keys}

    def test_returns_static_env(self):
        """from_env returns a StaticDistributedEnvironment instance."""
        with patch.dict(os.environ, self._clear_dist_env_vars(), clear=True):
            env = from_env()
            self.assertIsInstance(env, StaticDistributedEnvironment)

    def test_default_values_exported(self):
        """When no env vars are set, defaults are exported."""
        with patch.dict(os.environ, self._clear_dist_env_vars(), clear=True):
            env = from_env()
            self.assertEqual(env.rank, 0)
            self.assertEqual(env.world_size, 1)
            self.assertEqual(os.environ["RANK"], "0")
            self.assertEqual(os.environ["WORLD_SIZE"], "1")

    def test_env_vars_override_defaults(self):
        """Environment variables override the default field values."""
        env_dict = self._clear_dist_env_vars()
        env_dict.update({
            "RANK": "5",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "4",
            "MASTER_ADDR": "head-node",
            "MASTER_PORT": "55555",
        })
        with patch.dict(os.environ, env_dict, clear=True):
            env = from_env()
            self.assertEqual(env.rank, 5)
            self.assertEqual(env.local_rank, 1)
            self.assertEqual(env.world_size, 8)
            self.assertEqual(env.local_world_size, 4)
            self.assertEqual(env.master_addr, "head-node")
            self.assertEqual(env.master_port, 55555)

    def test_kwargs_used_as_initial_values(self):
        """Keyword arguments are used as initial values before env sync."""
        with patch.dict(os.environ, self._clear_dist_env_vars(), clear=True):
            env = from_env(rank=10, world_size=20)
            # kwargs become the initial values exported to env
            self.assertEqual(env.rank, 10)
            self.assertEqual(env.world_size, 20)
            self.assertEqual(os.environ["RANK"], "10")
            self.assertEqual(os.environ["WORLD_SIZE"], "20")

    def test_env_vars_override_kwargs(self):
        """Environment variables take precedence over kwargs."""
        env_dict = self._clear_dist_env_vars()
        env_dict["RANK"] = "99"
        with patch.dict(os.environ, env_dict, clear=True):
            env = from_env(rank=0)
            # Env var wins over the kwarg
            self.assertEqual(env.rank, 99)

    def test_device_fields_not_affected_by_env(self):
        """device and device_type are not sync'd via init_from_env."""
        with patch.dict(os.environ, self._clear_dist_env_vars(), clear=True):
            env = from_env(device="cuda:3", device_type="cuda")
            self.assertEqual(env.device, "cuda:3")
            self.assertEqual(env.device_type, "cuda")


# ---------------------------------------------------------------------------
# get_barrier_fn
# ---------------------------------------------------------------------------


class TestGetBarrierFn(unittest.TestCase):
    """Test get_barrier_fn when world_size==1 (returns null_barrier)."""

    def test_returns_null_barrier_when_single_process(self):
        """When world_size is 1, get_barrier_fn returns null_barrier."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with patch("forgather.ml.distributed.dist") as mock_dist:
                mock_dist.is_available.return_value = False
                mock_dist.is_initialized.return_value = False
                barrier = get_barrier_fn()
                self.assertIs(barrier, null_barrier)

    def test_returns_null_barrier_dist_not_available(self):
        """When distributed is not available, returns null_barrier."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with patch("forgather.ml.distributed.dist") as mock_dist:
                mock_dist.is_available.return_value = False
                barrier = get_barrier_fn()
                self.assertIs(barrier, null_barrier)

    def test_returns_null_barrier_dist_not_initialized(self):
        """When distributed is available but not initialized, returns null_barrier."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with patch("forgather.ml.distributed.dist") as mock_dist:
                mock_dist.is_available.return_value = True
                mock_dist.is_initialized.return_value = False
                barrier = get_barrier_fn()
                self.assertIs(barrier, null_barrier)

    def test_null_barrier_is_callable(self):
        """The returned null_barrier is callable and returns None."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with patch("forgather.ml.distributed.dist") as mock_dist:
                mock_dist.is_available.return_value = False
                barrier = get_barrier_fn()
                result = barrier()
                self.assertIsNone(result)

    def test_assertion_error_when_world_size_not_1_and_not_initialized(self):
        """AssertionError raised when world_size > 1 but dist not initialized."""
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            with patch("forgather.ml.distributed.dist") as mock_dist:
                mock_dist.is_available.return_value = False
                mock_dist.is_initialized.return_value = False
                with self.assertRaises(AssertionError):
                    get_barrier_fn()


# ---------------------------------------------------------------------------
# Integration-style tests combining multiple functions
# ---------------------------------------------------------------------------


class TestDistributedIntegration(unittest.TestCase):
    """Integration tests combining multiple distributed utilities."""

    def _clear_dist_env_vars(self):
        """Return a clean env dict with distributed vars removed."""
        keys = [
            "LOCAL_RANK", "RANK", "WORLD_SIZE",
            "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
        ]
        return {k: v for k, v in os.environ.items() if k not in keys}

    def test_from_env_then_get_functions_consistent(self):
        """After from_env, get_rank/get_world_size read the exported values."""
        with patch.dict(os.environ, self._clear_dist_env_vars(), clear=True):
            env = from_env(rank=3, world_size=4, local_rank=1, local_world_size=2)
            self.assertEqual(get_rank(), 3)
            self.assertEqual(get_world_size(), 4)
            self.assertEqual(get_local_rank(), 1)
            self.assertEqual(get_local_world_size(), 2)

    def test_init_from_env_round_trip(self):
        """Values survive a round trip: object -> env -> new object."""
        with patch.dict(os.environ, self._clear_dist_env_vars(), clear=True):
            original = StaticDistributedEnvironment(
                rank=5, local_rank=2, world_size=8,
                local_world_size=4, master_addr="10.0.0.1", master_port=30000,
            )
            init_from_env(original)

            # Now create a new object and sync from env
            restored = StaticDistributedEnvironment()
            init_from_env(restored)

            self.assertEqual(restored.rank, original.rank)
            self.assertEqual(restored.local_rank, original.local_rank)
            self.assertEqual(restored.world_size, original.world_size)
            self.assertEqual(restored.local_world_size, original.local_world_size)
            self.assertEqual(restored.master_addr, original.master_addr)
            self.assertEqual(restored.master_port, original.master_port)

    def test_single_process_workflow(self):
        """Full single-process workflow: from_env + get_barrier_fn."""
        with patch.dict(os.environ, self._clear_dist_env_vars(), clear=True):
            env = from_env()
            self.assertEqual(env.world_size, 1)
            self.assertEqual(env.rank, 0)

            with patch("forgather.ml.distributed.dist") as mock_dist:
                mock_dist.is_available.return_value = False
                mock_dist.is_initialized.return_value = False
                barrier = get_barrier_fn()
                self.assertIs(barrier, null_barrier)
                self.assertIsNone(barrier())


if __name__ == "__main__":
    unittest.main()
