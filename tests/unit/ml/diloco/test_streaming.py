"""Tests for Streaming DiLoCo (Phase 3): FragmentManager, fragment server
endpoints, and end-to-end streaming worker sync."""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.fragments import FragmentManager
from forgather.ml.diloco.server import DiLoCoServer
from forgather.ml.diloco.worker import DiLoCoWorker


def _make_model(num_params=6):
    """Create a simple linear model with num_params parameters."""
    # Use a model with enough params for fragment splitting
    layers = []
    for i in range(num_params // 2):
        layers.append(nn.Linear(4, 4, bias=True))
    model = nn.Sequential(*layers)
    return model


class TestFragmentManager(unittest.TestCase):
    """Tests for FragmentManager parameter splitting and scheduling."""

    def test_contiguous_split(self):
        """Parameters are split into contiguous groups."""
        model = _make_model(6)  # 3 Linear layers = 6 params (3 weights + 3 biases)
        fm = FragmentManager(model, num_fragments=3)

        self.assertEqual(fm.num_fragments, 3)
        self.assertEqual(len(fm.fragments), 3)

        # All parameters assigned to exactly one fragment
        all_names = set()
        for frag in fm.fragments:
            for name in frag:
                self.assertNotIn(name, all_names, f"{name} in multiple fragments")
                all_names.add(name)

        # Total params match model
        model_names = {name for name, _ in model.named_parameters()}
        self.assertEqual(all_names, model_names)

    def test_split_uneven(self):
        """Uneven splits distribute remainder to first fragments."""
        model = _make_model(6)
        param_count = sum(1 for _ in model.parameters())

        fm = FragmentManager(model, num_fragments=4)
        sizes = [len(f) for f in fm.fragments]

        # All params assigned
        self.assertEqual(sum(sizes), param_count)
        # Difference between largest and smallest is at most 1
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_single_fragment(self):
        """Single fragment contains all parameters."""
        model = _make_model(6)
        fm = FragmentManager(model, num_fragments=1)

        self.assertEqual(len(fm.fragments), 1)
        param_count = sum(1 for _ in model.parameters())
        self.assertEqual(len(fm.fragments[0]), param_count)

    def test_invalid_num_fragments(self):
        """Invalid num_fragments raises ValueError."""
        model = _make_model(6)
        with self.assertRaises(ValueError):
            FragmentManager(model, num_fragments=0)

    def test_too_many_fragments(self):
        """More fragments than parameters raises ValueError."""
        model = _make_model(2)
        param_count = sum(1 for _ in model.parameters())
        with self.assertRaises(ValueError):
            FragmentManager(model, num_fragments=param_count + 1)

    def test_param_to_fragment_mapping(self):
        """param_to_fragment correctly maps every parameter."""
        model = _make_model(6)
        fm = FragmentManager(model, num_fragments=3)

        for frag_id, names in enumerate(fm.fragments):
            for name in names:
                self.assertEqual(fm.param_to_fragment[name], frag_id)

    def test_schedule_three_fragments(self):
        """Fragment schedule with 3 fragments and sync_every=600."""
        model = _make_model(6)
        fm = FragmentManager(model, num_fragments=3)

        # sync_every=600, 3 fragments -> interval=200
        self.assertIsNone(fm.get_fragment_schedule(0, 600))
        self.assertIsNone(fm.get_fragment_schedule(100, 600))

        self.assertEqual(fm.get_fragment_schedule(200, 600), 0)
        self.assertIsNone(fm.get_fragment_schedule(201, 600))
        self.assertIsNone(fm.get_fragment_schedule(399, 600))

        self.assertEqual(fm.get_fragment_schedule(400, 600), 1)
        self.assertEqual(fm.get_fragment_schedule(600, 600), 2)

        # Next round
        self.assertEqual(fm.get_fragment_schedule(800, 600), 0)
        self.assertEqual(fm.get_fragment_schedule(1000, 600), 1)
        self.assertEqual(fm.get_fragment_schedule(1200, 600), 2)

    def test_schedule_two_fragments(self):
        """Fragment schedule with 2 fragments and sync_every=100."""
        model = _make_model(6)
        fm = FragmentManager(model, num_fragments=2)

        # sync_every=100, 2 fragments -> interval=50
        self.assertIsNone(fm.get_fragment_schedule(0, 100))
        self.assertEqual(fm.get_fragment_schedule(50, 100), 0)
        self.assertEqual(fm.get_fragment_schedule(100, 100), 1)
        self.assertEqual(fm.get_fragment_schedule(150, 100), 0)
        self.assertEqual(fm.get_fragment_schedule(200, 100), 1)

    def test_is_last_fragment(self):
        """is_last_fragment correctly identifies the last fragment in a round."""
        model = _make_model(6)
        fm = FragmentManager(model, num_fragments=3)

        self.assertFalse(fm.is_last_fragment(200, 600))  # fragment 0
        self.assertFalse(fm.is_last_fragment(400, 600))  # fragment 1
        self.assertTrue(fm.is_last_fragment(600, 600))   # fragment 2 (last)

    def test_compute_fragment_pseudogradients(self):
        """Pseudo-gradients computed only for fragment's parameters."""
        model = _make_model(6)
        fm = FragmentManager(model, num_fragments=2)

        # Simulate global params (snapshot before training)
        global_params = {
            name: p.data.detach().clone().cpu()
            for name, p in model.named_parameters()
        }

        # Simulate training: modify all model params
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 0.5)

        # Compute pseudo-grads for fragment 0 only
        pseudograds = fm.compute_fragment_pseudogradients(
            0, global_params, model, bf16_comm=False
        )

        # Only fragment 0's params should be present
        frag0_names = set(fm.fragments[0])
        self.assertEqual(set(pseudograds.keys()), frag0_names)

        # Verify values: pseudo_grad = global - local (local moved by +0.5)
        for name in frag0_names:
            expected = global_params[name] - (global_params[name] + 0.5)
            torch.testing.assert_close(pseudograds[name], expected)

    def test_apply_fragment_global_params(self):
        """Applying fragment params updates only that fragment in the model."""
        model = _make_model(6)
        fm = FragmentManager(model, num_fragments=2)

        # Save initial state
        global_params = {
            name: p.data.detach().clone().cpu()
            for name, p in model.named_parameters()
        }

        # New params for fragment 0
        frag0_names = fm.fragments[0]
        new_params = {
            name: torch.ones_like(global_params[name]) * 99.0
            for name in frag0_names
        }

        fm.apply_fragment_global_params(0, new_params, model, global_params)

        # Fragment 0 params should be updated
        model_dict = dict(model.named_parameters())
        for name in frag0_names:
            torch.testing.assert_close(
                model_dict[name].data.cpu(),
                torch.ones_like(global_params[name]) * 99.0,
            )

        # Fragment 1 params should be unchanged
        frag1_names = fm.fragments[1]
        for name in frag1_names:
            torch.testing.assert_close(
                model_dict[name].data.cpu(),
                global_params[name],  # original from before apply
            )

        # Global snapshot updated for fragment 0
        for name in frag0_names:
            torch.testing.assert_close(
                global_params[name],
                torch.ones_like(global_params[name]) * 99.0,
            )


class TestFragmentServerDirect(unittest.TestCase):
    """Test server-side fragment handling without HTTP."""

    def _make_server(self, num_workers=1, async_mode=False):
        model = _make_model(6)
        state_dict = model.state_dict()
        server = DiLoCoServer(
            model_state_dict=state_dict,
            num_workers=num_workers,
            outer_optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.0),
            async_mode=async_mode,
        )
        return server, state_dict

    def test_get_params_by_names(self):
        """_get_params_by_names returns correct subset."""
        server, state_dict = self._make_server()
        names = list(state_dict.keys())[:2]
        result = server._get_params_by_names(names)

        self.assertEqual(set(result.keys()), set(names))
        for name in names:
            torch.testing.assert_close(result[name], state_dict[name].float())

    def test_fragment_outer_optimizer(self):
        """Fragment optimizer only updates fragment params."""
        server, state_dict = self._make_server()
        all_names = list(state_dict.keys())

        # Save initial state
        initial = server.get_global_params()

        # Create pseudo-grads for first 2 params only
        frag_names = all_names[:2]
        pseudograds = {name: torch.ones_like(state_dict[name]) * 0.5 for name in frag_names}

        server._apply_fragment_outer_optimizer([pseudograds])

        updated = server.get_global_params()

        # Fragment params should have changed
        for name in frag_names:
            diff = (updated[name] - initial[name]).abs().sum().item()
            self.assertGreater(diff, 0, f"Fragment param {name} should have changed")

        # Non-fragment params should be unchanged
        for name in all_names[2:]:
            torch.testing.assert_close(
                updated[name], initial[name],
                msg=f"Non-fragment param {name} should not change",
            )


class TestFragmentServerClient(unittest.TestCase):
    """Test fragment endpoints via HTTP server + client."""

    def _start_server(self, num_workers=1, async_mode=False):
        model = _make_model(6)
        state_dict = model.state_dict()
        server = DiLoCoServer(
            model_state_dict=state_dict,
            num_workers=num_workers,
            port=None,  # auto-select
            outer_optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.0),
            async_mode=async_mode,
        )
        server.start()
        client = DiLoCoClient(f"localhost:{server.port}")
        return server, client, state_dict

    def test_single_worker_fragment_sync(self):
        """Single worker can submit and receive fragment pseudo-gradients (sync mode)."""
        server, client, state_dict = self._start_server(num_workers=1)
        try:
            # Register
            client.register("w0", {"hostname": "test"})

            # Submit fragment (first 2 params)
            all_names = list(state_dict.keys())
            frag_names = all_names[:2]
            pseudograds = {name: torch.ones_like(state_dict[name]) * 0.1 for name in frag_names}

            result = client.submit_fragment_pseudogradients("w0", 0, pseudograds)

            # Should get back only the fragment's params
            self.assertEqual(set(result.keys()), set(frag_names))

            # Params should have been updated by outer optimizer
            for name in frag_names:
                diff = (result[name] - state_dict[name].float()).abs().sum().item()
                self.assertGreater(diff, 0)
        finally:
            server.stop()

    def test_single_worker_fragment_async(self):
        """Fragment submission works in async mode."""
        server, client, state_dict = self._start_server(num_workers=1, async_mode=True)
        try:
            client.register("w0", {"hostname": "test"})

            all_names = list(state_dict.keys())
            frag_names = all_names[:2]
            pseudograds = {name: torch.ones_like(state_dict[name]) * 0.1 for name in frag_names}

            result = client.submit_fragment_pseudogradients("w0", 0, pseudograds)
            self.assertEqual(set(result.keys()), set(frag_names))
        finally:
            server.stop()

    def test_two_workers_fragment_sync(self):
        """Two workers sync a fragment with barrier."""
        server, client, state_dict = self._start_server(num_workers=2)
        try:
            client.register("w0", {"hostname": "test"})
            client2 = DiLoCoClient(f"localhost:{server.port}")
            client2.register("w1", {"hostname": "test2"})

            all_names = list(state_dict.keys())
            frag_names = all_names[:2]

            pg1 = {name: torch.ones_like(state_dict[name]) * 0.1 for name in frag_names}
            pg2 = {name: torch.ones_like(state_dict[name]) * 0.3 for name in frag_names}

            results = [None, None]
            errors = [None, None]

            def submit(idx, c, worker_id, pg):
                try:
                    results[idx] = c.submit_fragment_pseudogradients(worker_id, 0, pg)
                except Exception as e:
                    errors[idx] = e

            t1 = threading.Thread(target=submit, args=(0, client, "w0", pg1))
            t2 = threading.Thread(target=submit, args=(1, client2, "w1", pg2))
            t1.start()
            t2.start()
            t1.join(timeout=30)
            t2.join(timeout=30)

            self.assertIsNone(errors[0], f"Worker 0 error: {errors[0]}")
            self.assertIsNone(errors[1], f"Worker 1 error: {errors[1]}")

            # Both should get same result (averaged pseudo-grads applied)
            for name in frag_names:
                torch.testing.assert_close(results[0][name], results[1][name])
        finally:
            server.stop()

    def test_multiple_fragments_sequential(self):
        """Single worker syncs multiple fragments sequentially."""
        server, client, state_dict = self._start_server(num_workers=1)
        try:
            client.register("w0", {"hostname": "test"})

            all_names = list(state_dict.keys())
            mid = len(all_names) // 2
            frag0_names = all_names[:mid]
            frag1_names = all_names[mid:]

            # Submit fragment 0
            pg0 = {name: torch.ones_like(state_dict[name]) * 0.1 for name in frag0_names}
            result0 = client.submit_fragment_pseudogradients("w0", 0, pg0)
            self.assertEqual(set(result0.keys()), set(frag0_names))

            # Submit fragment 1
            pg1 = {name: torch.ones_like(state_dict[name]) * 0.2 for name in frag1_names}
            result1 = client.submit_fragment_pseudogradients("w0", 1, pg1)
            self.assertEqual(set(result1.keys()), set(frag1_names))
        finally:
            server.stop()

    def test_fragment_status(self):
        """Fragment submissions tracked in status."""
        server, client, state_dict = self._start_server(num_workers=1, async_mode=True)
        try:
            client.register("w0", {"hostname": "test"})

            all_names = list(state_dict.keys())
            frag_names = all_names[:2]
            pg = {name: torch.ones_like(state_dict[name]) * 0.1 for name in frag_names}

            client.submit_fragment_pseudogradients("w0", 0, pg)

            status = client.get_status()
            self.assertGreater(status.get("fragment_submissions", 0), 0)
        finally:
            server.stop()


class TestStreamingWorker(unittest.TestCase):
    """End-to-end tests for streaming DiLoCo worker."""

    def _start_server(self, num_workers=1, async_mode=False):
        model = _make_model(6)
        state_dict = model.state_dict()
        server = DiLoCoServer(
            model_state_dict=state_dict,
            num_workers=num_workers,
            port=None,
            outer_optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.0),
            async_mode=async_mode,
        )
        server.start()
        return server, state_dict

    def test_streaming_worker_fragments_sync(self):
        """Streaming worker syncs fragments at staggered intervals."""
        server, state_dict = self._start_server(num_workers=1)
        try:
            model = _make_model(6)
            model.load_state_dict(state_dict)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            sync_every = 6
            num_fragments = 2
            # Fragment interval = 3 steps

            worker = DiLoCoWorker(
                model=model,
                optimizer=optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=sync_every,
                num_fragments=num_fragments,
                bf16_comm=False,
            )
            worker.start()

            try:
                # Train for sync_every steps
                for _ in range(sync_every):
                    x = torch.randn(2, 4)
                    out = model(x)
                    loss = out.sum()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Wait for any in-flight fragment
                worker._wait_and_apply_inflight_fragment()

                # Worker should have synced 2 fragments and completed 1 round
                self.assertEqual(worker._sync_count, 1)
                self.assertEqual(worker._fragment_syncs, 2)
            finally:
                worker.stop()
        finally:
            server.stop()

    def test_streaming_worker_async_mode(self):
        """Streaming worker works with async server."""
        server, state_dict = self._start_server(num_workers=1, async_mode=True)
        try:
            model = _make_model(6)
            model.load_state_dict(state_dict)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            sync_every = 6
            num_fragments = 3

            worker = DiLoCoWorker(
                model=model,
                optimizer=optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=sync_every,
                num_fragments=num_fragments,
                bf16_comm=False,
            )
            worker.start()

            try:
                # Train for 2 full rounds (12 steps)
                for _ in range(sync_every * 2):
                    x = torch.randn(2, 4)
                    out = model(x)
                    loss = out.sum()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                worker._wait_and_apply_inflight_fragment()

                self.assertEqual(worker._sync_count, 2)
                # 3 fragments per round * 2 rounds = 6
                self.assertEqual(worker._fragment_syncs, 6)
            finally:
                worker.stop()
        finally:
            server.stop()

    def test_streaming_params_updated_per_fragment(self):
        """Each fragment sync updates only that fragment's model parameters."""
        server, state_dict = self._start_server(num_workers=1)
        try:
            model = _make_model(6)
            model.load_state_dict(state_dict)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            sync_every = 4
            num_fragments = 2
            # Fragment interval = 2 steps

            worker = DiLoCoWorker(
                model=model,
                optimizer=optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=sync_every,
                num_fragments=num_fragments,
                bf16_comm=False,
            )
            worker.start()

            try:
                # Record state before training
                pre_train = {
                    name: p.data.clone()
                    for name, p in model.named_parameters()
                }

                # Train fragment_interval steps to trigger first fragment
                for _ in range(sync_every // num_fragments):
                    x = torch.randn(2, 4)
                    out = model(x)
                    loss = out.sum()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Wait for fragment 0 to complete
                worker._wait_and_apply_inflight_fragment()
                self.assertEqual(worker._fragment_syncs, 1)
            finally:
                worker.stop()
        finally:
            server.stop()

    def test_streaming_metrics(self):
        """Streaming worker includes fragment metrics."""
        server, state_dict = self._start_server(num_workers=1)
        try:
            model = _make_model(6)
            model.load_state_dict(state_dict)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            worker = DiLoCoWorker(
                model=model,
                optimizer=optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=6,
                num_fragments=2,
                bf16_comm=False,
            )
            worker.start()

            try:
                metrics = worker.sync_metrics
                self.assertIn("diloco/num_fragments", metrics)
                self.assertIn("diloco/fragment_syncs", metrics)
                self.assertEqual(metrics["diloco/num_fragments"], 2)
            finally:
                worker.stop()
        finally:
            server.stop()

    def test_streaming_force_sync_does_full_model(self):
        """force_sync does a full-model sync even in streaming mode."""
        server, state_dict = self._start_server(num_workers=1)
        try:
            model = _make_model(6)
            model.load_state_dict(state_dict)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            worker = DiLoCoWorker(
                model=model,
                optimizer=optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=100,
                num_fragments=2,
                bf16_comm=False,
            )
            worker.start()

            try:
                # Train a few steps (not enough to trigger fragment sync)
                for _ in range(3):
                    x = torch.randn(2, 4)
                    out = model(x)
                    loss = out.sum()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # force_sync does a full-model sync via _sync()
                worker.force_sync()
                self.assertEqual(worker._sync_count, 1)
            finally:
                worker.stop()
        finally:
            server.stop()

    def test_num_fragments_one_uses_standard_path(self):
        """num_fragments=1 behaves identically to non-streaming mode."""
        server, state_dict = self._start_server(num_workers=1)
        try:
            model = _make_model(6)
            model.load_state_dict(state_dict)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            worker = DiLoCoWorker(
                model=model,
                optimizer=optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=4,
                num_fragments=1,  # Should use standard path
                bf16_comm=False,
            )
            worker.start()

            try:
                # Fragment manager should not be created
                self.assertIsNone(worker._fragment_manager)

                for _ in range(4):
                    x = torch.randn(2, 4)
                    out = model(x)
                    loss = out.sum()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                self.assertEqual(worker._sync_count, 1)
                self.assertEqual(worker._fragment_syncs, 0)
            finally:
                worker.stop()
        finally:
            server.stop()


class TestStreamingBackgroundOverlap(unittest.TestCase):
    """Tests verifying that communication happens in background threads."""

    def test_inflight_thread_created(self):
        """Verify that fragment sync creates a background thread."""
        server_model = _make_model(6)
        state_dict = server_model.state_dict()
        server = DiLoCoServer(
            model_state_dict=state_dict,
            num_workers=1,
            port=None,
            outer_optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.0),
        )
        server.start()

        try:
            model = _make_model(6)
            model.load_state_dict(state_dict)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            worker = DiLoCoWorker(
                model=model,
                optimizer=optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=4,
                num_fragments=2,
                bf16_comm=False,
            )
            worker.start()

            try:
                # Train 2 steps to trigger fragment 0
                for _ in range(2):
                    x = torch.randn(2, 4)
                    out = model(x)
                    loss = out.sum()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # An inflight thread should have been created (may or may not
                # have completed yet depending on timing)
                # Wait and apply to verify it completes
                worker._wait_and_apply_inflight_fragment()
                self.assertEqual(worker._fragment_syncs, 1)
            finally:
                worker.stop()
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
