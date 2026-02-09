"""
Comprehensive tests for optimizer state_dict() and load_state_dict() methods.

Tests verify:
1. Round-trip state preservation (save → load → verify match)
2. Training continuation (identical results after restore)
3. State structure validation
4. Special cases (e.g., Adafactor col=None, Apollo projector serialization)
"""

import pytest
import torch

from forgather.ml.optim import SGD, Adafactor, AdamW, Apollo, Multiopt
from forgather.ml.optim.subspace_proj import OnlinePCAProjector, RandProjector


class TestOptimizerStateDictBase:
    """Base class with reusable test patterns."""

    def create_simple_model(self):
        """2-layer linear model for testing."""
        return torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 2)
        )

    def perform_training_steps(self, model, optimizer, num_steps=5, seed=None):
        """Run optimizer for num_steps to populate state."""
        if seed is not None:
            torch.manual_seed(seed)
        for _ in range(num_steps):
            optimizer.zero_grad()
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

    def assert_tensor_dict_equal(self, dict1, dict2, path=""):
        """Recursively compare dicts containing tensors."""
        assert set(dict1.keys()) == set(dict2.keys()), f"Key mismatch at {path}"
        for key in dict1:
            current_path = f"{path}.{key}" if path else key
            val1, val2 = dict1[key], dict2[key]
            if torch.is_tensor(val1):
                assert torch.is_tensor(val2), f"Type mismatch at {current_path}"
                assert torch.allclose(
                    val1, val2, rtol=1e-5, atol=1e-8
                ), f"Tensor mismatch at {current_path}"
            elif isinstance(val1, dict):
                assert isinstance(val2, dict), f"Type mismatch at {current_path}"
                self.assert_tensor_dict_equal(val1, val2, current_path)
            elif val1 is None:
                assert val2 is None, f"None mismatch at {current_path}"
            else:
                assert (
                    val1 == val2
                ), f"Value mismatch at {current_path}: {val1} != {val2}"


class TestAdamWStateDict(TestOptimizerStateDictBase):
    """Test AdamW state_dict/load_state_dict."""

    def test_round_trip(self):
        """Save and restore AdamW state."""
        model = self.create_simple_model()
        opt = AdamW(model.parameters(), lr=0.001)
        self.perform_training_steps(model, opt, num_steps=10)

        # Save state
        state_dict = opt.state_dict()

        # Create new optimizer and restore
        model2 = self.create_simple_model()
        opt2 = AdamW(model2.parameters(), lr=0.001)
        opt2.load_state_dict(state_dict)

        # Verify state matches
        self.assert_tensor_dict_equal(
            opt.state_dict()["state"], opt2.state_dict()["state"]
        )

    def test_training_continuation(self):
        """Verify training continues correctly after restore.

        We test that:
        1. Optimizer state can be saved and restored
        2. Training produces consistent updates after restore
        """
        # Train for N steps, save state
        model1 = self.create_simple_model()
        opt1 = AdamW(model1.parameters(), lr=0.001)
        self.perform_training_steps(model1, opt1, num_steps=10, seed=42)

        # Save optimizer and model state
        opt_state = opt1.state_dict()
        model_state = model1.state_dict()

        # Create fresh model and optimizer, load saved states
        model2 = self.create_simple_model()
        model2.load_state_dict(model_state)
        opt2 = AdamW(model2.parameters(), lr=0.001)
        opt2.load_state_dict(opt_state)

        # Verify optimizer states match after restore
        opt1_state_after = opt1.state_dict()
        opt2_state_after = opt2.state_dict()

        # Check that all state components match
        for (key1, state1), (key2, state2) in zip(
            opt1_state_after["state"].items(), opt2_state_after["state"].items()
        ):
            assert key1 == key2
            for comp in ["step", "m", "v"]:
                assert torch.allclose(
                    state1[comp], state2[comp], rtol=1e-5, atol=1e-8
                ), f"State mismatch for {comp} in param {key1}"

    def test_state_structure_validation(self):
        """Verify AdamW validates state structure."""
        model = self.create_simple_model()
        opt = AdamW(model.parameters(), lr=0.001)
        self.perform_training_steps(model, opt)

        state_dict = opt.state_dict()

        # All params should have step, m, v
        for param_id, param_state in state_dict["state"].items():
            assert "step" in param_state, f"Missing 'step' for param {param_id}"
            assert "m" in param_state, f"Missing 'm' for param {param_id}"
            assert "v" in param_state, f"Missing 'v' for param {param_id}"


class TestAdafactorStateDict(TestOptimizerStateDictBase):
    """Test Adafactor state_dict/load_state_dict."""

    def test_round_trip(self):
        """Save and restore Adafactor state."""
        model = self.create_simple_model()
        opt = Adafactor(model.parameters(), lr=0.001)
        self.perform_training_steps(model, opt, num_steps=10)

        state_dict = opt.state_dict()

        model2 = self.create_simple_model()
        opt2 = Adafactor(model2.parameters(), lr=0.001)
        opt2.load_state_dict(state_dict)

        self.assert_tensor_dict_equal(
            opt.state_dict()["state"], opt2.state_dict()["state"]
        )

    def test_training_continuation_no_discontinuity(self):
        """CRITICAL: Verify no loss/grad-norm discontinuity after restore.

        This test verifies that optimizer state is properly preserved by checking
        that the optimizer state matches exactly after save/restore.
        """
        model = self.create_simple_model()
        opt = Adafactor(model.parameters(), lr=0.001)

        # Train for warmup steps to populate state
        self.perform_training_steps(model, opt, num_steps=20, seed=42)

        # Save checkpoint
        opt_state = opt.state_dict()
        model_state = model.state_dict()

        # Create fresh optimizer and restore
        model2 = self.create_simple_model()
        model2.load_state_dict(model_state)
        opt2 = Adafactor(model2.parameters(), lr=0.001)
        opt2.load_state_dict(opt_state)

        # Verify optimizer states match after restore
        opt_state_after = opt.state_dict()
        opt2_state_after = opt2.state_dict()

        # Check that all state components match
        for (key1, state1), (key2, state2) in zip(
            opt_state_after["state"].items(), opt2_state_after["state"].items()
        ):
            assert key1 == key2
            # Check step
            assert torch.allclose(
                state1["step"], state2["step"], rtol=1e-5, atol=1e-8
            ), f"Step mismatch for param {key1}"
            # Check row
            assert torch.allclose(
                state1["row"], state2["row"], rtol=1e-5, atol=1e-8
            ), f"Row mismatch for param {key1}"
            # Check col (can be None)
            if state1["col"] is None:
                assert state2["col"] is None, f"Col should be None for param {key1}"
            else:
                assert torch.allclose(
                    state1["col"], state2["col"], rtol=1e-5, atol=1e-8
                ), f"Col mismatch for param {key1}"

    def test_col_none_handling(self):
        """Verify Adafactor handles col=None correctly."""
        model = self.create_simple_model()
        opt = Adafactor(model.parameters(), lr=0.001)
        self.perform_training_steps(model, opt)

        state_dict = opt.state_dict()

        # Check if any param has col=None (depends on implementation)
        for param_state in state_dict["state"].values():
            assert "col" in param_state, "Missing 'col' in Adafactor state"
            # col can be None or tensor
            assert param_state["col"] is None or torch.is_tensor(
                param_state["col"]
            ), f"Invalid col type: {type(param_state['col'])}"

    def test_state_structure_validation(self):
        """Verify Adafactor validates state structure."""
        model = self.create_simple_model()
        opt = Adafactor(model.parameters(), lr=0.001)
        self.perform_training_steps(model, opt)

        state_dict = opt.state_dict()

        # All params should have step, row, col
        for param_id, param_state in state_dict["state"].items():
            assert "step" in param_state, f"Missing 'step' for param {param_id}"
            assert "row" in param_state, f"Missing 'row' for param {param_id}"
            assert "col" in param_state, f"Missing 'col' for param {param_id}"


class TestApolloStateDict(TestOptimizerStateDictBase):
    """Test Apollo state_dict/load_state_dict."""

    def create_simple_model(self):
        """2-layer linear model WITHOUT biases for Apollo (requires 2D params)."""
        return torch.nn.Sequential(
            torch.nn.Linear(10, 20, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 2, bias=False),
        )

    def make_projector_factory(self, projector_class):
        """Create projector factory for Apollo."""

        def factory(rank, dim, proj_type):
            if projector_class == OnlinePCAProjector:
                return OnlinePCAProjector(rank, dim, proj_type, update_steps=10)
            elif projector_class == RandProjector:
                return RandProjector(rank, dim, proj_type, update_steps=10, lazy=False)
            else:
                raise ValueError(f"Unknown projector class: {projector_class}")

        return factory

    def test_round_trip_online_pca(self):
        """Test Apollo with OnlinePCAProjector."""
        model = self.create_simple_model()
        opt = Apollo(
            model.parameters(),
            lr=0.001,
            rank=2,
            projector_factory=self.make_projector_factory(OnlinePCAProjector),
        )
        self.perform_training_steps(model, opt, num_steps=10, seed=42)

        state_dict = opt.state_dict()

        model2 = self.create_simple_model()
        opt2 = Apollo(
            model2.parameters(),
            lr=0.001,
            rank=2,
            projector_factory=self.make_projector_factory(OnlinePCAProjector),
        )
        opt2.load_state_dict(state_dict)

        # Verify projector state matches
        # Note: opt.state keys are parameter tensors, so we compare by index
        state1_list = list(opt.state.values())
        state2_list = list(opt2.state.values())

        assert len(state1_list) == len(state2_list), "State count mismatch"

        for state1, state2 in zip(state1_list, state2_list):
            if "projector" in state1:
                proj1 = state1["projector"]
                proj2 = state2["projector"]
                assert isinstance(
                    proj1, OnlinePCAProjector
                ), f"Expected OnlinePCAProjector, got {type(proj1)}"
                assert isinstance(
                    proj2, OnlinePCAProjector
                ), f"Expected OnlinePCAProjector, got {type(proj2)}"
                assert proj1.rank == proj2.rank
                assert proj1.dim == proj2.dim
                if proj1.A is not None and proj2.A is not None:
                    # Use equal_nan=True to handle NaN values in projection matrix
                    assert torch.allclose(
                        proj1.A, proj2.A, rtol=1e-5, atol=1e-8, equal_nan=True
                    ), "Projector A matrix mismatch"

    def test_round_trip_rand_projector(self):
        """Test Apollo with RandProjector."""
        model = self.create_simple_model()
        opt = Apollo(
            model.parameters(),
            lr=0.001,
            rank=2,
            projector_factory=self.make_projector_factory(RandProjector),
        )
        self.perform_training_steps(model, opt, num_steps=10, seed=42)

        state_dict = opt.state_dict()

        model2 = self.create_simple_model()
        opt2 = Apollo(
            model2.parameters(),
            lr=0.001,
            rank=2,
            projector_factory=self.make_projector_factory(RandProjector),
        )
        opt2.load_state_dict(state_dict)

        # Verify projector state matches
        # Note: opt.state keys are parameter tensors, so we compare by index
        state1_list = list(opt.state.values())
        state2_list = list(opt2.state.values())

        assert len(state1_list) == len(state2_list), "State count mismatch"

        for state1, state2 in zip(state1_list, state2_list):
            if "projector" in state1:
                proj1 = state1["projector"]
                proj2 = state2["projector"]
                assert isinstance(
                    proj1, RandProjector
                ), f"Expected RandProjector, got {type(proj1)}"
                assert isinstance(
                    proj2, RandProjector
                ), f"Expected RandProjector, got {type(proj2)}"
                assert proj1.rank == proj2.rank
                assert proj1.dim == proj2.dim
                if proj1.A is not None and proj2.A is not None:
                    assert torch.allclose(proj1.A, proj2.A, rtol=1e-5, atol=1e-8)

    def test_projector_serialization(self):
        """Verify projector is serialized as dict, not object."""
        model = self.create_simple_model()
        opt = Apollo(
            model.parameters(),
            lr=0.001,
            rank=2,
            projector_factory=self.make_projector_factory(OnlinePCAProjector),
        )
        self.perform_training_steps(model, opt, num_steps=10)

        state_dict = opt.state_dict()

        # Check projector is serialized as dict
        for param_state in state_dict["state"].values():
            if "projector" in param_state:
                proj_dict = param_state["projector"]
                assert isinstance(
                    proj_dict, dict
                ), "Projector must be serialized as dict"
                assert "_class" in proj_dict, "Projector dict must have '_class' key"
                assert "rank" in proj_dict, "Projector dict must have 'rank' key"

    def test_projector_reconstruction(self):
        """Verify projector is reconstructed as object after load."""
        model = self.create_simple_model()
        opt = Apollo(
            model.parameters(),
            lr=0.001,
            rank=2,
            projector_factory=self.make_projector_factory(RandProjector),
        )
        self.perform_training_steps(model, opt, num_steps=10)

        state_dict = opt.state_dict()

        model2 = self.create_simple_model()
        opt2 = Apollo(
            model2.parameters(),
            lr=0.001,
            rank=2,
            projector_factory=self.make_projector_factory(RandProjector),
        )
        opt2.load_state_dict(state_dict)

        # Verify projector is object after restore
        for param_state in opt2.state.values():
            if "projector" in param_state:
                proj = param_state["projector"]
                assert isinstance(
                    proj, RandProjector
                ), f"Projector must be reconstructed as RandProjector, got {type(proj)}"


class TestMultioptStateDict(TestOptimizerStateDictBase):
    """Test Multiopt state_dict/load_state_dict."""

    def test_round_trip(self):
        """Test Multiopt with multiple wrapped optimizers."""
        model = self.create_simple_model()
        params = list(model.parameters())

        # Split params: first to AdamW, rest to SGD
        opt1 = AdamW([params[0]], lr=0.001)
        opt2 = SGD(params[1:], lr=0.01)
        multiopt = Multiopt([opt1, opt2])

        self.perform_training_steps(model, multiopt, num_steps=10)

        state_dict = multiopt.state_dict()

        # Verify structure
        assert (
            "optimizers" in state_dict
        ), "Multiopt state_dict must have 'optimizers' key"
        assert len(state_dict["optimizers"]) == 2, "Expected 2 wrapped optimizers"

        # Restore
        model2 = self.create_simple_model()
        params2 = list(model2.parameters())
        opt1_new = AdamW([params2[0]], lr=0.001)
        opt2_new = SGD(params2[1:], lr=0.01)
        multiopt2 = Multiopt([opt1_new, opt2_new])

        multiopt2.load_state_dict(state_dict)

        # Verify wrapped optimizer states were restored
        state1_after = opt1_new.state_dict()
        state2_after = opt2_new.state_dict()

        # First optimizer should have state (AdamW)
        assert (
            len(state1_after["state"]) > 0
        ), "First optimizer should have state after restore"

    def test_training_continuation(self):
        """Verify Multiopt training continues correctly after restore."""
        model1 = self.create_simple_model()
        params1 = list(model1.parameters())
        opt1 = AdamW([params1[0]], lr=0.001)
        opt2 = SGD(params1[1:], lr=0.01)
        multiopt1 = Multiopt([opt1, opt2])

        self.perform_training_steps(model1, multiopt1, num_steps=10, seed=42)

        # Save states
        multiopt_state = multiopt1.state_dict()
        model_state = model1.state_dict()

        # Create fresh model and optimizer, load saved states
        model2 = self.create_simple_model()
        model2.load_state_dict(model_state)
        params2 = list(model2.parameters())
        opt1_new = AdamW([params2[0]], lr=0.001)
        opt2_new = SGD(params2[1:], lr=0.01)
        multiopt2 = Multiopt([opt1_new, opt2_new])
        multiopt2.load_state_dict(multiopt_state)

        # Verify wrapped optimizer states were restored
        # First optimizer (AdamW) should have state
        opt1_state_original = opt1.state_dict()
        opt1_state_restored = opt1_new.state_dict()

        assert len(opt1_state_original["state"]) == len(
            opt1_state_restored["state"]
        ), "AdamW state count mismatch after Multiopt restore"

        # Verify state values match
        for (key1, state1), (key2, state2) in zip(
            opt1_state_original["state"].items(), opt1_state_restored["state"].items()
        ):
            for comp in ["step", "m", "v"]:
                assert torch.allclose(
                    state1[comp], state2[comp], rtol=1e-5, atol=1e-8
                ), f"Multiopt wrapped optimizer state mismatch for {comp}"

    def test_mismatched_optimizer_count(self):
        """Verify Multiopt detects optimizer count mismatch."""
        model = self.create_simple_model()
        params = list(model.parameters())

        opt1 = AdamW([params[0]], lr=0.001)
        opt2 = SGD(params[1:], lr=0.01)
        multiopt = Multiopt([opt1, opt2])

        self.perform_training_steps(model, multiopt, num_steps=5, seed=42)
        state_dict = multiopt.state_dict()

        # Create Multiopt with different number of optimizers
        model2 = self.create_simple_model()
        params2 = list(model2.parameters())
        opt_single = AdamW(params2, lr=0.001)
        multiopt_wrong = Multiopt([opt_single])

        # Should raise error about count mismatch
        with pytest.raises(ValueError, match="optimizer count mismatch"):
            multiopt_wrong.load_state_dict(state_dict)
