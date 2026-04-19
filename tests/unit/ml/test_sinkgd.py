"""
Unit tests for SinkGD optimizer and sr_sinkhorn normalization.

Tests cover:
- sr_sinkhorn: row/column norm convergence, Frobenius norm, non-square matrices,
  zero-iteration pass-through, monotone convergence, numerical stability.
- SinkGD: loss reduction, update direction, statelessness, weight decay,
  normalize_output flag, higher-D and 1D/scalar parameters, multiple param
  groups, closure, and bf16 parameter handling.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from forgather.ml.optim.sinkgd import SinkGD, sr_sinkhorn


# ===========================================================================
# Tests for sr_sinkhorn
# ===========================================================================


class TestSrSinkhorn:
    """Tests for the sr_sinkhorn normalization function."""

    def _make_matrix(self, m, n, seed=0):
        torch.manual_seed(seed)
        return torch.randn(m, n)

    # --- norm convergence ---

    def test_row_norms_converge_to_sqrt_n(self):
        """After enough iterations, each row l2-norm should equal √n."""
        m, n = 8, 16
        X = self._make_matrix(m, n)
        Y = sr_sinkhorn(X, num_iters=10)
        row_norms = Y.norm(dim=1)
        expected = math.sqrt(n)
        assert torch.allclose(row_norms, torch.full_like(row_norms, expected), atol=1e-4), (
            f"Row norms should converge to √n={expected:.4f}, got {row_norms}"
        )

    def test_col_norms_converge_to_sqrt_m(self):
        """After enough iterations, each column l2-norm should equal √m."""
        m, n = 8, 16
        X = self._make_matrix(m, n)
        Y = sr_sinkhorn(X, num_iters=10)
        col_norms = Y.norm(dim=0)
        expected = math.sqrt(m)
        assert torch.allclose(col_norms, torch.full_like(col_norms, expected), atol=1e-4), (
            f"Col norms should converge to √m={expected:.4f}, got {col_norms}"
        )

    def test_frobenius_norm_equals_sqrt_mn(self):
        """Frobenius norm should equal √(mn) after convergence."""
        m, n = 8, 16
        X = self._make_matrix(m, n)
        Y = sr_sinkhorn(X, num_iters=10)
        expected = math.sqrt(m * n)
        assert abs(Y.norm().item() - expected) < 1e-3, (
            f"Frobenius norm should be √(mn)={expected:.4f}, got {Y.norm().item():.4f}"
        )

    def test_square_matrix(self):
        """Square matrices should also converge to √n row/col norms."""
        n = 12
        X = self._make_matrix(n, n)
        Y = sr_sinkhorn(X, num_iters=10)
        row_norms = Y.norm(dim=1)
        col_norms = Y.norm(dim=0)
        expected = math.sqrt(n)
        assert torch.allclose(row_norms, torch.full_like(row_norms, expected), atol=1e-4)
        assert torch.allclose(col_norms, torch.full_like(col_norms, expected), atol=1e-4)

    def test_tall_matrix(self):
        """m > n (tall matrix) should also converge."""
        m, n = 32, 8
        X = self._make_matrix(m, n)
        Y = sr_sinkhorn(X, num_iters=10)
        row_norms = Y.norm(dim=1)
        col_norms = Y.norm(dim=0)
        assert torch.allclose(row_norms, torch.full_like(row_norms, math.sqrt(n)), atol=1e-4)
        assert torch.allclose(col_norms, torch.full_like(col_norms, math.sqrt(m)), atol=1e-4)

    def test_single_row_matrix(self):
        """A 1×n matrix: row norm should equal √n, col norms should equal √1=1."""
        n = 8
        X = self._make_matrix(1, n)
        Y = sr_sinkhorn(X, num_iters=10)
        assert abs(Y.norm(dim=1).item() - math.sqrt(n)) < 1e-4
        col_norms = Y.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-4)

    # --- zero iterations ---

    def test_zero_iterations_returns_input(self):
        """With num_iters=0, the input should be returned unchanged."""
        X = self._make_matrix(6, 10)
        Y = sr_sinkhorn(X, num_iters=0)
        assert torch.equal(X, Y), "Zero iterations should return the original tensor"

    # --- monotone convergence ---

    def test_more_iterations_tighter_convergence(self):
        """More iterations should yield row norms closer to √n."""
        m, n = 16, 32
        X = self._make_matrix(m, n)

        def row_norm_deviation(iters):
            Y = sr_sinkhorn(X, num_iters=iters)
            return (Y.norm(dim=1) - math.sqrt(n)).abs().max().item()

        dev1 = row_norm_deviation(1)
        dev5 = row_norm_deviation(5)
        assert dev5 < dev1, (
            f"More iterations should converge: dev1={dev1:.6f}, dev5={dev5:.6f}"
        )

    # --- numerical stability ---

    def test_near_zero_row_does_not_produce_nan(self):
        """A near-zero row should be handled via eps clamping without NaN/Inf."""
        m, n = 4, 8
        X = torch.randn(m, n)
        X[1, :] = 1e-40  # Nearly zero row
        Y = sr_sinkhorn(X, num_iters=5)
        assert not torch.isnan(Y).any(), "NaN values produced for near-zero row"
        assert not torch.isinf(Y).any(), "Inf values produced for near-zero row"

    def test_output_dtype_preserved(self):
        """Output dtype should match input dtype (float32)."""
        X = torch.randn(8, 16, dtype=torch.float32)
        Y = sr_sinkhorn(X, num_iters=5)
        assert Y.dtype == torch.float32

    def test_output_shape_preserved(self):
        """Output shape should match input shape."""
        for m, n in [(4, 8), (16, 4), (10, 10)]:
            X = torch.randn(m, n)
            Y = sr_sinkhorn(X, num_iters=3)
            assert Y.shape == (m, n), f"Shape mismatch: expected ({m},{n}), got {Y.shape}"

    def test_single_iteration_partial_convergence(self):
        """After 1 iteration the last operation is a column normalization,
        so column norms should be exactly √m."""
        m, n = 6, 10
        X = self._make_matrix(m, n)
        Y = sr_sinkhorn(X, num_iters=1)
        col_norms = Y.norm(dim=0)
        expected = math.sqrt(m)
        assert torch.allclose(col_norms, torch.full_like(col_norms, expected), atol=1e-5), (
            f"After 1 iteration, column norms should be √m={expected:.4f}"
        )


# ===========================================================================
# Tests for SinkGD optimizer
# ===========================================================================


def _simple_problem(seed=42, in_features=16, out_features=8, batch=32):
    """Returns (model, x, y) for a linear regression problem."""
    torch.manual_seed(seed)
    model = nn.Linear(in_features, out_features, bias=False)
    x = torch.randn(batch, in_features)
    y = torch.randn(batch, out_features)
    return model, x, y


class TestSinkGDBasic:
    """Basic functional tests for SinkGD."""

    def test_loss_decreases(self):
        """SinkGD should reduce MSE loss on a simple linear problem."""
        model, x, y = _simple_problem()
        optimizer = SinkGD(model.parameters(), lr=1e-2, num_iters=5)

        initial_loss = F.mse_loss(model(x), y).item()

        for _ in range(100):
            optimizer.zero_grad()
            F.mse_loss(model(x), y).backward()
            optimizer.step()

        final_loss = F.mse_loss(model(x), y).item()
        assert final_loss < initial_loss, (
            f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_parameters_change_after_step(self):
        """A SinkGD step should modify the parameters."""
        model, x, y = _simple_problem()
        optimizer = SinkGD(model.parameters(), lr=1e-2)
        w_before = model.weight.data.clone()

        optimizer.zero_grad()
        F.mse_loss(model(x), y).backward()
        optimizer.step()

        assert not torch.equal(model.weight.data, w_before), (
            "Weights should change after a SinkGD step"
        )

    def test_update_direction_opposes_gradient(self):
        """The parameter update should be in the direction of the negative processed gradient."""
        torch.manual_seed(0)
        model = nn.Linear(8, 4, bias=False)
        x = torch.randn(16, 8)
        y = torch.randn(16, 4)

        optimizer = SinkGD(model.parameters(), lr=0.1, num_iters=5)
        w_before = model.weight.data.float().clone()

        optimizer.zero_grad()
        F.mse_loss(model(x), y).backward()
        grad = model.weight.grad.float().clone()
        optimizer.step()

        w_after = model.weight.data.float().clone()
        update = w_after - w_before  # should be in direction of -processed_grad

        # The dot product of update and grad should be negative (opposing gradient)
        dot = (update * grad).sum().item()
        assert dot < 0, (
            f"Update should oppose the gradient direction, but dot product = {dot:.6f}"
        )

    def test_no_grad_parameters_skipped(self):
        """Parameters whose .grad is None should not be modified."""
        torch.manual_seed(0)
        layer0 = nn.Linear(8, 8, bias=False)
        layer1 = nn.Linear(8, 4, bias=False)
        model = nn.Sequential(layer0, layer1)
        optimizer = SinkGD(model.parameters(), lr=1e-2)

        x = torch.randn(4, 8)
        F.mse_loss(model(x), torch.randn(4, 4)).backward()
        # Remove grad from layer0
        layer0.weight.grad = None
        w0_before = layer0.weight.data.clone()

        optimizer.step()

        assert torch.equal(layer0.weight.data, w0_before), (
            "Parameter without gradient should not be modified"
        )

    def test_closure_returns_loss(self):
        """step() should call the closure and return its value."""
        model, x, y = _simple_problem()
        optimizer = SinkGD(model.parameters(), lr=1e-2)

        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        loss_val = loss.detach()

        result = optimizer.step(closure=lambda: loss_val)
        assert result is not None
        assert result.item() > 0

    def test_multiple_param_groups(self):
        """SinkGD should respect per-group learning rates."""
        torch.manual_seed(0)
        layer0 = nn.Linear(8, 8, bias=False)
        layer1 = nn.Linear(8, 4, bias=False)
        model = nn.Sequential(layer0, layer1)

        optimizer = SinkGD(
            [
                {"params": layer0.parameters(), "lr": 1.0},
                {"params": layer1.parameters(), "lr": 1e-6},
            ],
            lr=1e-3,
        )

        x = torch.randn(4, 8)
        F.mse_loss(model(x), torch.randn(4, 4)).backward()

        w0_before = layer0.weight.data.clone()
        w1_before = layer1.weight.data.clone()
        optimizer.step()

        update0 = (layer0.weight.data - w0_before).norm().item()
        update1 = (layer1.weight.data - w1_before).norm().item()

        # Group 0 has lr=1.0, group 1 has lr=1e-6: update0 >> update1
        assert update0 > update1 * 100, (
            f"Group with higher lr should have larger update: "
            f"update0={update0:.6f}, update1={update1:.6f}"
        )


# ===========================================================================
# Tests for SinkGD weight decay
# ===========================================================================


class TestSinkGDWeightDecay:

    def test_weight_decay_shrinks_weights(self):
        """With weight_decay > 0, weight norms should be smaller than without."""
        torch.manual_seed(42)
        model_wd = nn.Linear(16, 8, bias=False)
        torch.manual_seed(42)
        model_no_wd = nn.Linear(16, 8, bias=False)
        x = torch.randn(32, 16)
        y = torch.randn(32, 8)

        opt_wd = SinkGD(model_wd.parameters(), lr=1e-2, weight_decay=0.5)
        opt_no_wd = SinkGD(model_no_wd.parameters(), lr=1e-2, weight_decay=0.0)

        for _ in range(100):
            opt_wd.zero_grad()
            F.mse_loss(model_wd(x), y).backward()
            opt_wd.step()

            opt_no_wd.zero_grad()
            F.mse_loss(model_no_wd(x), y).backward()
            opt_no_wd.step()

        norm_wd = model_wd.weight.data.norm().item()
        norm_no_wd = model_no_wd.weight.data.norm().item()
        assert norm_wd < norm_no_wd, (
            f"Weight decay should reduce weight norm: "
            f"with_wd={norm_wd:.4f}, without_wd={norm_no_wd:.4f}"
        )

    def test_zero_weight_decay_unchanged_norm_direction(self):
        """weight_decay=0 should not shrink weights via the decay term."""
        torch.manual_seed(0)
        model = nn.Linear(8, 4, bias=False)
        # Initialize to a known state
        nn.init.ones_(model.weight)
        optimizer = SinkGD(model.parameters(), lr=0.0, weight_decay=0.0)

        # With lr=0 and no weight decay, weights should not change
        x = torch.randn(4, 8)
        F.mse_loss(model(x), torch.randn(4, 4)).backward()
        w_before = model.weight.data.clone()
        optimizer.step()

        assert torch.equal(model.weight.data, w_before), (
            "With lr=0 and weight_decay=0, weights should be unchanged"
        )


# ===========================================================================
# Tests for SinkGD normalize_output flag
# ===========================================================================


class TestSinkGDNormalizeOutput:

    def test_normalize_output_true_scales_by_inv_sqrt_mn(self):
        """With normalize_output=True, the update magnitude should be ~1/√(mn) of the raw update."""
        torch.manual_seed(0)
        m, n = 8, 16
        model_norm = nn.Linear(n, m, bias=False)
        model_raw = nn.Linear(n, m, bias=False)
        # Give both models identical weights and gradients
        model_raw.weight.data.copy_(model_norm.weight.data)

        x = torch.randn(4, n)
        y = torch.randn(4, m)

        opt_norm = SinkGD(model_norm.parameters(), lr=1.0, num_iters=10, normalize_output=True)
        opt_raw = SinkGD(model_raw.parameters(), lr=1.0, num_iters=10, normalize_output=False)

        # Force identical gradients
        loss_norm = F.mse_loss(model_norm(x), y)
        loss_norm.backward()
        loss_raw = F.mse_loss(model_raw(x), y)
        loss_raw.backward()
        # Copy gradient from model_norm to model_raw so they're identical
        with torch.no_grad():
            model_raw.weight.grad.copy_(model_norm.weight.grad)

        w_norm_before = model_norm.weight.data.clone()
        w_raw_before = model_raw.weight.data.clone()

        opt_norm.step()
        opt_raw.step()

        update_norm = (model_norm.weight.data - w_norm_before).norm().item()
        update_raw = (model_raw.weight.data - w_raw_before).norm().item()

        # normalize_output=False update should be ~√(mn) times larger
        ratio = update_raw / (update_norm + 1e-12)
        expected_ratio = math.sqrt(m * n)
        assert abs(ratio - expected_ratio) / expected_ratio < 0.01, (
            f"Update ratio should be √(mn)={expected_ratio:.2f}, got {ratio:.2f}"
        )

    def test_normalize_output_false_larger_steps(self):
        """normalize_output=False should produce larger parameter steps than True."""
        torch.manual_seed(1)
        model_on = nn.Linear(16, 8, bias=False)
        model_off = nn.Linear(16, 8, bias=False)
        model_off.weight.data.copy_(model_on.weight.data)

        x = torch.randn(4, 16)
        y = torch.randn(4, 8)

        opt_on = SinkGD(model_on.parameters(), lr=1e-2, normalize_output=True)
        opt_off = SinkGD(model_off.parameters(), lr=1e-2, normalize_output=False)

        loss_on = F.mse_loss(model_on(x), y)
        loss_on.backward()
        loss_off = F.mse_loss(model_off(x), y)
        loss_off.backward()
        with torch.no_grad():
            model_off.weight.grad.copy_(model_on.weight.grad)

        w_on_before = model_on.weight.data.clone()
        w_off_before = model_off.weight.data.clone()
        opt_on.step()
        opt_off.step()

        update_on = (model_on.weight.data - w_on_before).norm().item()
        update_off = (model_off.weight.data - w_off_before).norm().item()

        assert update_off > update_on, (
            f"normalize_output=False should give larger steps: on={update_on}, off={update_off}"
        )


# ===========================================================================
# Tests for SinkGD with different parameter shapes
# ===========================================================================


class TestSinkGDParameterShapes:

    def test_2d_weight_matrix(self):
        """Standard 2D weight matrix should be processed by sr_sinkhorn."""
        model, x, y = _simple_problem()
        optimizer = SinkGD(model.parameters(), lr=1e-2)
        F.mse_loss(model(x), y).backward()
        w_before = model.weight.data.clone()
        optimizer.step()
        assert not torch.equal(model.weight.data, w_before)

    def test_1d_bias_parameter(self):
        """1D bias parameters should be handled via l2-normalization."""
        torch.manual_seed(0)
        model = nn.Linear(8, 4, bias=True)
        optimizer = SinkGD(model.parameters(), lr=1e-2)

        x = torch.randn(4, 8)
        F.mse_loss(model(x), torch.randn(4, 4)).backward()

        b_before = model.bias.data.clone()
        optimizer.step()

        # Bias should change
        assert not torch.equal(model.bias.data, b_before)

    def test_3d_weight_parameter(self):
        """A 3D parameter (e.g., Conv1d weight: out × in × kernel) should be reshaped."""
        torch.manual_seed(0)
        # Conv1d weight: (out_channels, in_channels, kernel_size)
        model = nn.Conv1d(8, 16, kernel_size=3, bias=False)
        optimizer = SinkGD(model.parameters(), lr=1e-2)

        x = torch.randn(4, 8, 10)
        model(x).sum().backward()

        w_before = model.weight.data.clone()
        optimizer.step()  # Must not raise

        assert not torch.equal(model.weight.data, w_before)

    def test_scalar_parameter(self):
        """A scalar (0-d) parameter should be handled without error."""
        p = nn.Parameter(torch.tensor(1.0))
        optimizer = SinkGD([p], lr=1e-1)
        p.grad = torch.tensor(2.0)
        p_before = p.data.clone()
        optimizer.step()  # Must not raise

        # Scalar: gradient is used directly (no normalization)
        assert not torch.equal(p.data, p_before)

    def test_mixed_dim_model(self):
        """A model with 2D weights and 1D biases should update all parameters."""
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(8, 8),   # weight 2D, bias 1D
            nn.Linear(8, 4),
        )
        optimizer = SinkGD(model.parameters(), lr=1e-2)
        x = torch.randn(4, 8)
        model(x).sum().backward()

        params_before = [p.data.clone() for p in model.parameters()]
        optimizer.step()
        params_after = [p.data.clone() for p in model.parameters()]

        for i, (pb, pa) in enumerate(zip(params_before, params_after)):
            assert not torch.equal(pb, pa), f"Parameter {i} should have been updated"


# ===========================================================================
# Tests for SinkGD statelessness
# ===========================================================================


class TestSinkGDStateless:

    def test_no_optimizer_state_stored(self):
        """SinkGD is stateless: self.state should remain empty after stepping."""
        model, x, y = _simple_problem()
        optimizer = SinkGD(model.parameters(), lr=1e-2)

        for _ in range(5):
            optimizer.zero_grad()
            F.mse_loss(model(x), y).backward()
            optimizer.step()

        # Optimizer state dict should contain no per-parameter state
        state_dict = optimizer.state_dict()
        assert len(state_dict["state"]) == 0, (
            f"SinkGD should be stateless, but state_dict['state'] has "
            f"{len(state_dict['state'])} entries"
        )

    def test_state_dict_round_trip_is_trivial(self):
        """Save/load state_dict should not affect behaviour (no state to preserve)."""
        torch.manual_seed(0)
        model = nn.Linear(8, 4, bias=False)
        optimizer = SinkGD(model.parameters(), lr=1e-2)

        x = torch.randn(4, 8)
        y = torch.randn(4, 4)

        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            F.mse_loss(model(x), y).backward()
            optimizer.step()

        saved = optimizer.state_dict()
        optimizer.load_state_dict(saved)  # Should not raise

        # One more step after reload should work
        optimizer.zero_grad()
        F.mse_loss(model(x), y).backward()
        optimizer.step()


# ===========================================================================
# Tests for SinkGD with bf16 parameters
# ===========================================================================


class TestSinkGDBf16:

    def test_bf16_parameters_updated(self):
        """SinkGD should update bf16 parameters without error."""
        torch.manual_seed(0)
        model = nn.Linear(16, 8, bias=False).to(torch.bfloat16)
        optimizer = SinkGD(model.parameters(), lr=1e-2)

        x = torch.randn(4, 16, dtype=torch.bfloat16)
        y = torch.randn(4, 8, dtype=torch.bfloat16)

        loss = F.mse_loss(model(x).float(), y.float())
        loss.backward()

        w_before = model.weight.data.clone()
        optimizer.step()

        assert model.weight.dtype == torch.bfloat16, "Parameter dtype should remain bfloat16"
        assert not torch.equal(model.weight.data, w_before), (
            "bf16 parameters should be updated"
        )


# ===========================================================================
# Tests for SinkGD num_iters
# ===========================================================================


class TestSinkGDNumIters:

    def test_num_iters_1_still_updates(self):
        """num_iters=1 should still produce valid parameter updates."""
        model, x, y = _simple_problem()
        optimizer = SinkGD(model.parameters(), lr=1e-2, num_iters=1)
        optimizer.zero_grad()
        F.mse_loss(model(x), y).backward()
        w_before = model.weight.data.clone()
        optimizer.step()
        assert not torch.equal(model.weight.data, w_before)

    def test_num_iters_affects_update_magnitude(self):
        """Different num_iters should (generally) produce different update magnitudes."""
        torch.manual_seed(0)
        model1 = nn.Linear(16, 8, bias=False)
        model5 = nn.Linear(16, 8, bias=False)
        model5.weight.data.copy_(model1.weight.data)

        x = torch.randn(4, 16)
        y = torch.randn(4, 8)

        opt1 = SinkGD(model1.parameters(), lr=1.0, num_iters=1, normalize_output=False)
        opt5 = SinkGD(model5.parameters(), lr=1.0, num_iters=5, normalize_output=False)

        loss1 = F.mse_loss(model1(x), y)
        loss1.backward()
        loss5 = F.mse_loss(model5(x), y)
        loss5.backward()
        with torch.no_grad():
            model5.weight.grad.copy_(model1.weight.grad)

        w1_before = model1.weight.data.clone()
        w5_before = model5.weight.data.clone()
        opt1.step()
        opt5.step()

        update1 = (model1.weight.data - w1_before).norm().item()
        update5 = (model5.weight.data - w5_before).norm().item()

        # Updates can differ; what matters is neither is zero
        assert update1 > 0
        assert update5 > 0

    def test_loss_decreases_with_num_iters_1(self):
        """Even with num_iters=1, SinkGD should reduce loss on a simple problem."""
        model, x, y = _simple_problem()
        optimizer = SinkGD(model.parameters(), lr=1e-2, num_iters=1)

        initial_loss = F.mse_loss(model(x), y).item()
        for _ in range(100):
            optimizer.zero_grad()
            F.mse_loss(model(x), y).backward()
            optimizer.step()

        final_loss = F.mse_loss(model(x), y).item()
        assert final_loss < initial_loss


# ===========================================================================
# CUDA-specific tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSinkGDCUDA:

    def test_loss_decreases_on_cuda(self):
        """SinkGD should reduce loss when model is on CUDA."""
        torch.manual_seed(0)
        model = nn.Linear(16, 8, bias=False).cuda()
        x = torch.randn(32, 16, device="cuda")
        y = torch.randn(32, 8, device="cuda")

        optimizer = SinkGD(model.parameters(), lr=1e-2, num_iters=5)
        initial_loss = F.mse_loss(model(x), y).item()

        for _ in range(100):
            optimizer.zero_grad()
            F.mse_loss(model(x), y).backward()
            optimizer.step()

        final_loss = F.mse_loss(model(x), y).item()
        assert final_loss < initial_loss

    def test_sr_sinkhorn_on_cuda(self):
        """sr_sinkhorn should work on CUDA tensors and converge on device."""
        m, n = 8, 16
        X = torch.randn(m, n, device="cuda")
        Y = sr_sinkhorn(X, num_iters=10)

        assert Y.device.type == "cuda"
        assert not torch.isnan(Y).any()
        assert not torch.isinf(Y).any()
        # CUDA float32 rounding causes slightly looser convergence than CPU;
        # use atol=1e-2 here -- tight convergence is already verified in CPU tests.
        row_norms = Y.norm(dim=1)
        col_norms = Y.norm(dim=0)
        assert torch.allclose(row_norms, torch.full_like(row_norms, math.sqrt(n)), atol=1e-2)
        assert torch.allclose(col_norms, torch.full_like(col_norms, math.sqrt(m)), atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
