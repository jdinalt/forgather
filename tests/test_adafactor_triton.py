"""
Test script for Adafactor Triton implementation.

Tests:
1. Correctness: Compare Triton vs PyTorch outputs
2. Determinism: Verify bitwise-identical results across runs
3. Memory usage: Measure peak memory with both implementations
"""

import torch
import torch.nn as nn

try:
    from src.forgather.ml.optim.adafactor import Adafactor
except ModuleNotFoundError:
    from forgather.ml.optim.adafactor import Adafactor


class _TestModel(nn.Module):
    """Test model with 1D (LayerNorm), 2D (Linear), and 3D (Conv1d) parameters."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, 512)
        self.norm = nn.LayerNorm(512)
        # Conv1d weight is 3D: (out_channels, in_channels/groups, kernel_size)
        self.conv = nn.Conv1d(512, 512, kernel_size=4, padding="same")

    def forward(self, x):
        # x: (batch, 1024)
        x = self.norm(self.linear2(self.relu(self.linear1(x))))
        # Conv1d expects (batch, channels, length) — treat each sample as length-1
        x = self.conv(x.unsqueeze(-1)).squeeze(-1)
        return x


def create_test_model():
    """Create a simple model for testing."""
    return _TestModel()


def test_correctness():
    """Test that Triton and PyTorch implementations produce similar results."""
    print("=" * 60)
    print("Testing Correctness")
    print("=" * 60)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create two identical models
    model_pytorch = create_test_model().cuda()
    model_triton = create_test_model().cuda()

    # Copy weights to ensure they start identical
    model_triton.load_state_dict(model_pytorch.state_dict())

    # Create optimizers
    opt_pytorch = Adafactor(
        model_pytorch.parameters(),
        lr=1e-3,
        weight_decay=0.001,
        use_triton=False,
    )

    opt_triton = Adafactor(
        model_triton.parameters(),
        lr=1e-3,
        weight_decay=0.001,
        use_triton=True,
    )

    # Run several optimization steps
    n_steps = 5
    for step in range(n_steps):
        # Create identical random input
        torch.manual_seed(42 + step)
        x = torch.randn(32, 1024).cuda()
        target = torch.randn(32, 512).cuda()

        # PyTorch optimizer step
        opt_pytorch.zero_grad()
        out_pytorch = model_pytorch(x)
        loss_pytorch = ((out_pytorch - target) ** 2).mean()
        loss_pytorch.backward()
        opt_pytorch.step()

        # Triton optimizer step
        torch.manual_seed(42 + step)  # Reset for identical randomness
        opt_triton.zero_grad()
        out_triton = model_triton(x)
        loss_triton = ((out_triton - target) ** 2).mean()
        loss_triton.backward()
        opt_triton.step()

        print(f"\nStep {step + 1}/{n_steps}:")
        print(f"  PyTorch loss: {loss_pytorch.item():.6f}")
        print(f"  Triton loss:  {loss_triton.item():.6f}")

    # Compare final parameters
    print("\n" + "=" * 60)
    print("Parameter Comparison:")
    print("=" * 60)

    max_diff = 0.0
    for (name_pt, param_pt), (name_tr, param_tr) in zip(
        model_pytorch.named_parameters(), model_triton.named_parameters()
    ):
        assert name_pt == name_tr
        diff = (param_pt - param_tr).abs().max().item()
        rel_diff = diff / (param_pt.abs().max().item() + 1e-8)
        max_diff = max(max_diff, diff)

        print(f"{name_pt}:")
        print(f"  Max abs diff: {diff:.6e}")
        print(f"  Max rel diff: {rel_diff:.6e}")

    print(f"\nOverall max difference: {max_diff:.6e}")

    if max_diff < 1e-3:
        print("✓ PASSED: Results are numerically similar")
    else:
        print("✗ FAILED: Results differ significantly")

    return max_diff < 1e-3


def _run_triton_steps(n_steps=10):
    """Run Triton optimizer for n_steps and return final parameter state dicts."""
    torch.manual_seed(42)
    model = create_test_model().cuda()
    opt = Adafactor(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.001,
        use_triton=True,
    )

    for step in range(n_steps):
        torch.manual_seed(42 + step)
        x = torch.randn(32, 1024).cuda()
        target = torch.randn(32, 512).cuda()

        opt.zero_grad()
        out = model(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()

    # Collect final parameters as detached clones
    params = {name: p.detach().clone() for name, p in model.named_parameters()}
    return params


def test_determinism():
    """Test that the Triton optimizer produces bitwise-identical results across runs."""
    print("\n" + "=" * 60)
    print("Testing Determinism")
    print("=" * 60)

    params_a = _run_triton_steps(n_steps=10)
    params_b = _run_triton_steps(n_steps=10)

    all_identical = True
    for name in params_a:
        identical = torch.equal(params_a[name], params_b[name])
        if not identical:
            diff = (params_a[name] - params_b[name]).abs().max().item()
            print(f"  {name}: NOT identical (max diff: {diff:.6e})")
            all_identical = False
        else:
            print(f"  {name}: identical")

    if all_identical:
        print("PASSED: Triton optimizer is bitwise deterministic")
    else:
        print("FAILED: Triton optimizer is NOT deterministic")

    return all_identical


def test_memory_usage():
    """Test memory usage of both implementations."""
    print("\n" + "=" * 60)
    print("Testing Memory Usage")
    print("=" * 60)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Test PyTorch implementation
    print("\nPyTorch Implementation:")
    model_pytorch = create_test_model().cuda()
    opt_pytorch = Adafactor(
        model_pytorch.parameters(),
        lr=1e-3,
        weight_decay=0.001,
        use_triton=False,
    )

    baseline_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"  Baseline memory: {baseline_mem:.2f} MB")

    x = torch.randn(32, 1024).cuda()
    target = torch.randn(32, 512).cuda()

    opt_pytorch.zero_grad()
    out = model_pytorch(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    opt_pytorch.step()

    peak_mem_pytorch = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak memory: {peak_mem_pytorch:.2f} MB")
    print(f"  Optimizer overhead: {peak_mem_pytorch - baseline_mem:.2f} MB")

    # Clean up
    del model_pytorch, opt_pytorch, x, target, out, loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Test Triton implementation
    print("\nTriton Implementation:")
    model_triton = create_test_model().cuda()
    opt_triton = Adafactor(
        model_triton.parameters(),
        lr=1e-3,
        weight_decay=0.001,
        use_triton=True,
    )

    baseline_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"  Baseline memory: {baseline_mem:.2f} MB")

    x = torch.randn(32, 1024).cuda()
    target = torch.randn(32, 512).cuda()

    opt_triton.zero_grad()
    out = model_triton(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    opt_triton.step()

    peak_mem_triton = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak memory: {peak_mem_triton:.2f} MB")
    print(f"  Optimizer overhead: {peak_mem_triton - baseline_mem:.2f} MB")

    # Compare
    print("\n" + "=" * 60)
    savings = peak_mem_pytorch - peak_mem_triton
    savings_pct = (savings / peak_mem_pytorch) * 100

    print(f"Memory savings: {savings:.2f} MB ({savings_pct:.1f}%)")

    if savings > 0:
        print("✓ Triton implementation uses less memory")
    else:
        print("✗ Triton implementation uses more memory")


def main():
    """Run all tests."""
    print("Testing Adafactor Triton Implementation")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping tests.")
        return

    try:
        # Test correctness
        correctness_passed = test_correctness()

        # Test determinism
        determinism_passed = test_determinism()

        # Test memory usage
        test_memory_usage()

        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        all_passed = correctness_passed and determinism_passed
        if all_passed:
            print("All tests passed")
        else:
            if not correctness_passed:
                print("FAILED: Correctness test")
            if not determinism_passed:
                print("FAILED: Determinism test")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
