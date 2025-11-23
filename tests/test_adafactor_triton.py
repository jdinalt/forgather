"""
Test script for Adafactor Triton implementation.

Tests:
1. Correctness: Compare Triton vs PyTorch outputs
2. Memory usage: Measure peak memory with both implementations
3. Performance: Compare execution time
"""

import torch
import torch.nn as nn
from src.forgather.ml.optim.adafactor import Adafactor


def create_test_model():
    """Create a simple model for testing."""
    model = nn.Sequential(
        nn.Linear(1024, 2048),  # 2D weight matrix
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.LayerNorm(512),  # Has 1D parameters
    )
    return model


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

        # Test memory usage
        test_memory_usage()

        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        if correctness_passed:
            print("✓ All tests passed")
        else:
            print("✗ Some tests failed")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
