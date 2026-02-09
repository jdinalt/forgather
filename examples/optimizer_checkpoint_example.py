#!/usr/bin/env python3
"""
Example demonstrating optimizer checkpoint save/restore with custom optimizers.

This example shows how Apollo's projector state is properly serialized and
restored through the checkpoint system.
"""

import torch
import torch.nn as nn

from forgather.ml.optim import AdamW, Apollo, Multiopt
from forgather.ml.optim.subspace_proj import OnlinePCAProjector, RandProjector


def create_model():
    """Simple model for demonstration."""
    return nn.Sequential(
        nn.Linear(10, 20, bias=False), nn.ReLU(), nn.Linear(20, 2, bias=False)
    )


def train_step(model, optimizer):
    """Perform a single training step."""
    optimizer.zero_grad()
    x = torch.randn(4, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    return loss.item()


def demo_apollo_checkpoint():
    """Demonstrate Apollo checkpoint save/restore with projector state."""
    print("=" * 60)
    print("Apollo Optimizer Checkpoint Example")
    print("=" * 60)

    # Create model and optimizer
    model = create_model()
    opt = Apollo(
        model.parameters(),
        lr=0.001,
        rank=2,
        projector_factory=lambda rank, dim, proj_type: OnlinePCAProjector(
            rank, dim, proj_type, update_steps=10
        ),
    )

    # Train for some steps
    print("\n1. Training for 10 steps...")
    losses_before = []
    for i in range(10):
        loss = train_step(model, opt)
        losses_before.append(loss)
    print(f"   Final loss: {losses_before[-1]:.4f}")

    # Inspect projector state before save
    print("\n2. Inspecting projector state before save...")
    for param, state in opt.state.items():
        if "projector" in state:
            proj = state["projector"]
            print(f"   Projector type: {type(proj).__name__}")
            print(f"   Projector rank: {proj.rank}")
            print(f"   Projector dim: {proj.dim}")
            print(
                f"   Projector A shape: {proj.A.shape if proj.A is not None else None}"
            )
            break

    # Save checkpoint
    print("\n3. Saving checkpoint...")
    opt_state = opt.state_dict()
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": opt_state,  # Projector serialized here
        "step": 10,
    }
    # Note: We save just state_dict, not the full optimizer object
    # (which contains unpicklable lambda functions in defaults)
    torch.save(checkpoint, "/tmp/apollo_checkpoint.pt")
    print("   Checkpoint saved to /tmp/apollo_checkpoint.pt")

    # Verify projector is serialized as dict
    opt_state = checkpoint["optimizer"]
    for param_id, param_state in opt_state["state"].items():
        if "projector" in param_state:
            proj_dict = param_state["projector"]
            print(f"\n4. Projector serialization verified:")
            print(f"   Type in checkpoint: {type(proj_dict)}")
            print(f"   Keys: {list(proj_dict.keys())}")
            break

    # Create fresh model and optimizer
    print("\n5. Creating fresh model and optimizer...")
    model2 = create_model()
    opt2 = Apollo(
        model2.parameters(),
        lr=0.001,
        rank=2,
        projector_factory=lambda rank, dim, proj_type: OnlinePCAProjector(
            rank, dim, proj_type, update_steps=10
        ),
    )

    # Load checkpoint
    print("\n6. Loading checkpoint...")
    checkpoint = torch.load("/tmp/apollo_checkpoint.pt")
    model2.load_state_dict(checkpoint["model"])
    opt2.load_state_dict(checkpoint["optimizer"])  # Projector reconstructed here
    print("   Checkpoint loaded successfully")

    # Verify projector was reconstructed
    print("\n7. Verifying projector reconstruction...")
    for param, state in opt2.state.items():
        if "projector" in state:
            proj = state["projector"]
            print(f"   Projector type: {type(proj).__name__}")
            print(f"   Projector is object: {not isinstance(proj, dict)}")
            print(f"   Projector rank: {proj.rank}")
            print(f"   Projector dim: {proj.dim}")
            break

    # Continue training
    print("\n8. Continuing training for 5 more steps...")
    losses_after = []
    for i in range(5):
        loss = train_step(model2, opt2)
        losses_after.append(loss)
    print(f"   Final loss: {losses_after[-1]:.4f}")

    print("\n" + "=" * 60)
    print("SUCCESS: Apollo checkpoint save/restore works correctly!")
    print("=" * 60)


def demo_multiopt_checkpoint():
    """Demonstrate Multiopt checkpoint with mixed optimizers."""
    print("\n\n" + "=" * 60)
    print("Multiopt Checkpoint Example")
    print("=" * 60)

    # Create model with parameters split across optimizers
    model = create_model()
    params = list(model.parameters())

    print("\n1. Creating Multiopt with AdamW + Apollo...")
    opt1 = AdamW([params[0]], lr=0.001)
    opt2 = Apollo(
        [params[1]],
        lr=0.001,
        rank=2,
        projector_factory=lambda rank, dim, proj_type: RandProjector(
            rank, dim, proj_type, update_steps=10, lazy=False
        ),
    )
    multiopt = Multiopt([opt1, opt2])

    # Train
    print("\n2. Training for 10 steps...")
    for i in range(10):
        train_step(model, multiopt)

    # Save checkpoint
    print("\n3. Saving checkpoint...")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": multiopt.state_dict(),
    }
    torch.save(checkpoint, "/tmp/multiopt_checkpoint.pt")

    # Verify structure
    opt_state = checkpoint["optimizer"]
    print(f"\n4. Multiopt checkpoint structure:")
    print(f"   Has 'optimizers' key: {'optimizers' in opt_state}")
    print(f"   Number of wrapped optimizers: {len(opt_state['optimizers'])}")

    # Load checkpoint
    print("\n5. Loading checkpoint into fresh Multiopt...")
    model2 = create_model()
    params2 = list(model2.parameters())
    opt1_new = AdamW([params2[0]], lr=0.001)
    opt2_new = Apollo(
        [params2[1]],
        lr=0.001,
        rank=2,
        projector_factory=lambda rank, dim, proj_type: RandProjector(
            rank, dim, proj_type, update_steps=10, lazy=False
        ),
    )
    multiopt2 = Multiopt([opt1_new, opt2_new])

    checkpoint = torch.load("/tmp/multiopt_checkpoint.pt")
    model2.load_state_dict(checkpoint["model"])
    multiopt2.load_state_dict(checkpoint["optimizer"])
    print("   Checkpoint loaded successfully")

    # Continue training
    print("\n6. Continuing training...")
    for i in range(5):
        train_step(model2, multiopt2)

    print("\n" + "=" * 60)
    print("SUCCESS: Multiopt checkpoint save/restore works correctly!")
    print("=" * 60)


if __name__ == "__main__":
    torch.manual_seed(42)
    demo_apollo_checkpoint()

    torch.manual_seed(42)
    demo_multiopt_checkpoint()

    print("\n\nAll examples completed successfully!")
