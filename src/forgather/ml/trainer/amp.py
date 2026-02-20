"""Automatic Mixed Precision (AMP) support for Forgather trainers.

Encapsulates torch.autocast and torch.amp.GradScaler into a single AMPContext
that can be shared across trainer implementations.

Two modes:
- bf16: autocast with bfloat16, no loss scaling (bfloat16 has same exponent range as float32)
- fp16: autocast with float16 + GradScaler for dynamic loss scaling
"""

import logging
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

logger = logging.getLogger(__name__)


class AMPContext:
    """
    Manages autocast and GradScaler for mixed precision training.

    Usage::

        amp_ctx = AMPContext(mixed_precision="bf16", device_type="cuda")

        # Forward/backward:
        with amp_ctx.autocast():
            outputs = model(**inputs)
            loss = loss_fn(outputs, targets)
        amp_ctx.scale_loss(loss).backward()

        # Before optimizer step:
        amp_ctx.unscale_(optimizer)
        clip_grad_norm_(...)
        amp_ctx.optimizer_step(optimizer)
        optimizer.zero_grad()
    """

    def __init__(
        self,
        mixed_precision: str | None,
        device_type: str = "cuda",
        initial_scale: float = 2**16,
    ):
        self.mixed_precision = mixed_precision
        self.device_type = device_type
        self.enabled = mixed_precision is not None

        if mixed_precision == "bf16":
            self.amp_dtype = torch.bfloat16
            self.scaler: GradScaler | None = None
        elif mixed_precision == "fp16":
            self.amp_dtype = torch.float16
            self.scaler = GradScaler(device=device_type, init_scale=initial_scale)
        else:
            self.amp_dtype = None
            self.scaler = None

    def autocast(self):
        """Return autocast context manager, or nullcontext if disabled."""
        if not self.enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.amp_dtype)

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Scale loss for backward pass. Identity for bf16/disabled, scales for fp16."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def unscale_(self, optimizer) -> None:
        """Unscale gradients before clipping. No-op for bf16/disabled."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def optimizer_step(self, optimizer) -> None:
        """Step optimizer via scaler (fp16) or directly (bf16/disabled).

        For fp16, also calls scaler.update() and logs if the step was skipped
        due to inf/nan gradients.
        """
        if self.scaler is not None:
            old_scale = self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            new_scale = self.scaler.get_scale()
            if old_scale > new_scale:
                logger.warning(
                    f"GradScaler detected inf/nan gradients, skipped optimizer step. "
                    f"Loss scale: {old_scale:.1f} -> {new_scale:.1f}"
                )
        else:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        """Return scaler state for checkpointing."""
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scaler state from checkpoint."""
        if self.scaler is not None and state_dict:
            self.scaler.load_state_dict(state_dict)
            logger.info(
                f"Restored GradScaler state (scale={self.scaler.get_scale():.1f})"
            )
