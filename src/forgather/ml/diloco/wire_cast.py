"""Wire-precision casts for DiLoCo's bulk transport.

Each sync backend owns how pseudo-gradients are represented on the wire (issue
#154). This module holds the cast used by the HTTP star backend: fp32 / bf16,
with optional stochastic rounding (SR) on the narrowing cast. It lives with the
transport rather than in ``ParamView`` — which does model-level math only — so a
future backend (a collective shipping packed fp8/fp4, a shared-memory region that
casts nothing) can define its own representation without touching the model
abstraction. This backend stays bf16 / fp32.
"""

from __future__ import annotations

import torch


def cast_for_upload(
    pg: torch.Tensor, upload_dtype: str, upload_sr: bool
) -> torch.Tensor:
    """Cast a single pseudo-gradient tensor to the configured wire dtype.

    Semantics:

    * ``upload_dtype == "fp32"``: pass through (no cast). Any input dtype is
      accepted; tensors smaller than fp32 are silently upcast by the
      serialisation if needed.
    * ``upload_dtype == "bf16"``: cast to bf16. With ``upload_sr=True`` and an
      fp32-resolution input, route through
      :func:`fp32_to_bf16_stochastic_round` to preserve sub-ULP signal in
      expectation; otherwise round-to-nearest via ``.bfloat16()``. When the
      input is already bf16 the cast is identity and SR has no effect.
    """
    if upload_dtype == "fp32":
        return pg
    if upload_dtype == "bf16":
        if upload_sr and pg.dtype == torch.float32:
            from forgather.ml.optim.rounding_utils import (
                fp32_to_bf16_stochastic_round,
            )

            return fp32_to_bf16_stochastic_round(pg)
        return pg.to(torch.bfloat16)
    raise ValueError(f"unsupported upload_dtype: {upload_dtype!r}")
