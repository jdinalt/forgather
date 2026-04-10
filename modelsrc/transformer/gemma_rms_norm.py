from typing import Sequence, Union

import torch
from torch import Tensor, nn


class GemmaRMSNorm(nn.Module):
    """
    RMSNorm variant used by Google Gemma / Gemma-2 / Gemma-3.

    Differences from ``torch.nn.RMSNorm``:

    1. The learned weight is applied as ``(1.0 + weight)`` instead of ``weight``.
       As a consequence, the weight is initialized to zero (so a freshly-constructed
       ``GemmaRMSNorm`` behaves identically to a standard RMSNorm whose weight is one).
    2. Normalization is computed in float32 for numerical stability, and the result
       is cast back to the input's dtype before returning.
    """

    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int]],
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(
            torch.empty(self.normalized_shape, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_f32 = x_f32 * torch.rsqrt(variance + self.eps)
        out = x_f32 * (1.0 + self.weight.float())
        return out.to(orig_dtype)
