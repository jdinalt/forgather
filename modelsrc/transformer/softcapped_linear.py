from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class SoftcappedLinear(nn.Linear):
    """
    nn.Linear drop-in that applies logit softcapping: ``y = cap * tanh(Wx / cap)``.

    Used as the ``lm_head`` of models that need a bounded output distribution
    (Gemma-2 and larger Gemma-3 variants). When ``softcap`` is ``None``, ``forward``
    is numerically identical to ``nn.Linear.forward``, so it is a safe default
    replacement: only models that actually set a softcap value pay the ``tanh``
    cost.

    The softcap value is stored as an instance attribute so that the fused
    linear-cross-entropy loss path -- which bypasses ``forward`` and reads
    ``self.weight`` directly -- can auto-discover it off the lm_head module
    it receives from ``Trainer._maybe_get_fused_loss_fn``.

    Because this is a strict ``nn.Linear`` subclass, ``.weight`` / ``.bias`` /
    ``reset_parameters`` / tied-weights keys / ``init_prefix`` regex matching all
    behave identically to a plain ``nn.Linear``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        softcap: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.softcap = softcap

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, softcap={self.softcap}"

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.softcap is not None:
            y = self.softcap * torch.tanh(y / self.softcap)
        return y
