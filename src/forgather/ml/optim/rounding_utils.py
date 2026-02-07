"""
Stochastic Rounding Utilities for Optimizers

Implements stochastic rounding for bfloat16 precision training.

References:
- Stochastic Rounding: https://arxiv.org/abs/2010.06192
- Implementation source: https://github.com/pytorch/ao/blob/main/torchao/optim/quant_utils.py#L120
"""

import torch
from torch import Tensor


def fp32_to_bf16_stochastic_round(x_f32: Tensor) -> Tensor:
    """
    Convert FP32 tensor to BF16 with stochastic rounding.

    For an FP32 number [a31, ..., a16, a15, ..., a0] to be converted to BF16:
    - Round towards zero:   [a31, ..., a16,   0, ...,  0]
    - Round away from zero: [a31, ..., a16+1, 0, ...,  0]

    (Since the value can be negative, we use round towards/away from zero
    instead of round up/down)

    For stochastic rounding, we round away from zero with the probability of
    [a15, ..., a0] / 2^16, where the bit pattern [a15, ..., a0] is interpreted
    as uint16.

    Args:
        x_f32: Input tensor in FP32 precision

    Returns:
        Tensor converted to BF16 with stochastic rounding
    """
    # Generate random 16-bit integers for stochastic decision
    # We have to use int32 since most arithmetic ops are not implemented for uint32/int16/uint16
    rand_16bit = torch.randint(
        0, 1 << 16, x_f32.shape, device=x_f32.device, dtype=torch.int32
    )

    # View FP32 as int32 to manipulate bits
    x_f32_bits = x_f32.view(torch.int32)
    x_fraction = x_f32_bits & 0xFFFF  # Lower 16 bits (fractional part for BF16)
    x_bf16_towards_zero = x_f32_bits & 0xFFFF0000  # Upper 16 bits (BF16 mantissa)

    # Stochastically decide whether to round up
    # This is True with the probability of x_fraction / 2^16
    x_f32_bits = torch.where(
        rand_16bit < x_fraction,
        x_bf16_towards_zero + 0x10000,  # Round away from zero (might overflow -> UB for signed int)
        x_bf16_towards_zero,  # Round towards zero
    )

    # Alternative, slightly faster implementation (commented out):
    # x_f32_bits = (x_f32_bits + rand_16bit) & 0xFFFF0000

    return x_f32_bits.view(torch.float32).bfloat16()
