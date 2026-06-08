"""Bulk-tensor wire serialization for the DiLoCo HTTP star transport.

The bulk legs (pseudo-gradients up, averaged weights down) frame a state dict
into bytes. Two codecs are supported, negotiated via ``/info`` and stamped per
request in the frame's JSON header (``"fmt"``):

- ``"pickle"`` — ``torch.save`` / ``torch.load``. The historical format; dtypes
  ride implicitly inside the pickle. Default, for back-compat with older peers.
- ``"safetensors"`` — ``safetensors.torch.save`` / ``load``. No pickle (no
  arbitrary-code deserialization), an explicit dtype/shape header on the wire,
  zero-copy load, and the *same* format already used for on-disk checkpoints
  (``sharded_checkpoint.py``) — so wire and disk unify.

This is the single shared definition; ``DiLoCoClient`` and ``DiLoCoServer`` both
delegate here so the two ends can never drift.
"""

from typing import Dict

import torch

WIRE_FORMATS = ("pickle", "safetensors")
DEFAULT_WIRE_FORMAT = "pickle"


def serialize_state_dict(
    state_dict: Dict[str, torch.Tensor], fmt: str = DEFAULT_WIRE_FORMAT
) -> bytes:
    """Serialize a state dict to bytes using the named wire format."""
    if fmt == "pickle":
        import io

        buf = io.BytesIO()
        torch.save(state_dict, buf)
        return buf.getvalue()
    if fmt == "safetensors":
        from safetensors.torch import save as st_save

        # safetensors requires dense, contiguous CPU tensors. Pseudo-gradients
        # are freshly computed (global - local) and already on CPU, but the
        # upload cast / download path can yield non-contiguous views, so make
        # the guarantee explicit here rather than relying on the caller.
        # Precondition: no two entries share storage — safetensors.save raises
        # on aliased tensors, and ``.contiguous()`` does not clone an already-
        # contiguous view. Every bulk-leg state dict here is alias-free (per-name
        # pseudo-grad subtractions; server params are per-name clones), so this
        # holds. A future caller serializing a live tied-param state_dict would
        # need ``.clone()`` instead.
        dense = {k: v.detach().contiguous() for k, v in state_dict.items()}
        return st_save(dense)
    raise ValueError(f"Unknown wire format {fmt!r}; expected one of {WIRE_FORMATS}.")


def deserialize_state_dict(
    data: bytes, fmt: str = DEFAULT_WIRE_FORMAT
) -> Dict[str, torch.Tensor]:
    """Deserialize bytes to a state dict using the named wire format."""
    if fmt == "pickle":
        import io

        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if fmt == "safetensors":
        from safetensors.torch import load as st_load

        # safetensors loads onto CPU by default — preserves the historical
        # ``map_location="cpu"`` contract.
        return st_load(data)
    raise ValueError(f"Unknown wire format {fmt!r}; expected one of {WIRE_FORMATS}.")
