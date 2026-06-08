"""Tests for the DiLoCo bulk-tensor wire codecs (issue #154).

``wire_serialize`` is the single shared serialize/deserialize seam the client and
server both delegate to. The two codecs (``pickle``, ``safetensors``) must
round-trip a state dict identically; the format is negotiated via /info and
stamped per-request in the frame's ``fmt`` header.
"""

import pytest
import torch

from forgather.ml.diloco.wire_serialize import (
    WIRE_FORMATS,
    deserialize_state_dict,
    serialize_state_dict,
)


def _sd(seed=0):
    torch.manual_seed(seed)
    return {
        "layer0.weight": torch.randn(3, 4),
        "layer1.weight": torch.randn(4, 3),
        "layer1.bias": torch.randn(4),
    }


@pytest.mark.parametrize("fmt", WIRE_FORMATS)
def test_round_trip_identity(fmt):
    sd = _sd()
    out = deserialize_state_dict(serialize_state_dict(sd, fmt), fmt)
    assert sorted(out.keys()) == sorted(sd.keys())
    for k in sd:
        assert torch.equal(out[k], sd[k]), k


def test_both_codecs_agree():
    """pickle and safetensors must reconstruct the same tensors — they are
    interchangeable wire representations of one state dict."""
    sd = _sd(1)
    via_pickle = deserialize_state_dict(serialize_state_dict(sd, "pickle"), "pickle")
    via_st = deserialize_state_dict(
        serialize_state_dict(sd, "safetensors"), "safetensors"
    )
    for k in sd:
        assert torch.equal(via_pickle[k], via_st[k]), k


@pytest.mark.parametrize("fmt", WIRE_FORMATS)
def test_single_tensor_dict(fmt):
    sd = {"only.weight": torch.randn(5)}
    out = deserialize_state_dict(serialize_state_dict(sd, fmt), fmt)
    assert torch.equal(out["only.weight"], sd["only.weight"])


@pytest.mark.parametrize("fmt", WIRE_FORMATS)
def test_bf16_preserved(fmt):
    """The upload cast can hand the codec bf16 tensors; dtype must survive the
    wire (safetensors records it explicitly, pickle implicitly)."""
    sd = {"w": torch.randn(4, 4).to(torch.bfloat16)}
    out = deserialize_state_dict(serialize_state_dict(sd, fmt), fmt)
    assert out["w"].dtype == torch.bfloat16
    assert torch.equal(out["w"], sd["w"])


def test_safetensors_non_contiguous_does_not_raise():
    """safetensors rejects non-contiguous tensors; the codec must defensively
    make them contiguous (the upload cast / a transposed view can produce
    non-contiguous storage)."""
    base = torch.randn(4, 6)
    sd = {"t": base.t()}  # a transposed (non-contiguous) view
    assert not sd["t"].is_contiguous()
    out = deserialize_state_dict(serialize_state_dict(sd, "safetensors"), "safetensors")
    assert torch.equal(out["t"], base.t())


def test_unknown_format_fails_loud():
    with pytest.raises(ValueError, match="Unknown wire format"):
        serialize_state_dict(_sd(), "json")
    with pytest.raises(ValueError, match="Unknown wire format"):
        deserialize_state_dict(b"", "json")
