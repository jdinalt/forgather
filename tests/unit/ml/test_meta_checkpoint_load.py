"""End-to-end check of the HF-v5 meta checkpoint-load contract against a
real, self-contained forgather model directory (``models/tiny``).

Exercises the exact sequence the trainer uses for ``construct_model_on``
meta: build the skeleton on the meta device, materialize empty on the
target device, load the checkpoint (flagging loaded tensors), then
``initialize_missing_weights`` to recompute what the checkpoint didn't
carry (RoPE ``inv_freq``). Verifies loaded weights are preserved, derived
buffers are finite, and a forward pass runs.

Skipped when ``models/tiny`` isn't present (it's a generated artifact).
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.path.isdir("models/tiny") or not os.path.exists("models/tiny/config.json"),
    reason="models/tiny generated model dir not present",
)

MODEL_DIR = "models/tiny"


def _load_on_meta():
    from transformers import AutoConfig, AutoModelForCausalLM

    from forgather.ml.sharded_checkpoint import (
        create_sharing_metadata,
        initialize_missing_weights,
        load_checkpoint,
        retie_parameters,
    )

    config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    # 1. Construct the skeleton on meta (no allocation).
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # 2. Materialize empty on the target device + restore tying.
    sharing = create_sharing_metadata(model)
    model.to_empty(device="cpu")
    retie_parameters(model, sharing)
    # 3. Load the checkpoint (flags loaded tensors via flag_loaded_tensors).
    load_checkpoint(MODEL_DIR, model, device="cpu", strict=True)
    # 4. Initialize only what the checkpoint didn't fill (RoPE inv_freq).
    initialize_missing_weights(model)
    return model


def test_meta_load_preserves_weights_and_inits_buffers():
    model = _load_on_meta().eval()

    # No tensor left on meta — everything materialized.
    metas = [n for n, t in model.named_parameters() if t.is_meta]
    metas += [n for n, t in model.named_buffers() if t.is_meta]
    assert not metas, f"tensors left on meta: {metas}"

    # Loaded weights match the on-disk checkpoint (sample one param).
    from forgather.ml.sharded_checkpoint import load_checkpoint

    ckpt = load_checkpoint(MODEL_DIR, None, device="cpu")  # raw state dict
    sample_key = next(k for k in ckpt if k.endswith("weight") and ckpt[k].dim() >= 2)
    live = dict(model.named_parameters())[sample_key]
    assert torch.allclose(live, ckpt[sample_key]), f"{sample_key} not loaded"

    # Derived (non-persistent) buffers were recomputed: finite, non-empty.
    rope_buffers = [(n, t) for n, t in model.named_buffers() if n.endswith("inv_freq")]
    assert rope_buffers, "expected at least one RoPE inv_freq buffer"
    for n, t in rope_buffers:
        assert torch.isfinite(t).all(), f"{n} has non-finite values"
        assert t.abs().sum() > 0, f"{n} left at zero (not recomputed)"


def test_meta_loaded_model_runs_forward():
    model = _load_on_meta().eval()
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, 8))
    with torch.no_grad():
        out = model(input_ids=input_ids)
    # The model may return a CausalLMOutput or a raw logits tensor.
    logits = out.logits if hasattr(out, "logits") else out
    assert logits.shape[0] == 1 and logits.shape[1] == 8
    assert logits.shape[-1] == vocab
    assert torch.isfinite(logits).all()
