"""Detect and install torchao quantization on a model at load time.

The native checkpoint loader (:func:`forgather.ml.sharded_checkpoint.load_checkpoint`)
uses these helpers to make ``-c CHECKPOINT_PATH`` work transparently on
torchao-quantized artifacts produced by ``forgather finalize --quantize``.
Detection is two-tier:

1. ``<model_dir>/config.json`` carries a ``quantization_config`` block
   (written by finalize). HF's ``TorchAoConfig`` deserializes it directly
   into an ``AOBaseConfig`` — the fast path; all Forgather-finalized
   artifacts hit it.
2. The saved state_dict contains torchao tensor subclasses. We reconstruct
   the base config from the subclass type and its attributes. v1 supports
   :data:`IntxUnpackedToInt8Tensor` (``int8-dynamic-act-int4-weight``) and
   :data:`Int4Tensor` (``int4-weight-only``); the ``float8`` recipe is
   only available via the config.json hint (saving its tensor subclass
   for state_dict-only reload would need SM ≥8.9 hardware at finalize
   time, which we can't test on common GPUs).

Once a base config is in hand, :func:`install_torchao_quantization` runs
``quantize_(module, QATConfig(base_config, step="prepare"))`` followed by
``step="convert"``. The convert step swaps each ``nn.Linear`` for the
corresponding torchao quantized linear class, so that a subsequent
``load_state_dict`` lands the saved quantized tensor subclasses in slots
that know how to hold them.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def install_torchao_quantization(module, base_config) -> None:
    """Install torchao quantized linear modules in-place via prepare→convert.

    Mirrors what ``tools/finalize_model/finalize_model.py:_apply_quantize``
    does for QAT/PTQ at finalize time, but called at *load* time on an
    already-constructed model right before ``load_state_dict``. The scales
    computed during convert get overwritten by the loaded tensors; only
    the module type swap matters.

    Also registers torchao tensor subclasses with PyTorch's
    ``add_safe_globals`` so ``torch.load(weights_only=True)`` accepts them
    in the subsequent state_dict load. Without this, the load fails with
    ``UnpicklingError``: torchao classes aren't on PyTorch's default
    allowlist, and HF's ``TorchAoHfQuantizer`` does the same registration
    on its own load path.
    """
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig

    _register_torchao_safe_globals()
    quantize_(module, QATConfig(base_config, step="prepare"))
    quantize_(module, QATConfig(base_config, step="convert"))


_TORCHAO_SAFE_GLOBALS_REGISTERED = False


def _register_torchao_safe_globals() -> None:
    """Make torchao tensor subclasses safe to ``torch.load(weights_only=True)``.

    Idempotent. We scan ``torchao.quantization``, ``torchao.dtypes``, and a
    handful of enum modules for tensor-subclass / enum types and pass them
    to ``torch.serialization.add_safe_globals``. PyTorch's allowlist applies
    process-wide; subsequent ``torch.load`` calls with ``weights_only=True``
    will accept these classes.
    """
    global _TORCHAO_SAFE_GLOBALS_REGISTERED
    if _TORCHAO_SAFE_GLOBALS_REGISTERED:
        return

    import torch

    safe = []

    def _collect_from(module_name: str) -> None:
        try:
            mod = __import__(module_name, fromlist=["*"])
        except ImportError:
            return
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name, None)
            if not isinstance(obj, type):
                continue
            # tensor subclasses + enums used by their __setstate__
            mod_path = getattr(obj, "__module__", "")
            if mod_path.startswith("torchao"):
                safe.append(obj)

    _collect_from("torchao.quantization")
    _collect_from("torchao.dtypes")
    _collect_from("torchao.quantization.granularity")
    _collect_from("torchao.quantization.quant_primitives")

    if safe:
        # Deduplicate; add_safe_globals warns on duplicates.
        torch.serialization.add_safe_globals(list(set(safe)))
    _TORCHAO_SAFE_GLOBALS_REGISTERED = True


def _base_config_from_config_json(model_dir: str):
    """Parse ``<model_dir>/config.json`` for a torchao ``quantization_config`` block.

    Returns the underlying ``AOBaseConfig`` (``TorchAoConfig.quant_type``) or
    None when the block is absent / malformed / not torchao.

    ``model_dir`` may point at the model root (which has ``config.json``) *or*
    at a ``checkpoints/checkpoint-N`` subdir (which doesn't). For the latter
    we walk up two parents to find the root.
    """
    candidates = [model_dir]
    parent = os.path.dirname(model_dir)
    if os.path.basename(parent) == "checkpoints":
        candidates.append(os.path.dirname(parent))

    for d in candidates:
        cfg_path = os.path.join(d, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            continue
        block = cfg.get("quantization_config")
        if not isinstance(block, dict):
            continue
        if block.get("quant_method") != "torchao":
            continue
        try:
            from transformers import TorchAoConfig

            return TorchAoConfig.from_dict(block).quant_type
        except Exception as e:
            logger.warning(
                "quantization_config present in %s but TorchAoConfig.from_dict failed: %s",
                cfg_path,
                e,
            )
            continue
    return None


def _base_config_from_tensor(t):
    """Reconstruct an ``AOBaseConfig`` from a saved torchao tensor subclass.

    Recognises the v1 recipes in :data:`QAT_RECIPES`. Returns None for
    unknown subclasses so callers can produce their own error message.

    Assumes the **canonical Forgather recipe** was used at finalize time
    (defaults in :func:`recipe_to_base_config`). Non-default packing
    formats, mapping types, or qparams algorithms saved into the tensor
    will be silently coerced to defaults here; the config.json path
    preserves them faithfully. For artifacts produced by Forgather's
    ``--quantize`` flag, the two recover the same config. For
    hand-crafted torchao configs without a ``quantization_config`` block
    in ``config.json``, prefer to restore the block rather than rely on
    this reverse-lookup.
    """
    import torch

    cls_name = type(t).__name__

    if cls_name == "IntxUnpackedToInt8Tensor":
        # int8-dynamic-act-int4-weight: weight is int4 per-group; group_size
        # comes from block_size = (1, group_size).
        from torchao.quantization import Int8DynamicActivationIntxWeightConfig
        from torchao.quantization.granularity import PerGroup

        block = getattr(t, "block_size", None)
        group_size = block[1] if block is not None and len(block) == 2 else 32
        return Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(group_size=int(group_size)),
        )

    if cls_name == "Int4Tensor":
        # int4-weight-only: group_size is dim-1 of block_size.
        from torchao.quantization import Int4WeightOnlyConfig

        block = getattr(t, "block_size", None)
        group_size = block[1] if block is not None and len(block) == 2 else 128
        return Int4WeightOnlyConfig(group_size=int(group_size))

    return None


def _detect_quantized_tensor(state_dict):
    """Return the first torchao tensor subclass instance in ``state_dict``, or None."""
    for v in state_dict.values():
        # nn.Parameter wraps regular tensors; quantized weights show up as
        # bare tensor subclasses whose module path starts with "torchao".
        if type(v).__module__.startswith("torchao"):
            return v
    return None


def detect_torchao_quantization(
    *,
    model_dir: str | None = None,
    state_dict: dict[str, Any] | None = None,
):
    """Return a torchao ``AOBaseConfig`` if the model is quantized, else None.

    Both signals are optional; pass whichever you have. ``model_dir`` is
    the fast path (no shard load needed). ``state_dict`` enables detection
    for artifacts whose config.json lacks the ``quantization_config`` block.

    Raises:
        ValueError: If ``state_dict`` contains a torchao tensor subclass
            the v1 reverse-lookup doesn't recognise. The error names the
            class and points the user at ``forgather finalize --quantize``
            to restore the metadata.
    """
    if model_dir is not None:
        cfg = _base_config_from_config_json(model_dir)
        if cfg is not None:
            return cfg

    if state_dict is not None:
        sample = _detect_quantized_tensor(state_dict)
        if sample is not None:
            cfg = _base_config_from_tensor(sample)
            if cfg is not None:
                return cfg
            cls_name = type(sample).__name__
            float8_hint = ""
            if "Float8" in cls_name:
                float8_hint = (
                    " (the v1 reverse-lookup does not cover float8 — float8 "
                    "checkpoints must carry a `quantization_config` block.)"
                )
            raise ValueError(
                f"State dict contains a torchao quantized tensor subclass "
                f"{cls_name!r} that this version of Forgather doesn't know "
                f"how to reverse-engineer into a base config.{float8_hint} "
                f"Restore the `quantization_config` block in "
                f"`<model_dir>/config.json` (written by `forgather finalize "
                f"--quantize <recipe>`), or re-finalize the source model "
                f"with `--quantize`."
            )

    return None
