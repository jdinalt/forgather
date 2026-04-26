"""
Helpers for finalizing a trained Forgather model into a clean directory.

A "finalized" model directory contains the model source code, tokenizer,
config.json, generation_config.json, and exactly one preserved checkpoint
(model weights, optionally optimizer state). Scheduler/dataset/RNG/trainer
state from the source training run are dropped.

These helpers are called by ``tools/finalize_model/finalize_model.py`` and
``forgather finalize``.
"""

from __future__ import annotations

import datetime
import glob
import json
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from forgather.ml.sharded_checkpoint import (
    create_pretrained_symlinks,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from forgather.ml.trainer.checkpoint_types import (
    CheckpointManifest,
    ComponentManifest,
    SharingPattern,
)

logger = logging.getLogger(__name__)


# Token names that the auto-stop-token detector treats as end-of-turn markers.
_END_OF_TURN_TOKEN_NAMES = ("<|im_end|>", "<|eot|>", "<|end_of_turn|>")

# Files that live at the model root and should be copied verbatim when
# duplicating the source model. Subdirectories like ``checkpoints/``,
# ``runs/``, ``evals/``, and ``__pycache__/`` are not copied.
_ROOT_MARKER_FILES = (".package_files_copied",)


@dataclass
class SourceBundle:
    """Loaded source artifacts for a finalize operation."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    config: PretrainedConfig
    checkpoint_dir: Optional[str]
    step: Optional[int]


# ---------------------------------------------------------------------------
# Checkpoint resolution + source loading
# ---------------------------------------------------------------------------


def _parse_step_from_path(path: str) -> Optional[int]:
    base = os.path.basename(os.path.normpath(path))
    if base.startswith("checkpoint-"):
        suffix = base[len("checkpoint-") :]
        if suffix.isdigit():
            return int(suffix)
    return None


def resolve_checkpoint_dir(
    source: str, explicit_path: Optional[str]
) -> Tuple[Optional[str], Optional[int]]:
    """Resolve which directory holds the model weights to load.

    If ``explicit_path`` is given, that directory is used directly.
    Otherwise, ``find_latest_checkpoint(source)`` is used. Returns the
    absolute path and the parsed step number (or ``None`` when the path
    name does not encode a step, e.g. when weights are at the source root).
    """
    if explicit_path:
        ckpt_dir = os.path.abspath(explicit_path)
        if not os.path.isdir(ckpt_dir):
            raise ValueError(f"Checkpoint path does not exist: {ckpt_dir}")
        return ckpt_dir, _parse_step_from_path(ckpt_dir)

    latest = find_latest_checkpoint(source)
    if latest is None:
        return None, None
    return os.path.abspath(latest), _parse_step_from_path(latest)


def load_source_artifacts(
    source: str,
    checkpoint_dir: Optional[str],
    dtype: Optional[torch.dtype],
    device: str,
) -> SourceBundle:
    """Load the source config, tokenizer, and model.

    Config and tokenizer are loaded from the source root. Weights come
    from ``checkpoint_dir`` when provided, else from the source root via
    ``AutoModelForCausalLM.from_pretrained``.

    When ``dtype`` is None, the model is left in whatever dtype the
    checkpoint was saved in.
    """
    source = os.path.abspath(source)

    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)

    if checkpoint_dir is not None and os.path.abspath(checkpoint_dir) != source:
        logger.info(f"Loading config from {source} and weights from {checkpoint_dir}")
        model = cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_config(config, trust_remote_code=True),
        )
        # assign=True replaces the (uninitialized) params with the loaded
        # tensors verbatim, preserving the dtype the checkpoint was saved in.
        load_checkpoint(checkpoint_dir, model, device=device, assign=True)
    else:
        logger.info(f"Loading model from {source}")
        model = cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True),
        )
        if device and device != "cpu":
            model = cast(PreTrainedModel, model.to(device))  # type: ignore[arg-type]

    if dtype is not None:
        logger.info(f"Casting model to dtype {dtype}")
        model = cast(PreTrainedModel, model.to(dtype=dtype))  # type: ignore[call-arg]

    return SourceBundle(
        model=model,
        tokenizer=tokenizer,
        config=config,
        checkpoint_dir=checkpoint_dir,
        step=_parse_step_from_path(checkpoint_dir) if checkpoint_dir else None,
    )


# ---------------------------------------------------------------------------
# Source-tree copy
# ---------------------------------------------------------------------------


def copy_model_source(src_root: str, dst_root: str) -> List[str]:
    """Copy ``*.py`` and known marker files from ``src_root`` to ``dst_root``.

    Subdirectories (``checkpoints/``, ``runs/``, ``evals/``,
    ``__pycache__/``, ``output_models/``) are skipped.
    """
    os.makedirs(dst_root, exist_ok=True)
    copied: List[str] = []

    for path in sorted(glob.glob(os.path.join(src_root, "*.py"))):
        if not os.path.isfile(path):
            continue
        dst = os.path.join(dst_root, os.path.basename(path))
        shutil.copy2(path, dst)
        copied.append(dst)

    for name in _ROOT_MARKER_FILES:
        src = os.path.join(src_root, name)
        if os.path.isfile(src):
            dst = os.path.join(dst_root, name)
            shutil.copy2(src, dst)
            copied.append(dst)

    logger.info(f"Copied {len(copied)} source/marker file(s) to {dst_root}")
    return copied


# ---------------------------------------------------------------------------
# Stop-token detection
# ---------------------------------------------------------------------------


def detect_stop_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    source_eos_id: Optional[int],
    added_token_names: List[str],
    user_stop_tokens: List[str],
    auto: bool,
) -> List[int]:
    """Build the merged stop-token-id list.

    The original ``source_eos_id`` (if any) is always first. When
    ``auto`` is True, any added token whose name matches one of the
    known end-of-turn names contributes its ID. ``user_stop_tokens``
    are explicit token strings to look up and append.
    """
    ids: List[int] = []
    if source_eos_id is not None:
        ids.append(int(source_eos_id))

    def _maybe_add(token_name: str) -> None:
        if not token_name:
            return
        token_id_raw = tokenizer.convert_tokens_to_ids(token_name)
        if isinstance(token_id_raw, list):
            token_id_raw = token_id_raw[0] if token_id_raw else None
        if token_id_raw is None or token_id_raw == tokenizer.unk_token_id:
            logger.warning(
                f"Stop-token '{token_name}' not found in tokenizer vocabulary"
            )
            return
        token_id = int(token_id_raw)
        if token_id not in ids:
            ids.append(token_id)

    if auto:
        for name in added_token_names:
            if name in _END_OF_TURN_TOKEN_NAMES:
                _maybe_add(name)
        # If the YAML defined a new ``eos_token`` (different string from
        # the original), the tokenizer's current ``eos_token_id`` reflects
        # that; include it. We've already retained the original eos above.
        current_eos = tokenizer.eos_token_id
        if current_eos is not None:
            current_eos_int = int(
                current_eos[0] if isinstance(current_eos, list) else current_eos
            )
            if current_eos_int not in ids:
                ids.append(current_eos_int)

    for tok in user_stop_tokens:
        _maybe_add(tok.strip())

    return ids


# ---------------------------------------------------------------------------
# Generation config synthesis
# ---------------------------------------------------------------------------


_PRESET_PASSTHROUGH_KEYS = {
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "encoder_no_repeat_ngram_size",
    "renormalize_logits",
    "num_beams",
    "num_beam_groups",
    "diversity_penalty",
    "early_stopping",
    "length_penalty",
    "penalty_alpha",
    "do_sample",
    "min_new_tokens",
    "max_new_tokens",
    "max_length",
    "guidance_scale",
    "presence_penalty",
    "frequency_penalty",
}


def translate_inference_preset_to_hf(preset: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a Forgather inference preset to the HF generation config schema.

    Forgather presets use ``max_tokens`` (mirroring chat-completion APIs);
    HuggingFace expects ``max_new_tokens``. Sampling defaults are inferred
    when not explicit: presence of ``temperature``/``top_p``/``top_k`` and
    no ``do_sample`` implies ``do_sample=true``; presence of ``num_beams``
    > 1 with no ``do_sample`` implies ``do_sample=false``.
    """
    out: Dict[str, Any] = {}

    if "max_tokens" in preset:
        out["max_new_tokens"] = int(preset["max_tokens"])

    for key, value in preset.items():
        if key == "max_tokens":
            continue
        if key in _PRESET_PASSTHROUGH_KEYS:
            out[key] = value

    if "do_sample" not in out:
        if any(
            k in out and out[k] is not None
            for k in ("temperature", "top_p", "top_k", "min_p")
        ):
            out["do_sample"] = True
        elif out.get("num_beams", 1) and out.get("num_beams", 1) > 1:
            out["do_sample"] = False

    return out


def _resolve_preset_path(preset_arg: str) -> str:
    """Resolve a ``--generation-config`` value to a JSON file path.

    A path that exists is used directly. A bare name is searched first
    in ``$FORGATHER_ROOT/generation_config/`` (when discoverable), then
    in ``~/.forgather/generation_config/``.
    """
    if os.path.exists(preset_arg) and os.path.isfile(preset_arg):
        return preset_arg

    candidates: List[str] = []
    name = preset_arg if preset_arg.endswith(".json") else f"{preset_arg}.json"

    # Try the user's installed forgather workspace, if present.
    try:
        from forgather import MetaConfig

        forgather_root = MetaConfig.find_workspace_dir(__file__)
        if forgather_root:
            candidates.append(os.path.join(forgather_root, "generation_config", name))
    except Exception:
        pass

    home = os.path.expanduser("~")
    candidates.append(os.path.join(home, ".forgather", "generation_config", name))

    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(
        f"Generation config preset '{preset_arg}' not found. "
        f"Looked in: {candidates}"
    )


def build_generation_config(
    source: str,
    mode: str,
    tokenizer: PreTrainedTokenizerBase,
    stop_token_ids: List[int],
) -> Optional[Dict[str, Any]]:
    """Build the generation_config.json dict for the finalized model.

    ``mode`` is one of:
        - ``"none"``: return ``None`` (caller skips writing).
        - ``"carry"``: copy the source's generation_config.json if present;
          otherwise synthesize a minimal ``{bos,pad,eos}`` dict.
        - any other string: treated as a preset name or path; loaded and
          translated via ``translate_inference_preset_to_hf``.

    Token IDs from ``tokenizer`` are always overlaid last. ``eos_token_id``
    is written as a list when ``stop_token_ids`` has more than one entry,
    or as a scalar when there is only one.
    """
    if mode == "none":
        return None

    base: Dict[str, Any] = {}

    if mode == "carry":
        src_path = os.path.join(source, "generation_config.json")
        if os.path.isfile(src_path):
            try:
                with open(src_path, "r") as f:
                    base = json.load(f)
                logger.info(f"Carrying forward {src_path}")
            except Exception as e:
                logger.warning(f"Failed to read {src_path}: {e}; synthesizing minimal")
                base = {}
    else:
        preset_path = _resolve_preset_path(mode)
        with open(preset_path, "r") as f:
            preset = json.load(f)
        logger.info(f"Loaded generation config preset from {preset_path}")
        base = translate_inference_preset_to_hf(preset)

    def _coerce_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, list):
            value = value[0] if value else None
        if value is None:
            return None
        return int(value)

    # Overlay token IDs from the (possibly-updated) tokenizer/config.
    bos = _coerce_int(getattr(tokenizer, "bos_token_id", None))
    pad = _coerce_int(getattr(tokenizer, "pad_token_id", None))

    if bos is not None:
        base["bos_token_id"] = bos
    if pad is not None:
        base["pad_token_id"] = pad

    if stop_token_ids:
        if len(stop_token_ids) == 1:
            base["eos_token_id"] = int(stop_token_ids[0])
        else:
            base["eos_token_id"] = [int(i) for i in stop_token_ids]
    else:
        eos = _coerce_int(getattr(tokenizer, "eos_token_id", None))
        if eos is not None:
            base["eos_token_id"] = eos

    return base


def write_generation_config(dest: str, gen_cfg: Dict[str, Any]) -> str:
    """Write generation_config.json to ``dest``."""
    path = os.path.join(dest, "generation_config.json")
    with open(path, "w") as f:
        json.dump(gen_cfg, f, indent=2)
    logger.info(f"Wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Checkpoint writing
# ---------------------------------------------------------------------------


def _write_finalize_manifest(
    checkpoint_dir: str,
    components: Dict[str, ComponentManifest],
) -> str:
    manifest = CheckpointManifest(
        checkpoint_path=checkpoint_dir,
        world_size=1,
        timestamp=datetime.datetime.now(),
        components=components,
        pytorch_version=torch.__version__,
        metadata={"finalized": True},
    )
    path = os.path.join(checkpoint_dir, "checkpoint_manifest.json")
    manifest.save(path)
    return path


def _component_size_bytes(*paths: str) -> int:
    total = 0
    for p in paths:
        if os.path.isfile(p):
            total += os.path.getsize(p)
    return total


def write_finalized_checkpoint(
    dest: str,
    model: PreTrainedModel,
    source_checkpoint_dir: Optional[str],
    step: Optional[int],
    safetensors: bool,
    keep_optimizer: bool,
    root_copy: bool,
) -> Optional[str]:
    """Write the finalized weights (and optional optimizer) to ``dest``.

    When ``root_copy`` is True, weights are written directly into ``dest``
    and no ``checkpoints/`` subdirectory is created. Otherwise, weights
    go into ``dest/checkpoints/checkpoint-<step>/`` and matching root
    symlinks are created via ``create_pretrained_symlinks``.

    Returns the absolute path of the checkpoint dir (or of ``dest`` when
    ``root_copy`` is True).
    """
    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)

    if root_copy:
        if keep_optimizer:
            raise ValueError(
                "--root-copy is incompatible with --keep-optimizer "
                "(no checkpoints/ directory is created)"
            )
        save_checkpoint(
            output_dir=dest,
            module=model,
            safetensors=safetensors,
            include_param_sharing=True,
        )
        return dest

    step_label = step if step is not None else 0
    checkpoint_dir = os.path.join(dest, "checkpoints", f"checkpoint-{step_label}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    save_checkpoint(
        output_dir=checkpoint_dir,
        module=model,
        safetensors=safetensors,
        include_param_sharing=True,
    )

    components: Dict[str, ComponentManifest] = {}

    # Record the model component (size approximated from shard files).
    weight_glob = "*.safetensors*" if safetensors else "pytorch_model*.bin*"
    weight_files = sorted(glob.glob(os.path.join(checkpoint_dir, weight_glob)))
    components["model"] = ComponentManifest(
        key="model",
        sharing_pattern=SharingPattern.REPLICATED.value,
        ranks=[0],
        replicated_across=[0],
        size_bytes=_component_size_bytes(*weight_files),
    )

    if keep_optimizer:
        if not source_checkpoint_dir:
            raise ValueError(
                "--keep-optimizer requires a source checkpoint directory; "
                "the source has no checkpoints/ subdir to copy from"
            )
        src_opt = os.path.join(source_checkpoint_dir, "optimizer_state.pt")
        if not os.path.isfile(src_opt):
            raise FileNotFoundError(
                f"optimizer_state.pt not found in {source_checkpoint_dir}"
            )
        dst_opt = os.path.join(checkpoint_dir, "optimizer_state.pt")
        shutil.copy2(src_opt, dst_opt)
        logger.info(f"Copied optimizer state to {dst_opt}")
        components["optimizer"] = ComponentManifest(
            key="optimizer",
            sharing_pattern=SharingPattern.REPLICATED.value,
            ranks=[0],
            replicated_across=[0],
            size_bytes=os.path.getsize(dst_opt),
        )

    _write_finalize_manifest(checkpoint_dir, components)

    # Create root-level symlinks so HF AutoModel.from_pretrained(dest) works.
    create_pretrained_symlinks(dest)

    return checkpoint_dir
