#!/usr/bin/env python3
"""Migrate a saved Forgather model to the current source schema.

Reads provenance (``forgather_arch`` / ``forgather_arch_version``) from the
source model's ``config.json``, walks the per-arch
``forgather_migrations`` chain registered with the model converter, and
materialises a fresh model directory at the destination using the current
Forgather sources. Saved hyperparameters (RoPE base, sliding window, etc.)
are carried through the migrations rather than reset to template defaults.

This is the in-Forgather counterpart to ``forgather convert``: same
converter-plugin layout, but the chained version migrations stay inside
the Forgather schema instead of round-tripping through HuggingFace.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from argparse import RawTextHelpFormatter
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Surface forgather + the example converters when invoked as a script.
from forgather import MetaConfig, Project

forgather_root = MetaConfig.find_workspace_dir(os.path.abspath(__file__))
if forgather_root and forgather_root not in sys.path:
    sys.path.insert(0, forgather_root)

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from forgather.ml.construct import torch_dtype
from forgather.ml.model_conversion import (
    ModelConverter,
    compose_migration_chain,
    discover_and_register_converters,
    get_converter,
    list_converters,
)
from forgather.ml.no_init_weights import no_init_weights
from forgather.ml.remap_params import remap_state_dict
from forgather.ml.sharded_checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from forgather.ml.utils import default_dtype

logger = logging.getLogger(__name__)


# Files at the source root we copy verbatim if the regenerated destination
# does not already produce them. Tokenizer files are saved by the project
# materialisation, so they are intentionally excluded here.
_PASSTHROUGH_FILES = (
    "generation_config.json",
    "chat_template.jinja",
    "chat_template.json",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description=(
            "Update a saved Forgather model to the current source schema. "
            "Rebuilds the model code from current sources and applies the "
            "registered version migrations to the saved checkpoint."
        ),
        epilog=(
            "Examples:\n"
            "\n"
            "Update with stamped provenance:\n"
            "  forgather update output_models/llama_4m out/llama_4m_v2\n"
            "\n"
            "Override stamped arch / from-version:\n"
            "  forgather update SRC DST --arch llama --from-version 1\n"
            "\n"
            "Stop the chain at an intermediate version:\n"
            "  forgather update SRC DST --to-version 2\n"
        ),
    )
    parser.add_argument(
        "src_model_path",
        type=os.path.expanduser,
        help="Source Forgather model directory",
    )
    parser.add_argument(
        "dst_model_path",
        type=os.path.expanduser,
        help="Destination directory (must not exist)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Converter registry key (overrides forgather_arch in source config)",
    )
    parser.add_argument(
        "--from-version",
        type=int,
        default=None,
        help=(
            "Source schema version (overrides forgather_arch_version in "
            "source config; required when no metadata is stamped)"
        ),
    )
    parser.add_argument(
        "--to-version",
        type=int,
        default=None,
        help=(
            "Target schema version (default: the converter's current " "arch_version)"
        ),
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=os.path.expanduser,
        default=None,
        help=(
            "Path to source checkpoint directory (default: latest under "
            "SRC/checkpoints/, falling back to SRC root)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used during migration (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help=(
            "Override saved dtype (bfloat16, float16, float32). "
            "Default: keep checkpoint dtype."
        ),
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help=(
            "Allow missing/unexpected keys when loading the migrated "
            "state_dict (default: strict)"
        ),
    )
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Save weights in safetensors format (default: PyTorch)",
    )
    parser.add_argument(
        "--converter-path",
        action="append",
        dest="converter_paths",
        type=os.path.expanduser,
        default=[],
        help="Additional directory to search for model converters",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and report the migration plan; write nothing",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args(argv)


def _identify(
    src_config: Any,
    arch_override: Optional[str],
    from_version_override: Optional[int],
) -> Tuple[str, int]:
    """Determine the (arch, from_version) for the source model."""
    arch = arch_override or getattr(src_config, "forgather_arch", None)
    if not arch:
        raise ValueError(
            "Source model has no 'forgather_arch' field in config.json and "
            "--arch was not supplied. Pass --arch <name> to identify the "
            f"source schema. Registered converters: {list_converters()}"
        )

    if from_version_override is not None:
        from_version = from_version_override
    else:
        from_version = getattr(src_config, "forgather_arch_version", None)
        if from_version is None:
            raise ValueError(
                "Source model has no 'forgather_arch_version' field in "
                "config.json and --from-version was not supplied. Pass "
                "--from-version <int> to declare the source schema version."
            )
    return arch, int(from_version)


def _resolve_checkpoint(src_model_path: str, explicit: Optional[str]) -> str:
    if explicit:
        ckpt = os.path.abspath(explicit)
        if not os.path.isdir(ckpt):
            raise ValueError(f"Checkpoint path does not exist: {ckpt}")
        return ckpt
    latest = find_latest_checkpoint(src_model_path)
    if latest:
        return os.path.abspath(latest)
    # No checkpoints/ subdir: weights live at the source root (the
    # finalize-style "root copy" layout).
    return os.path.abspath(src_model_path)


def _config_to_project_args(
    converter: ModelConverter, config_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Pick the migrated config fields that the project template consumes.

    The Project receives kwargs whose names match the Forgather config
    field names declared in ``get_config_field_mapping("from_forgather")``.
    Missing fields fall back to the project's defaults.
    """
    field_mapping = converter.get_config_field_mapping("from_forgather")
    project_args: Dict[str, Any] = {}
    for fg_field in field_mapping.keys():
        if fg_field in config_dict and config_dict[fg_field] is not None:
            project_args[fg_field] = config_dict[fg_field]
    return project_args


def _apply_migrations(
    converter: ModelConverter,
    config_dict: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
    from_version: int,
    to_version: int,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], List[str]]:
    """Walk the migration chain end-to-end."""
    chain = compose_migration_chain(converter, from_version, to_version)
    descriptions: List[str] = []
    for src_v, step in chain:
        logger.info(f"Applying migration {src_v}->{src_v + 1}: {step.description}")
        config_dict = step.migrate_config(config_dict)
        if step.param_subs:
            state_dict = remap_state_dict(state_dict, step.param_subs)
        if step.transform_state_dict is not None:
            state_dict = step.transform_state_dict(state_dict, config_dict)
        config_dict["forgather_arch_version"] = src_v + 1
        descriptions.append(f"{src_v}->{src_v + 1}: {step.description}")
    return config_dict, state_dict, descriptions


def _copy_passthrough_files(src: str, dst: str) -> List[str]:
    copied: List[str] = []
    for name in _PASSTHROUGH_FILES:
        src_path = os.path.join(src, name)
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(dst, name)
        if os.path.exists(dst_path):
            # The regen path already produced one; leave it in place.
            continue
        shutil.copy2(src_path, dst_path)
        copied.append(dst_path)
    return copied


def _write_audit(
    dst: str,
    src_model_path: str,
    arch: str,
    from_version: int,
    to_version: int,
    migration_descriptions: List[str],
    dtype: Optional[str],
    missing_keys: List[str],
    unexpected_keys: List[str],
) -> str:
    audit = {
        "schema": "forgather_update.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": os.path.abspath(src_model_path),
        "arch": arch,
        "from_version": from_version,
        "to_version": to_version,
        "migrations": migration_descriptions,
        "dtype": dtype,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }
    path = os.path.join(dst, "forgather_update.json")
    with open(path, "w") as f:
        json.dump(audit, f, indent=2)
        f.write("\n")
    return path


def update(args) -> int:
    src = os.path.abspath(args.src_model_path)
    dst = os.path.abspath(args.dst_model_path)

    if not os.path.isdir(src):
        logger.error(f"Source directory does not exist: {src}")
        return 1
    if os.path.exists(dst):
        logger.error(f"Destination already exists: {dst}")
        return 1
    if not os.path.isdir(os.path.dirname(dst)):
        logger.error(
            f"Destination parent directory does not exist: {os.path.dirname(dst)}"
        )
        return 1

    # ---- 1. Discover converters ---------------------------------------
    discover_and_register_converters(args.converter_paths or None, forgather_root)
    available = list_converters()
    logger.info(f"Available converters: {available}")
    if not available:
        logger.error("No model converters found")
        return 1

    # ---- 2. Identify arch + versions ----------------------------------
    src_config = AutoConfig.from_pretrained(src, trust_remote_code=True)
    arch, from_version = _identify(src_config, args.arch, args.from_version)
    converter_cls = get_converter(arch)
    # Concrete converters (LlamaConverter, MistralConverter, ...) take no
    # init args; the base class signature is just for typing.
    converter = converter_cls()  # type: ignore[call-arg]
    if not hasattr(converter, "get_project_info"):
        raise ValueError(
            f"Converter for arch '{arch}' does not implement get_project_info(); "
            "forgather update requires the converter to know how to regenerate "
            "Forgather model code (typical for HFConverter subclasses)."
        )

    to_version = (
        args.to_version if args.to_version is not None else converter.arch_version
    )
    logger.info(f"Arch: {arch}")
    logger.info(f"Migrating: v{from_version} -> v{to_version}")

    if from_version == to_version:
        logger.info("Source already at target schema version; nothing to migrate")

    # Pre-resolve the chain so a missing step fails before we touch weights.
    chain_preview = compose_migration_chain(converter, from_version, to_version)
    for src_v, step in chain_preview:
        logger.info(f"  step {src_v}->{src_v + 1}: {step.description}")

    # ---- 3. Resolve dtype ---------------------------------------------
    if args.dtype is not None:
        dtype_str = args.dtype
    elif getattr(src_config, "dtype", None) is not None:
        dtype_str = str(src_config.dtype).replace("torch.", "")
    elif getattr(src_config, "torch_dtype", None) is not None:
        dtype_str = str(src_config.torch_dtype).replace("torch.", "")
    else:
        dtype_str = None
    new_dtype = torch_dtype(dtype_str) if dtype_str else None
    logger.info(f"DType: {new_dtype}")

    # ---- 4. Load source weights ---------------------------------------
    ckpt = _resolve_checkpoint(src, args.checkpoint)
    logger.info(f"Source checkpoint: {ckpt}")

    with ExitStack() as stack:
        if new_dtype is not None:
            stack.enter_context(default_dtype(new_dtype))
        stack.enter_context(torch.device(args.device))
        stack.enter_context(no_init_weights())
        src_model = AutoModelForCausalLM.from_config(src_config, trust_remote_code=True)

    if os.path.abspath(ckpt) == src:
        # No checkpoint subdir: AutoModel.from_pretrained reads weights at root.
        src_model = AutoModelForCausalLM.from_pretrained(src, trust_remote_code=True)
    else:
        load_checkpoint(ckpt, src_model, device=args.device, assign=True, strict=True)

    src_state_dict = src_model.state_dict()
    logger.info(f"Loaded {len(src_state_dict)} parameters from source")

    # ---- 5. Apply migrations ------------------------------------------
    config_dict = src_config.to_dict()
    # Drop fields that the regenerated config will rebuild itself.
    config_dict = {k: v for k, v in config_dict.items() if k != "auto_map"}

    migrated_config, migrated_state_dict, migration_descriptions = _apply_migrations(
        converter, config_dict, src_state_dict, from_version, to_version
    )

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN: skipping all writes")
        logger.info(f"Would create: {dst}")
        for line in migration_descriptions:
            logger.info(f"  migration {line}")
        return 0

    # ---- 6. Materialise destination -----------------------------------
    project_info = converter.get_project_info()  # type: ignore[attr-defined]
    project_args = _config_to_project_args(converter, migrated_config)

    # Carry tokenizer source through so the regenerated tokenizer matches
    # the source model. The tokenizer files saved at SRC are valid tokenizer
    # directories, so we point the project at SRC for tokenizer construction.
    proj = Project(
        config_name=project_info["config_name"],
        project_dir=project_info["project_dir"],
        output_dir=dst,
        tokenizer_id_or_path=src,
        **project_args,
    )

    new_config, new_tokenizer, new_model_ctor = proj(
        "pretrained_config", "pretrained_tokenizer", "model"
    )

    # Vocab size from saved config is authoritative (handles added tokens).
    if (
        getattr(src_config, "vocab_size", None) is not None
        and src_config.vocab_size != new_config.vocab_size
    ):
        logger.info(
            f"Adjusting vocab_size {new_config.vocab_size} -> {src_config.vocab_size}"
        )
        new_config.vocab_size = src_config.vocab_size

    # Stamp current schema metadata.
    new_config.forgather_arch = arch
    new_config.forgather_arch_version = to_version

    # Carry Forgather-specific fields the project template doesn't already plumb.
    for carry in ("hf_model_type",):
        value = getattr(src_config, carry, None)
        if value is not None and not hasattr(new_config, carry):
            setattr(new_config, carry, value)
        elif value is not None:
            setattr(new_config, carry, value)
    if dtype_str is not None:
        new_config.dtype = dtype_str

    # ---- 7. Construct + load -----------------------------------------
    with ExitStack() as stack:
        if new_dtype is not None:
            stack.enter_context(default_dtype(new_dtype))
        stack.enter_context(torch.device(args.device))
        stack.enter_context(no_init_weights())
        new_model = new_model_ctor()

    strict = not args.no_strict
    result = new_model.load_state_dict(migrated_state_dict, strict=strict, assign=True)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    logger.info(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        logger.info(f"  missing: {missing}")
    if unexpected:
        logger.info(f"  unexpected: {unexpected}")

    if hasattr(new_config, "tie_word_embeddings") and new_config.tie_word_embeddings:
        if hasattr(new_model, "tie_weights"):
            new_model.tie_weights()

    # ---- 8. Save ------------------------------------------------------
    new_config.save_pretrained(save_directory=dst)
    new_tokenizer.save_pretrained(save_directory=dst)
    save_checkpoint(
        output_dir=dst,
        module=new_model,
        safetensors=args.safetensors,
        include_param_sharing=True,
    )
    copied = _copy_passthrough_files(src, dst)
    for path in copied:
        logger.info(f"Copied passthrough: {path}")

    audit_path = _write_audit(
        dst,
        src,
        arch,
        from_version,
        to_version,
        migration_descriptions,
        dtype_str,
        missing,
        unexpected,
    )
    logger.info(f"Wrote audit log: {audit_path}")
    logger.info("=" * 60)
    logger.info(f"Done. Updated model at: {dst}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    return update(args)


if __name__ == "__main__":
    sys.exit(main())
