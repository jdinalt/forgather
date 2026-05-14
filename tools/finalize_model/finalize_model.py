#!/usr/bin/env python3
"""
Finalize a trained Forgather model into a clean handoff directory.

Produces a destination directory containing:
    - The model source code (``*.py``)
    - The tokenizer
    - ``config.json``
    - A chat template (when provided)
    - A ``generation_config.json`` (synthesized when missing)
    - Exactly one preserved checkpoint (latest by default)
    - Optionally, the optimizer state from the source checkpoint

Scheduler, dataset, RNG, and trainer state from the source training run
are always dropped.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from argparse import RawTextHelpFormatter
from typing import List, Optional

# Add forgather to path when invoked as a standalone script.
from forgather import MetaConfig

forgather_root = MetaConfig.find_workspace_dir(os.path.abspath(__file__))
if forgather_root and forgather_root not in sys.path:
    sys.path.insert(0, forgather_root)

from forgather.ml.model_conversion.finalize import (
    build_generation_config,
    copy_model_source,
    detect_stop_token_ids,
    load_source_artifacts,
    resolve_checkpoint_dir,
    write_finalized_checkpoint,
    write_generation_config,
)
from forgather.ml.model_conversion.resize_embeddings import (
    DEFAULT_TOKEN_CONFIG,
    add_tokens_to_tokenizer,
    resize_word_embeddings,
    update_config_from_tokenizer,
)

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Finalize a trained Forgather model into a clean directory",
        epilog=(
            "Examples:\n"
            "\n"
            "Duplicate latest checkpoint into a clean directory:\n"
            "  forgather finalize output_models/wds out/wds_final\n"
            "\n"
            "Add ChatML tokens, set chat template, generate config from preset:\n"
            "  forgather finalize output_models/wds out/wds_chatml \\\n"
            "      --add-tokens chatml.yaml -t chatml.jinja \\\n"
            "      --generation-config precise\n"
            "\n"
            "Preserve optimizer state for warm-start fine-tuning:\n"
            "  forgather finalize output_models/wds out/wds_final --keep-optimizer\n"
            "\n"
            "Use a specific (non-latest) checkpoint:\n"
            "  forgather finalize output_models/wds out/wds_final \\\n"
            "      -c output_models/wds/checkpoints/checkpoint-385440\n"
        ),
    )
    parser.add_argument(
        "source",
        type=os.path.expanduser,
        help="Source model directory (a Forgather output_models/X tree, or a flat HF model dir)",
    )
    parser.add_argument(
        "dest",
        type=os.path.expanduser,
        help="Destination directory (must not exist)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=os.path.expanduser,
        default=None,
        help=(
            "Path to the source checkpoint directory.\n"
            "If omitted, the latest checkpoint under SOURCE/checkpoints/ is used.\n"
            "If neither resolves, weights are loaded from SOURCE itself."
        ),
    )
    parser.add_argument(
        "--add-tokens",
        type=os.path.expanduser,
        default=None,
        help="Path to a YAML file specifying tokens to add to the vocabulary",
    )
    parser.add_argument(
        "--skip-default-tokens",
        action="store_true",
        help="Skip default token handling (otherwise [PAD] is added if missing)",
    )
    parser.add_argument(
        "-t",
        "--chat-template-path",
        type=os.path.expanduser,
        default=None,
        help="Path to a Jinja2 chat template file to apply to the tokenizer",
    )
    parser.add_argument(
        "--no-auto-stop-tokens",
        action="store_true",
        help=(
            "Disable auto-detection of end-of-turn tokens (im_end / eot / "
            "end_of_turn) when --add-tokens introduces them."
        ),
    )
    parser.add_argument(
        "--stop-tokens",
        type=str,
        default="",
        help='Comma-separated explicit stop-token strings (e.g. "<|stop|>,<|end|>")',
    )
    parser.add_argument(
        "--generation-config",
        type=str,
        default="carry",
        help=(
            "How to produce the destination generation_config.json:\n"
            "  carry  - copy source's if present, else synthesize minimal (default)\n"
            "  none   - skip generation_config.json entirely\n"
            "  PATH   - path to a JSON file (Forgather inference-preset format:\n"
            "           keys like max_tokens, temperature, top_p, repetition_penalty)\n"
            "  NAME   - bare name resolved against ~/.config/forgather/generation_config/\n"
            "           NAME.json. No presets ship with this branch; populate that\n"
            "           directory yourself or pass an explicit PATH."
        ),
    )
    parser.add_argument(
        "--keep-optimizer",
        action="store_true",
        help="Copy optimizer_state.pt from the source checkpoint into the dest checkpoint",
    )
    parser.add_argument(
        "--root-copy",
        action="store_true",
        help=(
            "Write weights only at the model root and skip creating "
            "DEST/checkpoints/. Mutually exclusive with --keep-optimizer."
        ),
    )
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help=(
            "Save weights in safetensors format. Default is PyTorch (.bin), which "
            "natively handles tied embeddings; safetensors raises on tied weights."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Override saved dtype (bfloat16, float16, float32). Default: keep checkpoint dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the model onto during finalize (default: cpu)",
    )
    parser.add_argument(
        "--qat-convert",
        type=str,
        default=None,
        help=(
            "Run the torchao QAT convert step before saving: swaps "
            "FakeQuantizedLinear modules for real low-bit quantized linear "
            "ops. Pass the same recipe string that was used at training "
            "time (e.g. 'int8-dynamic-act-int4-weight'). On models that "
            "were not QAT-trained this is a no-op with a warning. See "
            "docs/trainers/qat-training.md."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and report what would be done; do not write anything",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args(argv)


def _resolve_dtype(dtype_str: Optional[str]):
    if dtype_str is None:
        return None
    from forgather.ml.construct import torch_dtype

    return torch_dtype(dtype_str)


def _apply_qat_convert(model, recipe: str) -> None:
    """Run torchao's QAT convert step in-place on a loaded model.

    Swaps every ``FakeQuantizedLinear`` for the real low-bit quantized linear
    op described by ``recipe``. If the model has no fake-quantized modules
    (i.e. it was not QAT-trained), logs a warning and returns without
    modifying the model so finalize still produces a valid (just non-
    quantized) artifact.
    """
    from torchao.quantization import quantize_
    from torchao.quantization.qat import FakeQuantizedLinear, QATConfig

    from forgather.ml.qat_recipes import QAT_RECIPES, recipe_to_base_config

    if recipe not in QAT_RECIPES:
        raise ValueError(
            f"--qat-convert must be one of {QAT_RECIPES}, got {recipe!r}"
        )

    fq_count = sum(1 for m in model.modules() if isinstance(m, FakeQuantizedLinear))
    if fq_count == 0:
        logger.warning(
            "--qat-convert %r requested but model has no FakeQuantizedLinear "
            "modules. Was this model trained with --qat-recipe? Skipping "
            "convert step; the saved artifact will be the un-quantized model.",
            recipe,
        )
        return

    base_config = recipe_to_base_config(recipe)
    quantize_(model, QATConfig(base_config, step="convert"))
    logger.info(
        f"QAT convert ({recipe}): converted {fq_count} FakeQuantizedLinear "
        f"modules to real quantized linear ops"
    )


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    source = os.path.abspath(args.source)
    dest = os.path.abspath(args.dest)

    if not os.path.isdir(source):
        logger.error(f"Source directory does not exist: {source}")
        return 1
    if os.path.exists(dest):
        logger.error(f"Destination already exists: {dest}")
        return 1
    if args.root_copy and args.keep_optimizer:
        logger.error("--root-copy and --keep-optimizer are mutually exclusive")
        return 1

    logger.info("Forgather Finalize Model")
    logger.info("=" * 60)
    logger.info(f"Source:      {source}")
    logger.info(f"Destination: {dest}")

    # ---- 1. Resolve source checkpoint ----------------------------------
    ckpt_dir, step = resolve_checkpoint_dir(source, args.checkpoint)
    if ckpt_dir is None:
        logger.info(
            "No checkpoints/ directory found in source; loading weights from source root"
        )
    else:
        logger.info(
            f"Source checkpoint: {ckpt_dir}"
            + (f" (step {step})" if step is not None else "")
        )

    # ---- 2. Load source artifacts --------------------------------------
    dtype = _resolve_dtype(args.dtype)
    bundle = load_source_artifacts(source, ckpt_dir, dtype, args.device)
    model = bundle.model
    tokenizer = bundle.tokenizer
    config = bundle.config

    logger.info(f"Loaded {type(model).__name__}; vocab_size={len(tokenizer)}")

    # Snapshot the original EOS *before* any token additions; downstream code
    # in step 5 may call update_config_from_tokenizer which mutates
    # config.eos_token_id to the (possibly reassigned) tokenizer eos. We need
    # the unmutated original to seed detect_stop_token_ids so the merged
    # generation_config.eos_token_id list contains BOTH the original EOS and
    # any new ChatML / end-of-turn tokens.
    original_source_eos_id = getattr(config, "eos_token_id", None)
    if isinstance(original_source_eos_id, list):
        original_source_eos_id = (
            original_source_eos_id[0] if original_source_eos_id else None
        )

    # ---- 3. Determine token additions ----------------------------------
    token_config = args.add_tokens
    if token_config is None and not args.skip_default_tokens:
        logger.info("Using default token configuration (adds missing PAD)")
        token_config = DEFAULT_TOKEN_CONFIG
    elif args.skip_default_tokens:
        logger.info("Skipping default token handling")

    added_token_names: List[str] = []
    weights_changed = False

    if token_config is not None:
        # Track names being added so the stop-token detector can find them later.
        # Both the named-tokens path and the special_tokens / regular_tokens lists
        # contribute names.
        if isinstance(token_config, dict):
            for k in ("bos_token", "eos_token", "pad_token", "unk_token"):
                v = token_config.get(k)
                if isinstance(v, str):
                    added_token_names.append(v)
                elif isinstance(v, dict) and "token" in v:
                    added_token_names.append(v["token"])
            added_token_names.extend(token_config.get("special_tokens", []) or [])
            added_token_names.extend(token_config.get("regular_tokens", []) or [])
        elif isinstance(token_config, str):
            import yaml

            with open(token_config, "r") as f:
                yaml_doc = yaml.safe_load(f)
            for k in ("bos_token", "eos_token", "pad_token", "unk_token"):
                v = (yaml_doc or {}).get(k)
                if isinstance(v, str):
                    added_token_names.append(v)
                elif isinstance(v, dict) and "token" in v:
                    added_token_names.append(v["token"])
            added_token_names.extend((yaml_doc or {}).get("special_tokens", []) or [])
            added_token_names.extend((yaml_doc or {}).get("regular_tokens", []) or [])

        num_added, token_inits = add_tokens_to_tokenizer(tokenizer, token_config)
        logger.info(f"Added {num_added} token(s); new vocab size: {len(tokenizer)}")
        if num_added > 0:
            weights_changed = True
            resize_word_embeddings(model, tokenizer, token_inits)
            update_config_from_tokenizer(config, tokenizer)

    # ---- 4. Apply chat template ----------------------------------------
    if args.chat_template_path:
        tpl = os.path.abspath(args.chat_template_path)
        if not os.path.isfile(tpl):
            logger.error(f"Chat template not found: {tpl}")
            return 1
        with open(tpl, "r") as f:
            tokenizer.chat_template = f.read()
        logger.info(f"Chat template set from {tpl}")
    elif tokenizer.chat_template is None:
        logger.warning(
            "Tokenizer has no chat_template; consider providing one via -t before chat fine-tuning"
        )

    # ---- 5. Stop tokens for generation_config --------------------------
    user_stop = (
        [s for s in args.stop_tokens.split(",") if s.strip()]
        if args.stop_tokens
        else []
    )
    # Use the snapshot captured before update_config_from_tokenizer ran,
    # so the merged eos list always carries the original source EOS even
    # when --add-tokens reassigned tokenizer.eos_token to a new value.
    stop_token_ids = detect_stop_token_ids(
        tokenizer=tokenizer,
        source_eos_id=original_source_eos_id,
        added_token_names=added_token_names,
        user_stop_tokens=user_stop,
        auto=not args.no_auto_stop_tokens,
    )
    logger.info(f"Stop token IDs: {stop_token_ids}")

    # ---- Dry run? ------------------------------------------------------
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN: skipping all writes")
        logger.info(f"Would create: {dest}")
        if ckpt_dir and not args.root_copy:
            logger.info(
                f"Would write checkpoint at: {dest}/checkpoints/checkpoint-{step or 0}"
            )
        if weights_changed:
            logger.info("Vocabulary expansion would resize embeddings")
        if args.keep_optimizer:
            logger.info("Optimizer state would be carried")
        if args.generation_config != "none":
            logger.info(
                f"Would write generation_config.json (mode={args.generation_config})"
            )
        if args.qat_convert:
            logger.info(
                f"Would run QAT convert step with recipe '{args.qat_convert}'"
            )
        return 0

    # ---- 6. QAT convert (optional) -------------------------------------
    if args.qat_convert:
        _apply_qat_convert(model, args.qat_convert)
        if args.safetensors:
            # torchao's quantized tensor subclasses wrap multiple inner
            # tensors and do not expose a single .storage().data_ptr(),
            # so safetensors saves fail with "Attempted to access the
            # data pointer on an invalid python storage". Force .bin.
            logger.warning(
                "--safetensors is incompatible with QAT-converted models "
                "(torchao subclass tensors lack a single storage pointer). "
                "Saving as PyTorch (.bin) instead."
            )
            args.safetensors = False

    # ---- 7. Materialize destination ------------------------------------
    os.makedirs(dest, exist_ok=False)
    copy_model_source(source, dest)

    tokenizer.save_pretrained(dest)
    config.save_pretrained(dest)

    write_finalized_checkpoint(
        dest=dest,
        model=model,
        source_checkpoint_dir=ckpt_dir,
        step=step,
        safetensors=args.safetensors,
        keep_optimizer=args.keep_optimizer,
        root_copy=args.root_copy,
    )

    gen_cfg = build_generation_config(
        source=source,
        mode=args.generation_config,
        tokenizer=tokenizer,
        stop_token_ids=stop_token_ids,
    )
    if gen_cfg is not None:
        write_generation_config(dest, gen_cfg)

    logger.info("=" * 60)
    logger.info(f"Done. Finalized model at: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
