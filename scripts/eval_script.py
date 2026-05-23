"""Standalone evaluation entrypoint used by `forgather eval test`.

Unlike `train_script.py`, this script does NOT load a training_script project.
Instead, it:

1. Loads a lightweight eval-config project to obtain the ``eval_dataset`` and
   ``data_collator``.
2. Loads the tokenizer directly via ``AutoTokenizer.from_pretrained(model_path)``.
3. Builds a ``model_init`` closure (mirrors tools/inference_server/service.py)
   and constructs the chosen trainer class directly in Python — no project is
   used for the trainer or model.
4. Calls ``trainer.evaluate()`` and writes a JSON results file plus a
   human-readable summary.

Launch via torchrun:

    torchrun --standalone --nproc-per-node N scripts/eval_script.py \
        --eval-project PATH --eval-config NAME --model PATH [...]
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from forgather.cli.eval_args import add_eval_script_args
from forgather.eval_config import EvalResult, TestConfig

logger = logging.getLogger(__name__)


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a named eval config",
    )
    parser.add_argument(
        "--eval-project", required=True, help="Path to eval-config project"
    )
    parser.add_argument("--eval-config", required=True, help="Config template name")
    parser.add_argument("--model", required=True, help="Path to model directory")
    # The args shared with `forgather eval test` are defined once in
    # forgather.cli.eval_args and registered here via add_eval_script_args.
    add_eval_script_args(parser)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def init_logging(level_name):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_model_init(model_path, dtype, attn_implementation, device, use_checkpoint):
    """Return a no-arg callable that constructs the model.

    Matches the inference-server pattern: when loading from a checkpoint, build
    the model structure with no_init_weights() on the target device; the trainer
    will load the checkpoint weights afterwards. When ``use_checkpoint=False``,
    the model is loaded directly via ``from_pretrained``.
    """
    from forgather.ml.no_init_weights import no_init_weights
    from forgather.ml.utils import default_dtype

    def init_model():
        if use_checkpoint:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            with torch.device(device), default_dtype(dtype=dtype), no_init_weights():
                model = AutoModelForCausalLM.from_config(
                    cfg,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
            if torch.cuda.is_available():
                model = model.to(device)
        model.eval()
        return model

    return init_model


def resolve_checkpoint(args):
    """Return (checkpoint_arg, use_checkpoint) for the trainer.

    ``checkpoint_arg`` is what we pass as ``resume_from_checkpoint``:
    - True: auto-find latest
    - str path: explicit
    - False: do not resume

    Cache the result on ``args`` so repeated calls don't re-log.

    Quantization is handled transparently downstream: when the native
    loader at ``forgather.ml.sharded_checkpoint.load_checkpoint`` detects
    torchao quantization (via ``config.json`` or a state_dict scan), it
    installs the matching quantized linear modules before
    ``load_state_dict``. Eval doesn't need to special-case quantized
    models here.
    """
    cached = getattr(args, "_resolved_checkpoint", None)
    if cached is not None:
        return cached

    if args.no_checkpoint:
        result = (False, False)
    elif args.checkpoint:
        result = (args.checkpoint, True)
    else:
        result = (True, True)

    args._resolved_checkpoint = result
    return result


def build_trainer(args, model_init, eval_dataset, data_collator, tokenizer, device):
    """Construct the selected trainer directly, no project involved."""
    from forgather.ml.distributed import DistributedEnvironment
    from forgather.ml.loss import CausalLoss, LinearCrossEntropyLoss

    if args.fused_loss:
        fused_loss_factory = LinearCrossEntropyLoss
    else:
        fused_loss_factory = None

    checkpoint_arg, _ = resolve_checkpoint(args)
    output_dir = args.output_dir or args.model
    batch_size = args.batch_size
    loss_fn = CausalLoss()

    common = dict(
        output_dir=output_dir,
        device=device,
        per_device_eval_batch_size=batch_size,
        max_eval_steps=args.max_steps,
        resume_from_checkpoint=checkpoint_arg,
        construct_model_on="device",
        default_dtype=args.dtype,
        torch_compile=args.compile,
        save_strategy="no",
        save_safetensors=False,
        eval_strategy="no",
        logging_strategy="no",
        dataloader_num_workers=1,
    )

    init_distributed = lambda: DistributedEnvironment(backend="cuda:nccl,cpu:gloo")

    if args.trainer == "ddp":
        from forgather.ml.trainer.ddp import DDPTrainer, DDPTrainingArguments

        trainer_args = DDPTrainingArguments(**common)
        trainer = DDPTrainer(
            args=trainer_args,
            model_init=model_init,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_loss_func=loss_fn,
            distributed_env=init_distributed(),
            fused_loss_factory=fused_loss_factory,
        )
    elif args.trainer == "simple":
        from forgather.ml.distributed import from_env
        from forgather.ml.trainer.trainer import Trainer, TrainingArguments

        trainer_args = TrainingArguments(**common)
        trainer = Trainer(
            args=trainer_args,
            model_init=model_init,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_loss_func=loss_fn,
            distributed_env=from_env(),
            fused_loss_factory=fused_loss_factory,
        )
    elif args.trainer == "pipeline":
        from torch.distributed.pipelining import ScheduleGPipe

        from forgather.ml.trainer.pipeline import (
            PipelineTrainer,
            PipelineTrainingArguments,
            create_manual_causal_lm_splitter,
        )

        common["n_microbatches"] = 4

        distributed_env = DistributedEnvironment()
        trainer_args = PipelineTrainingArguments(**common)
        trainer = PipelineTrainer(
            args=trainer_args,
            model_init=model_init,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_loss_func=loss_fn,
            distributed_env=init_distributed(),
            pipe_schedule_factory=ScheduleGPipe,
            model_splitter=create_manual_causal_lm_splitter(),
            fused_loss_factory=fused_loss_factory,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown trainer: {args.trainer}")

    return trainer


def format_header(record):
    lines = [
        "=" * 72,
        f"Evaluation: {record['eval_name']}  ({record['config_name']})",
        "-" * 72,
        f"Model:            {record['model_path']}",
    ]
    if record.get("checkpoint_path"):
        lines.append(f"Checkpoint:       {record['checkpoint_path']}")
    lines.extend(
        [
            f"Dataset:          {record['dataset_proj']}  [{record['dataset_config']}]",
            f"Target:           {record['dataset_target']}",
            f"Trainer:          {record['trainer']}  (world_size={record['world_size']})",
            f"Batch size:       {record['batch_size']}  max_length={record['max_length']}",
            f"Dtype:            {record['dtype']}  attn={record['attn_implementation']}",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


def format_results(record):
    lines = [
        "=" * 72,
        f"eval_loss:        {record['eval_loss']:.6f}",
        f"perplexity:       {record['perplexity']:.4f}",
    ]
    if record.get("bpb") is not None:
        lines.append(f"bpb:              {record['bpb']:.4f}")
    if record.get("bpc") is not None:
        lines.append(f"bpc:              {record['bpc']:.4f}")
    if record.get("tokens_per_byte") is not None:
        lines.append(f"tokens_per_byte:  {record['tokens_per_byte']:.4f}")
    lines.append(f"wall_time:        {record['wall_time_s']:.2f} s")
    lines.append("=" * 72)
    return "\n".join(lines)


def _compute_corpus_stats(eval_dataset, tokenizer, max_examples, pad_token_id):
    """Count bytes / chars / predicted tokens over the eval dataset prefix.

    Iterates the *pre-collation* dataset, decoding each ``input_ids`` sequence
    to UTF-8 text and counting predicted-token positions. Returns
    ``(total_bytes, total_chars, total_predicted_tokens, n_examples)``.

    Predicted-position semantics match the trainer's causal loss: positions
    ``labels[1:]`` where the label is neither ``-100`` (explicit ignore) nor
    ``pad_token_id`` (the collator's pad-as-ignore convention, which has
    not been applied yet at pre-collation time).

    ``max_examples`` caps the iteration; ``<= 0`` means scan everything.
    """
    total_bytes = 0
    total_chars = 0
    total_predicted_tokens = 0
    n_examples = 0

    if hasattr(eval_dataset, "__len__"):
        ds_len = len(eval_dataset)
        if max_examples is None or max_examples <= 0:
            max_examples = ds_len
        else:
            max_examples = min(max_examples, ds_len)

    for ex in eval_dataset:
        if max_examples is not None and max_examples > 0 and n_examples >= max_examples:
            break
        input_ids = ex["input_ids"]
        labels = ex.get("labels", input_ids)
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        total_bytes += len(text.encode("utf-8"))
        total_chars += len(text)
        # Predicted positions: only labels[1:] contribute to causal loss
        # (position 0 is input-only). Drop both -100 (explicit ignore) and
        # pad — pre-padded examples (pipeline trainer uses padding=max_length)
        # would otherwise inflate the count vs. what the trainer counts.
        total_predicted_tokens += sum(
            1 for t in labels[1:] if t != -100 and t != pad_token_id
        )
        n_examples += 1

    return total_bytes, total_chars, total_predicted_tokens, n_examples


@record
def main():
    args = parse_args()
    init_logging(args.log_level)
    from forgather.ml.data_collator import DataCollatorForCausalLM
    from forgather.project import Project

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_rank_zero = rank == 0

    test_proj = Project(args.eval_config, args.eval_project)
    test_config = TestConfig(**test_proj())

    # Apply config defaults for unset CLI flags.
    if args.batch_size is None:
        args.batch_size = test_config.default_batch_size
    if args.max_length is None:
        args.max_length = test_config.default_max_length
    if args.stride is None:
        args.stride = test_config.default_stride

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset from project
    dataset_proj = Project(
        test_config.dataset_config,
        test_config.dataset_proj,
        max_length=args.max_length,
        stride=args.stride,
    )
    dataset_proj_meta: dict = dataset_proj("meta")
    dataset_config_class = dataset_proj_meta["config_class"]
    if dataset_config_class != "type.dataset":
        raise TypeError(f"Expected class type.dataset, found {dataset_config_class}")

    eval_dataset = dataset_proj(
        test_config.dataset_target, tokenizer=tokenizer, preprocess_args=dict()
    )

    # Device for model construction.
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    dtype = DTYPE_MAP[args.dtype]
    attn_impl = args.attn_implementation

    checkpoint_arg, use_checkpoint = resolve_checkpoint(args)
    model_init = build_model_init(args.model, dtype, attn_impl, device, use_checkpoint)

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        return_tensors="pt",
        max_length=args.max_length,
        padding="max_length" if args.trainer == "pipeline" else True,
    )

    trainer = build_trainer(
        args, model_init, eval_dataset, data_collator, tokenizer, device
    )

    if is_rank_zero:
        resolved_ckpt = (
            args.checkpoint
            if (args.checkpoint and use_checkpoint)
            else (
                getattr(trainer.args, "resume_from_checkpoint", None)
                if use_checkpoint
                else None
            )
        )
        if resolved_ckpt is True:
            resolved_ckpt = None  # trainer may leave as True if no checkpoint existed

        result = EvalResult.from_config(
            test_config,
            model_path=os.path.abspath(args.model),
            checkpoint_path=resolved_ckpt,
            batch_size=args.batch_size,
            max_length=args.max_length,
            stride=args.stride,
            dtype=args.dtype,
            attn_implementation=attn_impl,
            trainer=args.trainer,
            world_size=world_size,
        )
        print(format_header(asdict(result)))
    else:
        result = None

    # Pre-pass: compute byte/char/predicted-token counts so we can report
    # tokenizer-agnostic metrics (BPB/BPC) alongside the raw token PPL.
    #
    # Rebuild the dataset for this rank-0 pre-pass instead of iterating
    # ``eval_dataset`` in place — iterable datasets (ComposableIterableDataset,
    # HF streaming) carry an internal cursor, and a partial-iteration would
    # leave the trainer reading from the post-prepass position. Two
    # independent factory instances dodge the problem entirely.
    #
    # The example cap mirrors the trainer's max_eval_steps semantics. DDP
    # with the default ``dispatch_batches=True`` and the simple/pipeline
    # trainers all cap at ``max_steps * batch_size`` (one logical iterator
    # over the corpus); DDP ``dispatch_batches=False`` (all-shards) caps
    # per-rank, so the global cap is ``* world_size``.
    corpus_stats = None
    if is_rank_zero:
        per_rank_shards = (
            args.trainer == "ddp"
            and hasattr(trainer, "_dispatch_eval_batches")
            and not trainer._dispatch_eval_batches()
        )
        if args.max_steps and args.max_steps > 0:
            shard_factor = max(world_size, 1) if per_rank_shards else 1
            max_examples = args.max_steps * args.batch_size * shard_factor
        else:
            max_examples = 0  # signal "all examples"
        prepass_dataset = dataset_proj(
            test_config.dataset_target, tokenizer=tokenizer, preprocess_args=dict()
        )
        corpus_stats = _compute_corpus_stats(
            prepass_dataset, tokenizer, max_examples, tokenizer.pad_token_id
        )

    start = time.time()
    metrics = trainer.evaluate()
    wall_time = time.time() - start

    eval_loss = float(metrics.get("eval_loss", float("nan")))

    # Persist results (rank 0 only).
    if is_rank_zero:
        from forgather.ml.analysis.metrics import get_bpb, get_bpc, get_perplexity

        now = datetime.now(UTC)
        result.eval_loss = eval_loss
        result.perplexity = get_perplexity(eval_loss)
        result.wall_time_s = wall_time

        # Fill in tokenizer-agnostic metrics from the pre-pass counts.
        # Assumes ``eval_loss`` ≈ true token-mean cross-entropy. The trainer
        # reports a mean of per-step-mean losses; when batches contain equal
        # numbers of valid (non-ignored) tokens — the common case for eval
        # with fixed ``max_length`` and full-length sequences — this equals
        # the token-weighted mean exactly. Variable-length batches introduce
        # a small (typically <1%) approximation error.
        if corpus_stats is not None:
            total_bytes, total_chars, total_predicted_tokens, _ = corpus_stats
            result.total_bytes = total_bytes
            result.total_chars = total_chars
            result.total_predicted_tokens = total_predicted_tokens
            if total_bytes > 0 and total_predicted_tokens > 0:
                tokens_per_byte = total_predicted_tokens / total_bytes
                result.tokens_per_byte = tokens_per_byte
                result.bpb = get_bpb(eval_loss, tokens_per_byte)
                if total_chars > 0:
                    tokens_per_char = total_predicted_tokens / total_chars
                    result.bpc = get_bpc(eval_loss, tokens_per_char)
        # UTC timestamp in ISO 8601 with a single trailing "Z" suffix.
        result.timestamp = now.replace(tzinfo=None).isoformat() + "Z"

        output_root = Path(args.output_dir or args.model)
        eval_name = result.eval_name or "eval"
        ts = now.strftime("%Y%m%dT%H%M%S")
        run_dir = output_root / "evals" / f"{eval_name}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        print(format_results(asdict(result)))
        print(f"\nWrote: {out_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
