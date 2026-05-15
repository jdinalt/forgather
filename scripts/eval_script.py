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
    parser.add_argument(
        "--trainer",
        choices=["ddp", "simple", "pipeline"],
        default="ddp",
        help="Trainer backend",
    )
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path")
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Skip checkpoint resume; load via from_pretrained on the model dir",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write evals/. Defaults to --model.",
    )
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


def _model_is_quantized(model_path: str) -> bool:
    """True iff ``<model_path>/config.json`` carries a ``quantization_config`` block.

    Written by ``forgather finalize --quantize``; HF's
    ``from_pretrained()`` consumes it to install the TorchAoHfQuantizer
    pre-process path. Unreadable or malformed config.json -> False.
    """
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path) as f:
            return "quantization_config" in json.load(f)
    except (OSError, ValueError):
        return False


def resolve_checkpoint(args):
    """Return (checkpoint_arg, use_checkpoint) for the trainer.

    ``checkpoint_arg`` is what we pass as ``resume_from_checkpoint``:
    - True: auto-find latest
    - str path: explicit
    - False: do not resume

    Quantized models force ``(False, False)`` regardless of flags: only HF
    `from_pretrained()` runs the TorchAoHfQuantizer pre-process that
    installs the right quantized linear modules before weight load. The
    checkpoint-resume path uses `from_config()` + `load_state_dict()`,
    which has no quantizer hook and crashes with `'Parameter' object has
    no attribute 'tensor_data_names'` on torchao tensor subclasses.

    Cache the result on ``args`` so repeated calls don't re-log.
    """
    cached = getattr(args, "_resolved_checkpoint", None)
    if cached is not None:
        return cached

    if _model_is_quantized(args.model):
        if args.checkpoint:
            logger.warning(
                "Detected quantization_config in model config; ignoring "
                "--checkpoint %r and loading via from_pretrained().",
                args.checkpoint,
            )
        else:
            logger.info(
                "Detected quantization_config in model config; "
                "loading via from_pretrained() (overrides default checkpoint resume)."
            )
        result = (False, False)
    elif args.no_checkpoint:
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
    from forgather.ml.loss import CausalLoss

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
        f"wall_time:        {record['wall_time_s']:.2f} s",
        "=" * 72,
    ]
    return "\n".join(lines)


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

    start = time.time()
    metrics = trainer.evaluate()
    wall_time = time.time() - start

    eval_loss = float(metrics.get("eval_loss", float("nan")))

    # Persist results (rank 0 only).
    if is_rank_zero:
        from forgather.ml.analysis.metrics import get_perplexity

        now = datetime.now(UTC)
        result.eval_loss = eval_loss
        result.perplexity = get_perplexity(eval_loss)
        result.wall_time_s = wall_time
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
