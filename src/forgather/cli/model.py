from contextlib import ExitStack
from functools import partial
from pprint import pformat
import os
import shutil

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, IterableDataset

from forgather import Project, from_project
from forgather.ml.construct import torch_dtype
from forgather.ml.data_collator import DataCollatorForCausalLM
from forgather.ml.no_init_weights import no_init_weights
from forgather.ml.sharded_checkpoint import load_checkpoint, save_checkpoint
from forgather.ml.utils import count_parameters, default_dtype

from .dynamic_args import get_dynamic_args
from .utils import assert_project_class, write_output_or_edit


def optimizer_hook(optimizer, name, parameter):
    optimizer.step()
    optimizer.zero_grad()


def load_model(args):
    config_name = args.config_template
    if args.config_template is None:
        args.config_template = ""

    project_args = get_dynamic_args(args)
    proj = Project(
        config_name=args.config_template, project_dir=args.project_dir, **project_args
    )
    proj_meta = proj("meta")
    config_class = proj_meta["config_class"]
    if config_class != "type.model":
        raise TypeError(f"Expected class type.model, found {config_class}")

    output_dir = proj_meta["output_dir"]
    if os.path.exists(output_dir):
        if args.refresh_model:
            if not os.path.isdir(output_dir):
                raise NotADirectoryError("The model's output path is not a directory!?")
            print(f"Deleting output directory at '{output_dir}'")
            shutil.rmtree(output_dir)
        else:
            print(f"Model definition already exist at '{output_dir}'. If you wish to rebuild the model from source code, pass '--refresh-model'")
    
    return proj("pretrained_config", "pretrained_tokenizer", "model"), proj_meta


def model_cmd(args):
    """Model test commands."""
    assert_project_class(args, "type.model")

    print(f"{args=}")
    if hasattr(args, "model_subcommand"):
        match args.model_subcommand:
            case "construct":
                model_construct_cmd(args)
            case "test":
                model_test_cmd(args)


def construct_model(model_ctor, args, meta):
    output_dir = meta["output_dir"]

    with ExitStack() as exit_stack:
        exit_stack.enter_context(torch.device(args.device))
        if args.dtype:
            exit_stack.enter_context(default_dtype(torch_dtype(args.dtype)))
        if args.no_init_weights:
            exit_stack.enter_context(no_init_weights())
        model = model_ctor()
    if args.load_from_checkpoint:
        assert (
            args.device != "meta"
        ), "Load from checkpoint is not supported on meta device. Please specify a real device. e.g. '--device cpu'"
        print(f"Loading model from checkpoint {args.load_from_checkpoint}...")
        load_checkpoint(
            args.load_from_checkpoint, model, device=args.dtype, strict=True
        )
    if args.save_checkpoint:
        assert (
            args.device != "meta"
        ), "Saving model from meta device is unsupported. Please specify a real device. e.g. '--device cpu'"
        print("Saving model weights...")
        save_checkpoint(output_dir, module=model, safetensors=args.safetensors)
    
    if args.gradient_checkpointing:
        assert hasattr(model, "gradient_checkpointing_enable")
        print(f"Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    return model


def model_construct_cmd(args):
    (config, tokenizer, model_ctor), meta = load_model(args)
    data = f"{'Model Configuration':-^80}\n" + pformat(config) + "\n\n"
    data += f"{'Model Tokenizer':-^80}\n" + pformat(tokenizer) + "\n\n"

    model = construct_model(model_ctor, args, meta)
    model_header = f"Model on '{args.device}' device"
    data += f"{model_header:-^80}\n" + pformat(model) + "\n\n"
    data += f"parameters={count_parameters(model)}\n"
    write_output_or_edit(args, data, ".txt")


class RandomDatasetIterator(IterableDataset):
    """
    Yields random sequence of ints
    """

    def __init__(self, vocab_size, sequence_length):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

    def __next__(self):
        return dict(
            input_ids=torch.randint(
                0, self.vocab_size, (self.sequence_length,), device="cpu"
            ).tolist()
        )

    def __iter__(self):
        return self


def model_test_cmd(args):
    torch.autograd.set_detect_anomaly(True)
    torch._dynamo.config.recompile_limit = 32
    (config, tokenizer, model_ctor), meta = load_model(args)

    model = construct_model(model_ctor, args, meta)
    print(f"Setting learning-rate={args.lr}")
    optimizer = SGD(model.parameters(), lr=args.lr)

    if args.fuse_optim_with_backward:
        for name, p in model.named_parameters():
            if p.requires_grad:
                hook = partial(
                    optimizer_hook,
                    optimizer,
                    name,
                )
                p.register_post_accumulate_grad_hook(hook)

    if args.dataset_project:
        print(f"Loading dataset: {args.dataset_project}:{args.dataset_config}")
        dataset = from_project(
            project_dir=args.dataset_project,
            config_template=args.dataset_config,
            targets="train_dataset",
            preprocess_args=dict(),
            tokenizer=tokenizer,
        )
    else:
        dataset = RandomDatasetIterator(
            tokenizer.vocab_size,
            args.sequence_length,
        )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        return_tensors="pt",
        truncation=True,
        max_length=args.sequence_length,
        padding="max_length",
        packed_sequences=args.packed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )

    model.train()
    for i, batch in enumerate(dataloader):
        if i == args.steps:
            break

        if args.device == "meta":
            batch = {
                key: torch.empty_like(value, device="meta")
                for key, value in batch.items()
            }
        else:
            batch = {key: value.to(device=args.device) for key, value in batch.items()}

        if i == 0:
            print(f"{batch.keys()=}")

        loss, logits = model(**batch)
        print(f"step: {i+1}, loss: {loss}, logits.shape: {logits.shape}")

        loss.backward()
        if not args.fuse_optim_with_backward:
            optimizer.step()
        optimizer.zero_grad()

    print("Test Completed")
