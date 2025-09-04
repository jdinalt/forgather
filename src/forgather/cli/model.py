import os
import argparse
from argparse import RawTextHelpFormatter
from pprint import pformat


from forgather.config import ConfigEnvironment
from forgather.ml.datasets import plot_token_length_histogram
from forgather import Project

from .dynamic_args import get_dynamic_args
from .utils import write_output, write_output_or_edit


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

    return proj("pretrained_config", "pretrained_tokenizer", "pretrained_model_ctor")


def model_cmd(args):
    """Model test commands."""
    if hasattr(args, "model_subcommand"):
        match args.model_subcommand:
            case "construct":
                model_construct_cmd(args)
            case "test":
                model_test_cmd(args)


def model_construct_cmd(args):
    config, tokenizer, model_ctor = load_model(args)
    data = f"{'Model Configuration':-^80}\n" + pformat(config) + "\n\n"
    data += f"{'Model Tokenizer':-^80}\n" + pformat(tokenizer) + "\n\n"

    import torch
    from forgather.ml.utils import count_parameters

    with torch.device(args.device):
        model = model_ctor()
    model_header = f"Model on '{args.device}' device"
    data += f"{model_header:-^80}\n" + pformat(model) + "\n\n"
    data += f"parameters={count_parameters(model)}\n"
    write_output_or_edit(args, data, ".txt")


def model_test_cmd(args):
    config, tokenizer, model_ctor = load_model(args)
    print(tokenizer.vocab_size)

    import torch

    with torch.device(args.device):
        model = model_ctor()
        batch = torch.randint(
            0, tokenizer.vocab_size, (args.batch_size, args.sequence_length)
        )

    print(f"Testing model with batch: {batch.shape}")
    loss, logits = model(input_ids=batch, labels=batch)

    print(f"loss = {loss}")
    print(f"logits = {logits.shape}")

    loss.backward()
    print("Model completed forward and backward steps")
