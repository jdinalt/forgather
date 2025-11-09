import os
import argparse
from argparse import RawTextHelpFormatter

import torch
from transformers import AutoTokenizer

from forgather.config import ConfigEnvironment
from forgather.ml.datasets import plot_token_length_histogram
from forgather import Project

from .dynamic_args import get_dynamic_args
from .utils import write_output, write_output_or_edit


def dataset_cmd(args):
    config_name = args.config_template
    project_args = dict(
        tokenizer_path=args.tokenizer_path,
    )

    if args.config_template is None:
        args.config_template = ""

    # Merge in dynamic args
    project_args |= get_dynamic_args(args)
    proj = Project(
        config_name=args.config_template, project_dir=args.project_dir, **project_args
    )
    proj_meta = proj("meta")
    config_class = proj_meta["config_class"]
    main_feature = proj_meta["main_feature"]
    features = args.features
    if not features:
        features = [main_feature]

    if config_class != "type.dataset":
        raise TypeError(f"Expected class type.dataset, found {config_class}")

    data = ""
    if args.pp:
        data += "Preprocessed configuration:\n" + proj.pp_config + "\n"

    template_args = dict(
        tokenizer=None,
        preprocess_args=dict(),
    )

    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        data += "Tokenizer:\n" + repr(tokenizer) + "\n"
        template_args["tokenizer"] = tokenizer

    split = proj(args.target, **template_args)

    if args.histogram:
        assert args.tokenizer_path, "Tokenizer must be provided to plot histogram"
        args.project_dir
        args.config_template
        cfg_name, _ = os.path.splitext(os.path.basename(proj.config_name))
        cfg_name += ".svg"
        histogram_path = os.path.join(os.path.realpath(args.project_dir), cfg_name)
        print(f"Generating token-length histogram: {histogram_path}")
        if not args.tokenized:
            plot_token_length_histogram(
                split,
                tokenizer=tokenizer,
                sample_size=args.histogram_samples,
                feature=main_feature,
                min=None,
                max=None,
                output_file=histogram_path,
            )
        else:
            plot_token_length_histogram(
                split,
                tokenizer=None,
                sample_size=args.histogram_samples,
                feature="input_ids",
                min=None,
                max=None,
                output_file=histogram_path,
            )

    if args.examples:
        print(f"Printing {args.examples} examples from the dataset:")
        if args.tokenized:
            assert tokenizer, "Decoding a tokenized dataset requires the tokenizer"
            for i, example in zip(range(args.examples), split):
                input_ids = example["input_ids"]
                n_documents = (
                    (torch.tensor(input_ids) == tokenizer.bos_token_id).sum().item()
                )
                header = f" {i} Tokens: {len(input_ids)}, Documents: {n_documents} "
                data += f"{header:-^80}" + "\n" + tokenizer.decode(input_ids) + "\n"

        else:
            print("Dumping raw examples.")
            for i, example in zip(range(args.examples), split):
                data += f"{i:-^80}\n"
                for feature in features:
                    data += f"{feature:*^16}\n\n" + str(example[feature]) + "\n"
    write_output_or_edit(args, data, ".txt")
