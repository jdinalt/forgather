import os

import torch
from transformers import AutoTokenizer

from forgather import Project
from forgather.ml.datasets import (
    InterleavedDataset,
    SimpleArrowIterableDataset,
    plot_token_length_histogram,
)

from .dynamic_args import get_dynamic_args
from .utils import assert_project_class, write_output_or_edit


def dataset_cmd(args):
    assert_project_class(args, "type.dataset")
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

    if args.num_shards is not None:
        print(
            f"Requesting shard_dataset: num_shards={args.num_shards}, index={args.shard_index}"
        )
        template_args["shard_dataset"] = dict(
            num_shards=args.num_shards,
            index=args.shard_index,
        )
    if args.select_range is not None:
        template_args["select_range"] = args.select_range

    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        data += "Tokenizer:\n" + repr(tokenizer) + "\n"
        template_args["tokenizer"] = tokenizer

    if args.seed is not None:
        template_args["shuffle"] = True
        template_args["seed"] = args.seed

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

    print(f"{split=}")

    if args.examples:
        stride = args.example_stride if args.example_stride else 1
        # Print incrementally if not outputting to file or editor
        print_incremental = args.output_file is None and not args.edit
        if print_incremental:
            print(data)
            data = ""

        print(f"Printing {args.examples} examples from the dataset (stride={stride}):")

        if args.tokenized:
            assert tokenizer, "Decoding a tokenized dataset requires the tokenizer"
            example_count = 0
            dataset_index = 0

            for example in split:
                # Check if this is an index we want to print
                if dataset_index % stride == 0 and example_count < args.examples:
                    input_ids = example["input_ids"]
                    document_starts = example.get("document_starts", None)
                    # Use explicit document boundaries if available (preferred)
                    if document_starts:
                        n_documents = len(document_starts)
                        print(f"Document Starts: {document_starts}")
                    # Fall back to counting EOS tokens (legacy, less reliable)
                    elif tokenizer.eos_token_id is not None:
                        n_documents = (
                            (torch.tensor(input_ids) == tokenizer.eos_token_id)
                            .sum()
                            .item()
                        )
                    else:
                        n_documents = "unknown"

                    header = f" {dataset_index} Tokens: {len(input_ids)}, Documents: {n_documents}, Features: {example.keys()}"

                    # Show estimated lengths, where relevant
                    if isinstance(split, SimpleArrowIterableDataset):
                        header += f", Estimated Len: {len(split)}"
                    elif isinstance(split, InterleavedDataset):
                        header += f", InterleavedDataset Lengths: {get_interleaved_lengths(split)}"

                    decoded_text = tokenizer.decode(input_ids)

                    # Apply truncation if specified
                    if args.truncate and len(decoded_text) > args.truncate:
                        decoded_text = decoded_text[: args.truncate] + "..."

                    output = f"{header:-^80}" + "\n" + decoded_text + "\n"
                    if print_incremental:
                        print(output)
                    else:
                        data += output
                    example_count += 1

                dataset_index += 1

                # Stop if we've printed enough examples
                if example_count >= args.examples:
                    break

        else:
            print("Dumping raw examples.")
            example_count = 0
            dataset_index = 0

            for example in split:
                # Check if this is an index we want to print
                if dataset_index % stride == 0 and example_count < args.examples:
                    header = f" {dataset_index} Features: {example.keys()}"
                    output = f"{header:-^80}" + "\n"
                    for feature in features:
                        feature_text = str(example[feature])

                        # Apply truncation if specified
                        if args.truncate and len(feature_text) > args.truncate:
                            feature_text = feature_text[: args.truncate] + "..."

                        output += f"{feature:*^16}\n\n" + feature_text + "\n"

                    if print_incremental:
                        print(output)
                    else:
                        data += output
                    example_count += 1

                dataset_index += 1

                # Stop if we've printed enough examples
                if example_count >= args.examples:
                    break
    write_output_or_edit(args, data, ".txt")


def get_interleaved_lengths(dataset) -> str:
    s = str(len(dataset)) + " ["
    for ds in dataset.datasets:
        if isinstance(ds, SimpleArrowIterableDataset):
            s += str(len(ds))
        elif isinstance(ds, InterleavedDataset):
            s += get_interleaved_lengths(ds)
        else:
            s += "*" + str(len(ds))
        s += ", "
    s += "]"
    return s
