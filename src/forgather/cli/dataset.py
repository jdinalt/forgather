import itertools
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


def _detect_tokenized(example) -> bool:
    """Heuristic: a dataset is tokenized if examples carry ``input_ids``
    whose elements are ints. Plain-text datasets expose string-valued
    features (``text``, ``content``, …) and lack ``input_ids`` entirely.

    The first element is enough to disambiguate — text fields don't
    survive HF's column dtype unification as a list of ints, and
    tokenized fields don't survive as strings. Empty ``input_ids`` is
    rare but treated as tokenized (the field exists).
    """
    if not isinstance(example, dict):
        return False
    if "input_ids" not in example:
        return False
    ids = example["input_ids"]
    try:
        if len(ids) == 0:
            return True
        first = ids[0]
    except TypeError:
        return False
    # bool is a subclass of int; explicitly reject it. torch tensors return
    # 0-d tensors when indexed with [0] for 1-d, so accept those via .item().
    if isinstance(first, bool):
        return False
    if isinstance(first, int):
        return True
    if torch.is_tensor(first):
        try:
            return first.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            )
        except AttributeError:
            return False
    return False


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
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )
        data += "Tokenizer:\n" + repr(tokenizer) + "\n"
        template_args["tokenizer"] = tokenizer

    if args.seed is not None:
        template_args["shuffle"] = True
        template_args["seed"] = args.seed

    split = proj(args.target, **template_args)

    # Resolve tokenized/raw once up-front so both histogram and examples
    # branches see a consistent answer. The deprecated --tokenized flag
    # forces the choice; otherwise we peek the first example. The peeked
    # example is chained back onto ``split`` so downstream iteration
    # doesn't lose it (matters for IterableDatasets, which have no
    # cheap reset).
    if args.tokenized:
        tokenized = True
    else:
        iterator = iter(split)
        peeked = None
        try:
            peeked = next(iterator)
        except StopIteration:
            pass
        if peeked is None:
            tokenized = False
        else:
            tokenized = _detect_tokenized(peeked)
            print(
                f"Auto-detected dataset format: "
                f"{'tokenized' if tokenized else 'raw text'}"
            )
            split = itertools.chain([peeked], iterator)

    if args.histogram:
        assert args.tokenizer_path, "Tokenizer must be provided to plot histogram"
        args.project_dir
        args.config_template
        cfg_name, _ = os.path.splitext(os.path.basename(proj.config_name))
        cfg_name += ".svg"
        histogram_path = os.path.join(os.path.realpath(args.project_dir), cfg_name)
        print(f"Generating token-length histogram: {histogram_path}")
        if not tokenized:
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

        if tokenized:
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
