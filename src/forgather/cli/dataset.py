import os

from forgather.config import ConfigEnvironment
from forgather.latent import Latent
from forgather.ml.datasets import plot_token_length_histogram
from forgather import Project


def dataset_cmd(args):
    config_name = args.config_template
    project_args = dict(
        project_dir=args.project_dir,
        tokenizer_path=args.tokenizer_path,
    )

    if args.config_template is None:
        args.config_template = ""
    if args.chat_template:
        project_args["chat_template"] = args.chat_template

    proj = Project(args.config_template, **project_args)
    proj_meta = proj("meta")
    config_class = proj_meta["config_class"]
    main_feature = proj_meta["main_feature"]
    assert (
        config_class == "type.dataset"
    ), f"Expected class type.dataset, found {config_class}"

    if args.pp:
        print("Preprocessed configuration:")
        print(proj.pp_config)

    split = proj(args.target)

    if args.tokenizer_path:
        tokenizer = proj("tokenizer")
        print("Tokenizer:")
        print(tokenizer)

    if args.histogram:
        assert args.tokenizer_path, "Tokenizer must be provided to plot histogram"
        args.project_dir
        args.config_template
        cfg_name, _ = os.path.splitext(os.path.basename(args.config_template))
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
        print(f"Printing {args.examples} examples from the train dataset:")
        if args.tokenizer_path:
            for i, example in zip(range(args.examples), split):
                print("-" * 40)
                print(tokenizer.decode(example["input_ids"]))

        else:
            print("Tokenizer path not provided, skipping tokenization.")
            for i, example in zip(range(args.examples), split):
                print("-" * 40)
                print(example[main_feature])
