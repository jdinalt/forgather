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

    try:
        train_dataset_split = proj("train_dataset_split")
        print("Train dataset:")
        print(train_dataset_split)
    except KeyError:
        train_dataset_split = None
        print("No train dataset found.")

    try:
        eval_dataset_split = proj("eval_dataset_split")
        print("Eval dataset:")
        print(eval_dataset_split)
    except KeyError:
        eval_dataset_split = None
        print("No eval dataset found.")

    if args.tokenizer_path:
        tokenizer = proj("tokenizer")
        print("Tokenizer:")
        print(tokenizer)

    if args.histogram_path:
        assert args.tokenizer_path, "Tokenizer path must be provided to plot histogram"
        print("Plotting token length histogram...")
        if args.use_split:
            if args.sample_eval:
                assert (
                    eval_dataset_split
                ), "Eval dataset split must be exist to sample from eval dataset"
                split = eval_dataset_split
            else:
                assert (
                    train_dataset_split
                ), "Train dataset split must exist to sample from train dataset"
                split = train_dataset_split
            plot_token_length_histogram(
                split,
                tokenizer=tokenizer,
                sample_size=args.histogram_samples,
                feature=main_feature,
                min=None,
                max=None,
                output_file=args.histogram_path,
            )
        else:
            if args.sample_eval:
                dataset = proj("eval_dataset")
            else:
                dataset = proj("train_dataset")
            plot_token_length_histogram(
                dataset,
                tokenizer=None,
                sample_size=args.histogram_samples,
                feature="input_ids",
                min=None,
                max=None,
                output_file=args.histogram_path,
            )

    if args.examples:
        print(f"Printing {args.examples} examples from the train dataset:")
        if args.tokenizer_path:
            tokenizer = proj("tokenizer")
            train_dataset = proj("train_dataset")
            print("Tokenized train dataset:")
            print(train_dataset)

            eval_dataset = proj("eval_dataset")
            print("Eval dataset:")
            print(eval_dataset)

            dataset = eval_dataset if args.sample_eval else train_dataset
            for i, example in enumerate(dataset):
                print("-" * 40)
                print(tokenizer.decode(example["input_ids"]))
                if i >= args.examples - 1:
                    break

        else:
            print("Tokenizer path not provided, skipping tokenization.")
            if args.sample_eval:
                assert (
                    eval_dataset_split
                ), "Eval dataset split must be exist to sample from eval dataset"
                split = eval_dataset_split
            else:
                assert (
                    train_dataset_split
                ), "Train dataset split must exist to sample from train dataset"
                split = train_dataset_split
            for example in split.shuffle().select(range(args.examples)):
                print("-" * 40)
                print(example[main_feature])
