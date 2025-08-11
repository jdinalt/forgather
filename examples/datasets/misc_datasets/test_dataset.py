#!/usr/bin/env python3
import os
import argparse
from argparse import RawTextHelpFormatter

from forgather.config import ConfigEnvironment
from forgather.latent import Latent
from forgather.ml.datasets import plot_token_length_histogram

from forgather import Project

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Test Samantha dataset preprocessing",
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer to test",
    )

    parser.add_argument(
        "--histogram-path",
        type=str,
        default=None,
        help="Path to save the histogram plot (.svg), if provided",
    )

    parser.add_argument(
        "-n",
        "--examples",
        type=int,
        default=None,
        help="Number of examples to print",
    )

    parser.add_argument(
        "-t",
        "--config-template",
        type=str,
        default=None,
        help="Config template to use for the project",
    )

    parser.add_argument(
        "--sample-eval",
        action="store_true",
        help="If set, sample from the eval dataset instead of the train dataset",
    )
    
    args = parser.parse_args(args)
    return args

def main():
    args = parse_args()
    config_name = args.config_template
    if args.config_template:
        # Ensure config template is just the name, not a path
        args.config_template = os.path.basename(args.config_template)
    else:
        # Use default config template
        args.config_template = "" 
    proj = Project(args.config_template, tokenizer_path=args.tokenizer_path)
    
    print("Preprocessed configuration:")
    print(proj.pp_config)
    
    train_dataset_split = proj("train_dataset_split")
    print("Train dataset:")
    print(train_dataset_split)
    
    if args.histogram_path:
        assert args.tokenizer_path, "Tokenizer path must be provided to plot histogram"
        tokenizer = proj("tokenizer")
        print("Tokenizer:")
        print(tokenizer)

        print("Plotting token length histogram...")
        plot_token_length_histogram(
            train_dataset_split,
            tokenizer,
            sample_size=1000,
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

            eval_dataset = proj("train_dataset")
            print("Eval dataset:")
            print(eval_dataset)

            dataset = eval_dataset if args.sample_eval else train_dataset
            for i, example in enumerate(dataset):
                print('-'*40)
                print(tokenizer.decode(example["input_ids"]))
                if i >= args.examples - 1:
                    break

        else:
            print("Tokenizer path not provided, skipping tokenization.")
            assert not args.sample_eval, "Cannot sample eval dataset without tokenizer"
            for example in train_dataset_split.shuffle().select(range(args.examples)):
                print('-'*40)
                print(example['text'])

if __name__ == "__main__":
    main()