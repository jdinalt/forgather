#!/usr/bin/env python3
import os
import argparse
from argparse import RawTextHelpFormatter
import logging
import matplotlib.pyplot as plt
import numpy as np

from forgather.config import ConfigEnvironment
from forgather.latent import Latent

forgather_root = "../../../"

config_template = """
-- set ns = namespace()
-- set ns.forgather_dir = forgather_root

tokenizer: &tokenizer !singleton:transformers:AutoTokenizer.from_pretrained@tokenizer
    - "{{ tokenizer_path }}"

-- include "dataset_config"
"""

dataset_config_template = """
-- extends "datasets/samantha.yaml"

-- block datasets_meta_config
    == super()
    -- set datasets_ns.chat_template = chat_template_path
-- endblock datasets_meta_config
"""

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Test Samantha dataset preprocessing",
    )
    parser.add_argument(
        "tokenizer_path",
        type=str,
        help="Path to tokenizer to test",
    )

    parser.add_argument(
        "-t",
        "--template-path",
        type=str,
        default="",
        help="Path to chat template",
    )

    parser.add_argument(
        "-n",
        "--examples",
        type=int,
        default=3,
        help="Number of examples to print",
    )
    
    args = parser.parse_args(args)
    return args

def dataset_histogram(dataset, tokenizer, sample_size=1000, min=None, max=None):
    import torch
    lengths =  torch.tensor([ len(sample) for sample in dataset["input_ids"] ])

    # Supress debug junk
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    print("Dataset Stats")
    print(f"sample size: {len(lengths)}")
    print(f"min: {lengths.min()}")
    print(f"max: {lengths.max()}")
    print(f"mean: {lengths.float().mean()}")
    print(f"median: {lengths.float().median()}")
    print(f"std: {lengths.float().std()}")
    counts, bins = np.histogram(lengths.numpy(), bins=100, density=True)
    fig, axs = plt.subplots(1, 1, figsize=(20, 5))
    axs.stairs(counts, bins)
    plt.savefig("histogram.svg", format="svg")

def main():
    args = parse_args()
    
    template_searchpath = [
        os.path.join(forgather_root, "templatelib/base"),
        os.path.join(forgather_root, "templatelib/examples"),
    ]
    
    env = ConfigEnvironment(searchpath=template_searchpath)
    template_loader = env.get_loader()
    template_loader.add_template("config", config_template)
    template_loader.add_template("dataset_config", dataset_config_template)
    
    pp_config = env.preprocess(
        "config",
        forgather_root=forgather_root,
        tokenizer_path=args.tokenizer_path,
        chat_template_path=args.template_path,
    )

    print(pp_config)
    
    config = env.load_from_ppstring(pp_config).config
    tokenizer = Latent.materialize(config["tokenizer"])
    print(tokenizer)
    
    train_dataset = Latent.materialize(config["train_dataset"])
    print(train_dataset)
    
    eval_dataset = Latent.materialize(config["eval_dataset"])
    print(eval_dataset)
    
    for example in train_dataset.select(range(args.examples)):
        print('-'*40)
        print(tokenizer.decode(example["input_ids"]))

    dataset_histogram(train_dataset, tokenizer)

if __name__ == "__main__":
    main()