# Local Dataset

Template project for using local datasets

This project takes a path as an argument and loads a dataset via [datasets.load_from_disk](https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_from_disk)

It's primarily intended as an example of how to do this, but provided that the dataset has "train" and "validation" splits and the main feature is "text," it should work without further modification.

## Basic CLI Usage

```bash
# Show the first 3 example from train dataset
forgather dataset --target validation_dataset_split --dataset-path /path/to/my_local_dataset -n 3

# Generate sequence-length histogram
forgather dataset --target train_dataset_split --dataset-path /path/to/my_local_dataset \
-T /path/to/tokenizer --histogram

# Tokenize the first 3 examples from "train" with specified tokenizer and decode
forgather dataset --target train_dataset --dataset-path /path/to/my_local_dataset \
-T /path/to/tokenizer -n 3
```

## Importing as a Sub-project

You can import the dataset into another project, like this:

```yaml
[datasets_definition]
.define: &dataset_dict !call:forgather:from_project
    project_dir: "{{ joinpath(ns.forgather_dir, "examples/datasets/local_dataset") }}"
    config_template: "local_dataset.yaml"
    targets: [ "train_dataset", "eval_dataset" ] 
    preprocess_args: *tokenizer_args
    tokenizer: *tokenizer
    dataset-path: "/path/to/my_local_dataset"

    [train_dataset]
train_dataset: &train_dataset !call:getitem [ *dataset_dict, 'train_dataset' ]

    [eval_dataset]
eval_dataset: &eval_dataset !call:getitem [ *dataset_dict, 'eval_dataset' ]
```

## Inline in Project
You could also just copy and inline the code into a project, like this:

```yaml
[datasets_definition]
.define: &dataset_dict !singleton:datasets:load_from_disk
    arg0: "{{ dataset_path | default("/path/to/dataset") }}"

    [train_dataset]
train_dataset: &train_dataset !singleton:forgather.ml.datasets:preprocess_dataset@train_dataset
    dataset: !call:getitem [ *dataset_dict, "train" ]
    tokenizer: *tokenizer
    desc: "Tokenizing train"
    fn_kwargs: !var "preprocess_args"
    to_iterable: True

    [eval_dataset]
eval_dataset: &eval_dataset !singleton:forgather.ml.datasets:preprocess_dataset@eval_dataset
    dataset: !call:getitem [ *dataset_dict, "validation" ]
    tokenizer: *tokenizer
    desc: "Tokenizing validation"
    fn_kwargs: !var "preprocess_args"
```

## Sliding Window

There is an alternative configuration, which demonstrates how to use the sliding-window map-function with a dataset.
This is useful to datasets with examples longer than your context length. It dynamically breaks the long examples into multiple blocks, of "window-size," and overlaps the the block with "stride" tokens.

This can of course be used with datasets downloaded from the HF Hub too.

```bash
# Break the examples down into blocks of 512 tokens, overlapping each block by 64 tokens. Show first 3 blocks.
forgather -t sliding_window.yaml dataset --target train_dataset --dataset-path /path/to/my_local_dataset \
-T /path/to/tokenizer --window-size 512 --stride 64 -n 3
```

## Other Load Methods

The configs have been written for datasets saved with ds.save_to_disk(), but it should be easy enough to change the load method.

For example, if you have a directory with parquet files to load, you can do this:

```yaml
[train_dataset_split]
train_dataset_split: &train_dataset_split !singleton:datasets:load_dataset
    arg0: "parquet"
    data_dir: "path/to/data/dir"
```

See: https://huggingface.co/docs/datasets/en/package_reference/loading_methods