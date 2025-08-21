from dataclasses import dataclass, field
from types import NoneType
import logging
from collections.abc import Sequence
import os

import torch
from datasets import Dataset
from datasets.distributed import split_dataset_by_node

from .distributed import main_process_first


def normalize_range(
    length, select_range: range | int | float | Sequence | NoneType
) -> range:
    """
    Convert various input types to a range
    Args:
        length: The length of the dataset.
        select_range: The range to normalize. Can be:
            - None: No range, use the full dataset.
            - int: Use the first 'n' records.
            - float: Use the first 'n' percent of records.
            - Sequence: A sequence of two values, interpreted as (start, end).
            - range: A range object to use directly.
    Returns:
        A range object representing the normalized range.

    Examples:
    ```
    normalize_range(1000, None) -> range(0, 1000)
    normalize_range(1000, 0.25) -> range(0, 250)
    normalize_range(1000, 500) -> range(0, 500)
    normalize_range(1000, [100, 900]) -> range(100, 900)
    normalize_range(1000, [100, 0.9]) -> range(100, 900)
    normalize_range(1000, (1, 1.0, 4)) -> range(1, 1000, 4)
    normalize_range(1000, range(10, 100)) -> range(10, 100)
    ```
    The range values will be constrained [0, length)
    ```
    normalize_range(1000, (-10, 2.0)) -> range(0, 1000)
    ```
    """

    def normalize_value(value):
        if isinstance(value, float):
            value = int(value * length)
        elif isinstance(value, int):
            if value < 0:
                value = length - value
        else:
            raise ValueError(
                f"Unsupported data-type for dataset range value: {type(value)}"
            )
        value = max(value, 0)
        value = min(value, length)
        return value

    if select_range is None or isinstance(select_range, range):
        return select_range
    elif isinstance(select_range, float) or isinstance(select_range, int):
        return range(normalize_value(select_range))
    elif isinstance(select_range, Sequence):
        return range(*tuple(normalize_value(value) for value in select_range))
    else:
        raise ValueError(
            f"Unsupported data-type for dataset range: {type(select_range)}"
        )


def default_tokenize_map_fn(element, tokenizer, feature, **kwargs):
    """
    Default map function for tokenizing a dataset element.
    Args:
        element: The dataset element to tokenize.
        tokenizer: The tokenizer to use for tokenization.
        feature: The feature in the element to tokenize.
        **kwargs: Additional keyword arguments for the tokenizer.
    Returns:
        A dictionary with a single key "input_ids" containing the tokenized input.
    """
    outputs = tokenizer(
        element[feature],
        **kwargs,
    )
    return {"input_ids": outputs["input_ids"]}


# This ensures that the dataset is preprocessed by rank0 and cached before other
# ranks join in. In the context of Huggingace datasets, the result is that the
# preprocessed dataset will be cached by rank0 and the cached dataset will be loaded
# by the other ranks, which avoid potential race conditions and duplicate work.
@main_process_first()
def preprocess_dataset(
    dataset: Dataset,
    tokenizer,
    *,
    select_range: range | int | float | Sequence | NoneType = None,
    to_iterable=False,
    feature="text",
    shuffle=False,
    num_shards=256,
    distributed_environment=None,
    desc="Tokenizing Dataset",
    seed=42,
    shuffle_buffer_size=10000,
    map_fn=default_tokenize_map_fn,
    map_kwargs=None,
    fn_kwargs=None,
):
    """
    This is a farily generic and flxible dataset preprocessor to quickly get a dataset
    up and running for evaluation. For production use, write a custom preprocessor!

    Args:
        dataset: The dataset to preprocess.
        tokenizer: The tokenizer to use for tokenization.
        select_range: Range of records to select from the dataset.
        to_iterable: If True, convert the dataset to an iterable dataset.
        feature: The feature in the dataset to tokenize (default is 'text').
        shuffle: If True, shuffle the dataset before processing.
        num_shards: Number of shards for the iterable dataset.
        distributed_environment: Environment for distributed processing.
        desc: Description for the progress bar.
        seed: Random seed for shuffling.
        shuffle_buffer_size: Buffer size for shuffling in iterable datasets.
        map_fn: Function to apply for tokenization.
        map_kwargs: Additional keyword arguments for the map function.
        fn_kwargs: Additional keyword arguments for the map function.
        parallel_tokenizer: If True, enable parallel tokenization.
    Returns:
        The tokenized dataset.
    """

    # assert hasattr(dataset, __getitem__), "A map-style dataset is required for this function"
    if fn_kwargs is None:
        fn_kwargs = dict()

    fn_kwargs = (
        dict(
            tokenizer=tokenizer,
            feature=feature,
        )
        | fn_kwargs
    )

    if map_kwargs is None:
        map_kwargs = dict()

    map_kwargs = (
        dict(
            batched=True,
            remove_columns=dataset.column_names,
            fn_kwargs=fn_kwargs,
        )
        | map_kwargs
    )

    if select_range is not None:
        select_range = normalize_range(len(dataset), select_range)
        dataset = dataset.select(select_range)

    if to_iterable:
        dataset = dataset.to_iterable_dataset(num_shards=num_shards)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    else:
        map_kwargs["desc"] = desc
        if shuffle:
            dataset = dataset.shuffle(seed=seed)

    if distributed_environment is not None:
        dataset = split_dataset_by_node(
            dataset,
            world_size=distributed_environment.world_size,
            rank=distributed_environment.rank,
        )

    tokenized_data = dataset.map(
        map_fn,
        **map_kwargs,
    )
    return tokenized_data


def plot_token_length_histogram(
    dataset,
    tokenizer,
    output_file=None,
    sample_size=1000,
    feature="text",
    min=None,
    max=None,
):
    """
    Plot a histogram of token lengths in the dataset.
    If output_file is provided, save the histogram to that file.
    Otherwise, display the histogram.
    Args:
        dataset: The dataset to analyze.
        tokenizer: The tokenizer to use for tokenization.
        output_file: Optional; if provided, save the histogram to this file.
        sample_size: Number of samples to use for the histogram.
        feature: The feature in the dataset to analyze (default is 'text').
        min: Minimum length for the histogram (optional).
        max: Maximum length for the histogram (optional).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import islice

    # Suppress matplotlib warnings about missing fonts
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    if tokenizer:
        samples = [sample[feature] for sample in islice(dataset.shuffle(), sample_size)]
        outputs = tokenizer(
            samples,
            return_length=True,
        )
        lengths = torch.tensor(outputs["length"])
    else:
        lengths = torch.tensor(
            [
                len(sample["input_ids"])
                for sample in islice(dataset.shuffle(), sample_size)
            ]
        )
    print(f"sample size: {len(lengths)}")
    print(f"min: {lengths.min()}")
    print(f"max: {lengths.max()}")
    print(f"mean: {lengths.float().mean()}")
    print(f"median: {lengths.float().median()}")
    print(f"std: {lengths.float().std()}")
    counts, bins = np.histogram(lengths.numpy(), bins=100, density=True)
    fig, axs = plt.subplots(1, 1, figsize=(20, 5))
    axs.stairs(counts, bins)

    if output_file:
        plt.savefig(output_file, format="svg")
    else:
        plt.show()
