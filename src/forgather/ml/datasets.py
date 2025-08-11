from dataclasses import dataclass, field
from types import NoneType

import torch
from datasets import DatasetDict, Dataset
from torch.utils.data import DataLoader
from datasets.distributed import split_dataset_by_node
from collections.abc import Sequence
import os
import re

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


@dataclass(kw_only=True)
class SplitSpec:
    """
    Holds a split specification, which describes and input to output split mapping.
    This is used to specify how to process a dataset split, including the input and output splits,
    the range of records to select, and the feature to process.
    """

    input_split: str
    output_split: str | NoneType = None
    select_range: range | int | float | Sequence | NoneType = None
    feature: str = "text"

    def __post_init__(self):
        if self.output_split is None:
            self.output_split = self.input_split


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


class InputTokenBlock:
    def __init__(self, input_ids, length):
        self.length = length
        self.input_ids = input_ids
        self.read_index = 0

    def __len__(self):
        return self.length

    def read(self, length):
        length = min(self.length, length)
        ids = self.input_ids[self.read_index : self.read_index + length]
        self.read_index += length
        self.length -= length
        return ids


class OutputTokenBlock:
    def __init__(self, max_length, input_ids=None):
        self.max_length = max_length
        if input_ids is not None:
            self.input_ids = input_ids
            self.length = len(input_ids)
            assert self.length <= max_length
        else:
            self.length = 0
            self.input_ids = []

    def __len__(self):
        return self.length

    def remaining(self):
        return self.max_length - self.length

    def get_ids(self):
        return self.input_ids

    def append(self, input_block):
        length = min((self.max_length - self.length), len(input_block))
        input_ids = input_block.read(length)
        self.input_ids += input_ids
        self.length += length


def block_tokenize_fn(
    element,
    tokenizer,
    feature,
    block_size=32,
    overflow=True,
    stride=0,
    min_len=1,
    max_len=None,
    add_bos=True,
    add_eos=False,
    combine=False,
    truncate_at=None,
):
    """
    Tokenizes the input element into blocks of tokens.

    The typical use case is to tokenize text into blocks of a given size, with options for overflow,
    stride, and special tokens. Useful when examples are too long for the model's maximum input size.

    Args:
        element: The input element to tokenize.
        tokenizer: The tokenizer to use for tokenization.
        feature: The feature in the element to tokenize.
        block_size: The maximum size of each output block.
        overflow: If True, allows overflow of input tokens into multiple outputs.
        stride: Number of tokens to overlap between blocks.
        min_len: Minimum length of the output blocks.
        max_len: Maximum length of the output blocks (optional).
        add_bos: If True, adds a beginning-of-sequence token.
        add_eos: If True, adds an end-of-sequence token.
        combine: If True, combines input into multiple outputs.
        truncate_at: If provided, truncates input at the first match of this regex.
    Returns:
        A dictionary with a single key "input_ids" containing a list of tokenized blocks.
    """
    assert min_len >= 1

    # If given a regex to truncate at, truncate at the first match.
    if truncate_at is not None:
        input_batch = []
        for text in element[feature]:
            match_offset = re.search(truncate_at, text)
            if match_offset is not None:
                text = text[: match_offset.start()]
            input_batch.append(text)
    else:
        input_batch = element[feature]

    outputs = tokenizer(
        input_batch,
        truncation=False,
        return_length=True,
        # Silence warning about exceeding model's max length
        # We are performing the truncation ourselves.
        max_length=9223372036854775807,
    )

    # A list of strings of tokens of maximum size 'block_size'
    output_batch = []

    # A container for accumulating output tokens.
    output_block = OutputTokenBlock(block_size)

    # A container for the input tokens from the current record in the input batch.
    input_block = None

    # Appends the output block to the output_batch
    # - Conditional upon minimum length
    # - Allocates next output block
    # - Transfers 'stride' tokens from end of old block to start of new block.
    def append_output_batch(output_block, output_batch, stride):
        stride_tokens = None
        # If the present output block is empty, just return and keep the current one.
        if not len(output_block):
            return output_block
        # If the output has at least the minimum number of tokens.
        elif len(output_block) >= min_len:
            # Save 'stride' tokens from the end to prefix the next block with
            if add_bos:
                stride_tokens = [tokenizer.bos_token_id]
            else:
                stride_tokens = []

            if stride != 0:
                stride_tokens += output_block.get_ids()[-stride:]

            # Append the block to the list of output blocks
            output_batch.append(output_block.get_ids())
        # else, we discard the output block

        # Allocate a new output block, initialized with 'stride' tokens
        output_block = OutputTokenBlock(block_size, stride_tokens)
        return output_block

    # Get next tokenized input record
    for record_length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        # print(f"record length {record_length}")
        # If we are not allowed to mix inputs in outputs, get a new output.
        if not combine:
            output_block = append_output_batch(output_block, output_batch, 0)

        # If the length of the record is less than the minimum, discard it.
        if record_length < min_len:
            continue

        # If the input record is longer than the maximum, discard it.
        if max_len is not None and record_length > max_len:
            continue

        # If we will be adding the EOS token, add it now.
        if add_eos:
            record_length += 1
            input_ids += [tokenizer.eos_token_id]

        # Encapsulate the input record in an input block
        input_block = InputTokenBlock(input_ids, record_length)

        # While the input block still has data to read...
        while len(input_block):
            # Move as much data from the input block to the output block as will fit.
            # Note: These classes perform bounds checking to prevent overflow/underflow.
            output_block.append(input_block)

            # If we will not being combining the input into multiple outputs.
            if not overflow:
                # Add to outputs and get next input.
                output_block = append_output_batch(output_block, output_batch, 0)
                break

            # If the output block is mostly full, allocate a new output block
            elif output_block.remaining() < min_len:
                # Add to outputs and continue with present input.
                output_block = append_output_batch(output_block, output_batch, stride)

    # Append the last output data.
    append_output_batch(output_block, output_batch, 0)
    return {"input_ids": output_batch}


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
    parallel_tokenizer=True,
):
    """
    Preprocess a dataset by tokenizing it with the provided tokenizer.
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

    os.environ["TOKENIZERS_PARALLELISM"] = "true" if parallel_tokenizer else "false"

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


def test_with_dataloader(
    ds,
    tokenizer,
    collate_fn,
    n=4,
    batch_size=2,
    prefetch_factor=2,
    drop_last=True,
    num_workers=1,
    pin_memory=True,
    **dataloder_kwargs,
):
    """
    Simple test function to test a dataset with a standard dataloader
    Args:
        ds: The dataset to test
        tokenizer: The tokenizer to use
        collate_fn: The collate function to use
        n: Number of batches to print
        batch_size: Batch size for the dataloader
        prefetch_factor: Prefetch factor for the dataloader
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of workers for the dataloader
        pin_memory: Whether to pin memory for the dataloader
        **dataloder_kwargs: Additional arguments for DataLoader
    """
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloder_kwargs,
    )
    for i, batch in enumerate(dl):
        print(f"batch {i}")
        for input_ids in batch["input_ids"]:
            print("---")
            print(repr(tokenizer.decode(input_ids)))
        if i == n:
            break

def plot_token_length_histogram(
        dataset,
        tokenizer,
        output_file=None,
        sample_size=1000,
        feature='text',
        min=None,
        max=None
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

    texts = dataset.shuffle()[:sample_size][feature]
    outputs = tokenizer(
        texts,
        return_length=True,
    )
    
    lengths = torch.tensor(outputs['length'])
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