import logging
import re
from collections.abc import Sequence
from contextlib import nullcontext
from types import NoneType
from typing import Any, Callable, Literal, Optional, Union

from datasets.distributed import split_dataset_by_node
from transformers import PreTrainedTokenizerBase

from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset

from ..distributed import get_rank, get_world_size, main_process_first
from .fast_hf_loader import SimpleArrowIterableDataset
from .iterable_with_length import (
    IterableDatasetWithLength,
    to_iterable_dataset_with_length,
)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def normalize_range(
    length, select_range: range | int | float | str | Sequence | NoneType
) -> range:
    """
    Convert various input types to a range
    Args:
        length: The length of the dataset.
        select_range: The range to normalize. Can be:
            - None: No range, use the full dataset.
            - int: Use the first 'n' records.
            - float: Use the first 'n' percent of records.
            - str: Slice notation (e.g., "100:", ":1000", "100:1000", "10%:", ":80%", "10%:80%")
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
    normalize_range(1000, "100:") -> range(100, 1000)
    normalize_range(1000, ":500") -> range(0, 500)
    normalize_range(1000, "100:500") -> range(100, 500)
    normalize_range(1000, "10%:") -> range(100, 1000)
    normalize_range(1000, ":80%") -> range(0, 800)
    normalize_range(1000, "10%:80%") -> range(100, 800)
    ```
    The range values will be constrained [0, length)
    ```
    normalize_range(1000, (-10, 2.0)) -> range(0, 1000)
    ```
    """

    def normalize_value(value):
        if isinstance(value, str):
            # Handle percentage strings
            if value.endswith("%"):
                percent = float(value[:-1]) / 100.0
                value = int(percent * length)
            else:
                value = int(value)
        elif isinstance(value, float):
            value = int(value * length)
        elif isinstance(value, int):
            if value < 0:
                value = length + value
        else:
            raise ValueError(
                f"Unsupported data-type for dataset range value: {type(value)}"
            )
        value = max(value, 0)
        value = min(value, length)
        return value

    if select_range is None or isinstance(select_range, range):
        return select_range
    elif isinstance(select_range, str):
        # Parse slice notation string (e.g., "100:", ":1000", "100:1000", "10%:", ":80%")
        match = re.match(r"^([^:]*)(?::([^:]*))?$", select_range)
        if not match:
            raise ValueError(f"Invalid slice notation string: {select_range}")

        start_str = match.group(1)
        end_str = match.group(2)

        # Parse start index
        if start_str:
            start_value = normalize_value(start_str)
        else:
            start_value = 0

        # Parse end index
        if end_str is not None:  # Colon was present
            if end_str:  # Non-empty end
                end_value = normalize_value(end_str)
            else:  # Empty end (e.g., "100:")
                end_value = length
        else:  # No colon present - treat as single value for first N
            # This handles cases like "100" -> first 100 elements
            return range(normalize_value(start_str))

        return range(start_value, end_value)
    elif isinstance(select_range, float) or isinstance(select_range, int):
        return range(normalize_value(select_range))
    elif isinstance(select_range, Sequence):
        return range(*tuple(normalize_value(value) for value in select_range))
    else:
        raise ValueError(
            f"Unsupported data-type for dataset range: {type(select_range)}"
        )


def default_tokenize_map_fn(
    batch: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    feature: str,
    add_eos: bool = False,
    **kwargs,
) -> dict[str, Any]:
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
    if add_eos:
        examples = [s + tokenizer.eos_token for s in batch[feature]]
    else:
        examples = batch[feature]

    outputs = tokenizer(
        examples,
        **kwargs,
    )
    return {"input_ids": outputs["input_ids"]}


def preprocess_dataset(
    dataset: HFDataset | HFIterableDataset | IterableDatasetWithLength,
    tokenizer: PreTrainedTokenizerBase,
    *,
    select_range: range | int | float | str | Sequence | NoneType = None,
    to_iterable: bool = False,
    feature: str = "text",
    shuffle: bool = False,
    num_shards: int = 256,
    desc: str = "Tokenizing Dataset",
    seed: int = 42,
    shuffle_buffer_size: int = 10000,
    map_fn: Callable = default_tokenize_map_fn,
    map_kwargs: Optional[dict[str, Any]] = None,
    fn_kwargs: Optional[dict[str, Any]] = None,
    dataset_type: Optional[Literal["map"] | Literal["iterable"]] = None,
    dataset_length: Optional[int] = None,
    remove_columns: bool = True,
    shard_dataset: Optional[Union[bool, dict[str, int]]] = None,
):
    """
    This is a fairly generic and flexible dataset preprocessor to quickly get a dataset
    up and running for evaluation. For production use, write a custom preprocessor!

    Args:
        dataset: The dataset to preprocess.
        tokenizer: The tokenizer to use for tokenization.
        select_range: Range of records to select from the dataset.
            Can be int, float, str (slice notation like "10%:80%"), sequence, or range.
        to_iterable: If True, convert the dataset to an iterable dataset.
        feature: The feature in the dataset to tokenize (default is 'text').
        shuffle: If True, shuffle the dataset before processing.
        num_shards: Number of shards, when converting map -> iterable dataset.
        desc: Description for the progress bar.
        seed: Random seed for shuffling.
        shuffle_buffer_size: Buffer size for shuffling in iterable datasets.
        map_fn: Function to apply for tokenization.
        map_kwargs: Additional keyword arguments for the map function.
        fn_kwargs: Additional keyword arguments for the map function.
        parallel_tokenizer: If True, enable parallel tokenization.
        dataset_type: Explicitly specify dataset type
        dataset_length: Set dataset length, when no __len__ is available.
        shard_dataset: Shard the dataset for distributed training.
            num_shards: The number of shards to split the dataset into
            index: The shard index to use

            If bool and True, num_shards defaults to WORLD_SIZE and index to RANK
    Returns:
        The tokenized dataset.
    """

    assert (
        dataset_type is None or dataset_type == "map" or dataset_type == "iterable"
    ), "dataset_type must be one of None, 'map', or 'iterable'"

    if shard_dataset is not None:
        # Set defaults for bool shard_dataset
        if isinstance(shard_dataset, bool):
            if shard_dataset:
                shard_dataset = dict(
                    num_shards=get_world_size(),
                    index=get_rank(),
                )
            else:
                shard_dataset = None

    # This ensures that the dataset is preprocessed by rank0 and cached before other
    # ranks join in. In the context of Huggingface datasets, the result is that the
    # preprocessed dataset will be cached by rank0 and the cached dataset will be loaded
    # by the other ranks, which avoid potential race conditions and duplicate work.
    #
    # If "shard_dataset" is set, each rank is expected to have its own shard, in which case
    # we don't want to use main_process_first()
    with main_process_first() if shard_dataset is None else nullcontext():
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
                fn_kwargs=fn_kwargs,
            )
            | map_kwargs
        )
        if remove_columns:
            map_kwargs["remove_columns"] = dataset.column_names

        if select_range is not None:
            select_range = normalize_range(len(dataset), select_range)
            assert hasattr(
                dataset, "select"
            ), "This dataset does not appear to support the 'select' API"
            dataset = dataset.select(select_range)

        if shard_dataset is not None:
            world_size = shard_dataset["num_shards"]
            rank = shard_dataset["index"]
            logger.debug(f"Sharding dataset: num_shards={world_size}, index={rank}")
            if isinstance(dataset, HFDataset | HFIterableDataset):
                dataset = split_dataset_by_node(
                    dataset,
                    world_size=world_size,
                    rank=rank,
                )
            else:
                assert hasattr(
                    dataset, "shard"
                ), f"Dataset of type {type(dataset)} does not have shard method."

                if not isinstance(dataset, SimpleArrowIterableDataset):
                    logger.warning(
                        f"Attempting to shard unknown dataset of type '{type(dataset)}' API may not be compatible..."
                    )

                # Use example-level sharding for SimpleArrowIterableDataset
                # File-level sharding doesn't work correctly with virtual splits (from select())
                shard_kwargs = {"num_shards": world_size, "index": rank}
                if isinstance(dataset, SimpleArrowIterableDataset):
                    shard_kwargs["mode"] = "example"

                dataset = dataset.shard(**shard_kwargs)

        # Map-style dataset?
        if (dataset_type and dataset_type == "map") or isinstance(dataset, HFDataset):
            assert (
                hasattr(dataset, "__getitem__")
                and hasattr(dataset, "__len__")
                and hasattr(dataset, "map")
                and hasattr(dataset, "shuffle")
            )
            if to_iterable:
                dataset = to_iterable_dataset_with_length(
                    dataset, num_shards=num_shards
                )
                if shuffle:
                    dataset = dataset.shuffle(
                        buffer_size=shuffle_buffer_size, seed=seed
                    )
            else:
                map_kwargs["desc"] = desc
                if shuffle:
                    dataset = dataset.shuffle(seed=seed)
        else:
            assert (
                hasattr(dataset, "__iter__")
                and hasattr(dataset, "map")
                and hasattr(dataset, "shuffle")
            )
            if not hasattr(dataset, "__len__") and dataset_length:
                dataset = IterableDatasetWithLength(dataset, dataset_length)
            if shuffle:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

        tokenized_data = dataset.map(
            map_fn,
            **map_kwargs,
        )
        return tokenized_data
