from collections.abc import Sequence
from types import NoneType
from typing import Any, Callable, Literal, Optional

from datasets.distributed import split_dataset_by_node
from transformers import PreTrainedTokenizerBase

from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset

from ..distributed import DistributedEnvInterface, main_local_process_first
from .iterable_with_length import (
    IterableDatasetWithLength,
    to_iterable_dataset_with_length,
)


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


# This ensures that the dataset is preprocessed by rank0 and cached before other
# ranks join in. In the context of Huggingace datasets, the result is that the
# preprocessed dataset will be cached by rank0 and the cached dataset will be loaded
# by the other ranks, which avoid potential race conditions and duplicate work.
@main_local_process_first()
def preprocess_dataset(
    dataset: HFDataset | HFIterableDataset | IterableDatasetWithLength,
    tokenizer: PreTrainedTokenizerBase,
    *,
    select_range: range | int | float | Sequence | NoneType = None,
    to_iterable: bool = False,
    feature: str = "text",
    shuffle: bool = False,
    num_shards: int = 256,
    distributed_environment: Optional[DistributedEnvInterface] = None,
    desc: str = "Tokenizing Dataset",
    seed: int = 42,
    shuffle_buffer_size: int = 10000,
    map_fn: Callable = default_tokenize_map_fn,
    map_kwargs: Optional[dict[str, Any]] = None,
    fn_kwargs: Optional[dict[str, Any]] = None,
    dataset_type: Optional[Literal["map"] | Literal["iterable"]] = None,
    dataset_length: Optional[int] = None,
    remove_columns: bool = True,
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
        dataset_type: Explicitly specify dataset type
        dataset_length: Set dataset length, when no __len__ is available.
    Returns:
        The tokenized dataset.
    """

    assert (
        dataset_type is None or dataset_type == "map" or dataset_type == "iterable"
    ), "dataset_type must be one of None, 'map', or 'iterable'"

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
        dataset = dataset.select(select_range)

    # Map-style dataset?
    if (dataset_type and dataset_type == "map") or isinstance(dataset, HFDataset):
        assert (
            hasattr(dataset, "__getitem__")
            and hasattr(dataset, "__len__")
            and hasattr(dataset, "map")
            and hasattr(dataset, "shuffle")
        )
        if to_iterable:
            dataset = to_iterable_dataset_with_length(dataset, num_shards=num_shards)
            if shuffle:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
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
