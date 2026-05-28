import logging
import os
import re
from collections.abc import Sequence
from contextlib import nullcontext
from types import NoneType
from typing import Any, Callable, Literal, Optional, Tuple, Union

from datasets.distributed import split_dataset_by_node
from transformers import PreTrainedTokenizerBase

from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset

from ..distributed import get_rank, get_world_size, main_process_first
from .composable_iterable_dataset import ComposableIterableDataset
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
    Parameters
    ----------
    length : int
        The length of the dataset.
    select_range : range, int, float, str, Sequence, or None
        The range to normalize. Can be:

        - None: No range, use the full dataset.
        - int: Use the first 'n' records.
        - float: Use the first 'n' percent of records.
        - str: Slice notation (e.g., "100:", ":1000", "100:1000", "10%:", ":80%", "10%:80%")
        - Sequence: A sequence of two values, interpreted as (start, end).
        - range: A range object to use directly.

    Returns
    -------
    range
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
    Parameters
    ----------
    batch : dict of str
        The dataset batch to tokenize.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to use for tokenization.
    feature : str
        The feature in the batch to tokenize.
    add_eos : bool, optional
        Whether to append the EOS token to each example.
    **kwargs
        Additional keyword arguments for the tokenizer.

    Returns
    -------
    dict
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


_PARTITION_METHODS = ("conventional", "work_units")
_PARTITION_PURPOSES = ("train", "eval")


def _resolve_partition_method(
    shard_dataset: Optional[Union[bool, dict]],
    has_diloco: bool,
    partition_purpose: str = "train",
) -> Tuple[Optional[str], Optional[dict]]:
    """Normalize ``shard_dataset`` into a (method, kwargs) pair.

    Accepted shapes:

    - ``None`` / ``False``                            → no partitioning.
    - ``True``                                        → conventional shard via WORLD_SIZE/RANK.
    - ``{"num_shards": N, "index": I}``               → legacy explicit conventional shard.
    - ``{"method": "conventional", ...}``             → conventional shard. Extra
      ``num_shards`` / ``index`` overrides WORLD_SIZE / RANK.
    - ``{"method": "work_units"}``                    → DiLoCo work-unit dispatch.

    Validity matrix (raises ``ValueError`` on the error cells):

    - ``method='work_units'`` requires ``DILOCO_SERVER`` set (otherwise
      there's no server to dispatch from). Also requires
      ``partition_purpose='train'`` — eval / test datasets are
      replicated across DiLoCo workers (every host runs the full eval)
      and don't go through the work-queue.
    - ``method='conventional'`` combined with ``DILOCO_SERVER`` set
      raises **only when ``partition_purpose='train'``** — for the
      train dataset, conventional sharding causes asymmetric-DDP row
      overlap (DDPx4 + DDPx8 hosts produce overlapping shard offsets
      and train on the same rows). Eval / test are replicated across
      hosts intentionally and conventional sharding is the right way
      to split eval work across the DDP ranks within each host (the
      cross-host duplication is harmless — metrics get averaged).

    Parameters
    ----------
    partition_purpose : {"train", "eval"}
        Distinguishes the train dataset's strict DiLoCo rules from the
        eval / test datasets' replicated-across-hosts behavior. The
        ``load_dataset.yaml`` template stamps this per-singleton; other
        callers default to ``"train"`` for safety.

    Returns
    -------
    (method, kwargs)
        ``method`` is one of ``"conventional"``, ``"work_units"``, or
        ``None`` (no partitioning). ``kwargs`` is the
        ``{num_shards, index}`` dict for ``"conventional"`` and
        ``None`` for the other two cases.
    """
    if partition_purpose not in _PARTITION_PURPOSES:
        raise ValueError(
            f"Unknown partition_purpose: {partition_purpose!r}. "
            f"Must be one of {_PARTITION_PURPOSES}."
        )

    if shard_dataset is None or shard_dataset is False:
        return None, None

    if shard_dataset is True:
        method = "conventional"
        kwargs = {"num_shards": get_world_size(), "index": get_rank()}
    elif isinstance(shard_dataset, dict):
        method = shard_dataset.get("method", "conventional")
        if method not in _PARTITION_METHODS:
            raise ValueError(
                f"Unknown shard_dataset.method: {method!r}. "
                f"Must be one of {_PARTITION_METHODS} or omitted "
                "(defaults to 'conventional')."
            )
        if method == "conventional":
            kwargs = {
                "num_shards": shard_dataset.get("num_shards", get_world_size()),
                "index": shard_dataset.get("index", get_rank()),
            }
        else:  # work_units
            kwargs = None
    else:
        raise TypeError(
            f"shard_dataset must be bool, dict, or None; got {type(shard_dataset).__name__}"
        )

    if method == "work_units" and not has_diloco:
        raise ValueError(
            "shard_dataset.method='work_units' requires DILOCO_SERVER "
            "to be set in the environment — there's no server to "
            "dispatch work units from. Either start a DiLoCo server "
            "and set DILOCO_SERVER, or use "
            "shard_dataset.method='conventional' (or omit method) for "
            "standard DDP."
        )
    if method == "work_units" and partition_purpose != "train":
        raise ValueError(
            f"shard_dataset.method='work_units' is only valid for the "
            f"train dataset (got partition_purpose={partition_purpose!r}). "
            "Eval / test datasets are replicated across DiLoCo workers "
            "by design — every host runs the full eval and metrics are "
            "averaged across hosts. Use shard_dataset: True / False "
            "for eval depending on whether you want within-host DDP "
            "sharding."
        )
    if method == "conventional" and has_diloco and partition_purpose == "train":
        raise ValueError(
            "shard_dataset.method='conventional' (or bool=True / "
            "{num_shards, index}) for the TRAIN dataset under DiLoCo "
            "(DILOCO_SERVER is set) causes asymmetric-DDP row overlap: "
            "hosts with different WORLD_SIZE produce overlapping "
            "per-rank shard offsets and train on the same rows. Use "
            "shard_dataset.method='work_units' so all DDP ranks across "
            "all DiLoCo hosts compete for units in one shared queue."
        )
    return method, kwargs


def preprocess_dataset(
    dataset: HFDataset | HFIterableDataset | IterableDatasetWithLength,
    tokenizer: PreTrainedTokenizerBase,
    *,
    select_range: range | int | float | str | Sequence | NoneType = None,
    to_iterable: bool = False,
    feature: str = "text",
    shuffle: bool = False,
    num_shards: int = 256,
    desc: str = "Dataset",
    seed: int = 42,
    shuffle_buffer_size: int = 1000,
    map_fn: Callable = default_tokenize_map_fn,
    map_kwargs: Optional[dict[str, Any]] = None,
    fn_kwargs: Optional[dict[str, Any]] = None,
    dataset_type: Optional[Literal["map"] | Literal["iterable"]] = None,
    dataset_length: Optional[int] = None,
    remove_columns: bool = True,
    shard_dataset: Optional[Union[bool, dict[str, int]]] = None,
    partition_purpose: str = "train",
):
    """
    This is a fairly generic and flexible dataset preprocessor to quickly get a dataset
    up and running for evaluation. For production use, write a custom preprocessor!

    Parameters
    ----------
    dataset : HFDataset, HFIterableDataset, or IterableDatasetWithLength
        The dataset to preprocess.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to use for tokenization.
    select_range : range, int, float, str, Sequence, or None, optional
        Range of records to select from the dataset.
        Can be int, float, str (slice notation like "10%:80%"), sequence, or range.
    to_iterable : bool, optional
        If True, convert the dataset to an iterable dataset.
    feature : str, optional
        The feature in the dataset to tokenize (default is 'text').
    shuffle : bool, optional
        If True, shuffle the dataset before processing.
    num_shards : int, optional
        Number of shards, when converting map -> iterable dataset.
    desc : str, optional
        Description for the progress bar.
    seed : int, optional
        Random seed for shuffling.
    shuffle_buffer_size : int, optional
        Buffer size for shuffling in iterable datasets.
    map_fn : callable, optional
        Function to apply for tokenization.
    map_kwargs : dict or None, optional
        Additional keyword arguments for the map function.
    fn_kwargs : dict or None, optional
        Additional keyword arguments for the map function.
    dataset_type : "map", "iterable", or None, optional
        Explicitly specify dataset type.
    dataset_length : int or None, optional
        Set dataset length, when no __len__ is available.
    shard_dataset : bool, dict, or None, optional
        Configure how the dataset is partitioned across DDP ranks.

        - ``None`` / ``False``: no partitioning (single-host or
          dispatch_batches mode).
        - ``True``: conventional sharding via WORLD_SIZE / RANK.
        - ``{"num_shards": N, "index": I}``: explicit conventional
          shard (legacy form; equivalent to
          ``{"method": "conventional", "num_shards": N, "index": I}``).
        - ``{"method": "conventional", ...}``: conventional shard. Can
          override ``num_shards`` / ``index``.
        - ``{"method": "work_units"}``: DiLoCo work-unit dispatch.
          Requires ``DILOCO_SERVER`` to be set; all DDP ranks across
          all DiLoCo hosts compete for units in one shared queue.

        Combining ``conventional`` with DiLoCo is rejected at preprocess
        time **for the train dataset** (``partition_purpose='train'``)
        — different host topologies (e.g. DDPx4 vs DDPx8) produce
        overlapping shard offsets. For eval / test
        (``partition_purpose='eval'``), conventional sharding under
        DiLoCo is allowed: every host runs the full eval (DDP-sharded
        within host), metrics are averaged across hosts, and the
        cross-host duplication is harmless. Combining ``work_units``
        without DiLoCo is always rejected (no server to dispatch from).
    partition_purpose : {"train", "eval"}, optional
        Distinguishes the train dataset's strict DiLoCo rules from the
        eval / test datasets' replicated-across-hosts behavior.
        ``load_dataset.yaml`` stamps this per-singleton; other callers
        default to ``"train"`` for safety. Determines which validity
        rules apply to ``shard_dataset``.

    Returns
    -------
    dataset
        The tokenized dataset.
    """

    assert (
        dataset_type is None or dataset_type == "map" or dataset_type == "iterable"
    ), "dataset_type must be one of None, 'map', or 'iterable'"

    # Resolve the partition method up-front so misconfiguration (e.g.
    # work_units without DILOCO_SERVER) fails at preprocess time
    # rather than mid-training when the dispatch loop fires.
    from ..diloco import diloco_is_enabled

    has_diloco = diloco_is_enabled()
    partition_method, partition_kwargs = _resolve_partition_method(
        shard_dataset,
        has_diloco=has_diloco,
        partition_purpose=partition_purpose,
    )

    # This ensures that the dataset is preprocessed by rank0 and cached before other
    # ranks join in. In the context of Huggingface datasets, the result is that the
    # preprocessed dataset will be cached by rank0 and the cached dataset will be loaded
    # by the other ranks, which avoid potential race conditions and duplicate work.
    #
    # If we're partitioning per rank (conventional shard or work-unit
    # dispatch), each rank is expected to handle its own slice — don't
    # take the main_process_first() lock.
    use_main_first = partition_method is None
    with main_process_first() if use_main_first else nullcontext():
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

        if partition_method == "conventional":
            world_size = partition_kwargs["num_shards"]
            rank = partition_kwargs["index"]

            # As an optimization, skip sharding output would be the same as the input
            if world_size > 1:
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

                    if not isinstance(dataset, ComposableIterableDataset):
                        logger.warning(
                            f"Attempting to shard unknown dataset of type '{type(dataset)}' API may not be compatible..."
                        )
                    dataset = dataset.shard(
                        num_shards=world_size,
                        index=rank,
                    )
        elif partition_method == "work_units":
            # DiLoCo work-unit dispatch — server-driven row partitioning
            # that subsumes conventional sharding. The wrap operates on
            # the composable's post-slice view bounds (and refuses to
            # compose with an already-applied shard).
            if not isinstance(dataset, ComposableIterableDataset):
                raise TypeError(
                    "shard_dataset.method='work_units' is only "
                    "supported on ComposableIterableDataset (the "
                    "dispatch loop reads slice bounds and load_args "
                    "off the wrapper). Got "
                    f"{type(dataset).__name__} — load via "
                    "fast_load_iterable_dataset."
                )
            from .work_unit_dispatch import maybe_enable_work_dispatch

            dataset = maybe_enable_work_dispatch(dataset)

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
                map_kwargs["desc"] = "Tokenizing " + desc
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
