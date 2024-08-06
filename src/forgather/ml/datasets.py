from dataclasses import dataclass, field
from types import NoneType

from datasets import DatasetDict, Dataset
from collections.abc import Sequence

from .distributed import main_process_first


def normalize_range(
    length, select_range: range | int | float | Sequence | NoneType
) -> range:
    """
    Convert various input types to a range

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
            pass
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
    """

    input_split: str
    output_split: str | NoneType = None
    select_range: range | int | float | Sequence | NoneType = None
    feature: str = "text"

    def __post_init__(self):
        if self.output_split is None:
            self.output_split = self.input_split


def default_tokenize_map_fn(element, tokenizer, feature, **kwargs):
    outputs = tokenizer(
        element[feature],
        **kwargs,
    )
    return {"input_ids": outputs["input_ids"]}


@main_process_first()
def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    *,
    select_range: range | int | float | Sequence | NoneType = None,
    feature="text",
    shuffle=False,
    desc="Tokenizing Dataset",
    map_fn=default_tokenize_map_fn,
    map_kwargs={},
    fn_kwargs={},
):
    """
    Tokenize a dataset

    tokenizer: The tokenizer to use
    select_range: See normalize_range()
    feature: The name of the feature to tokenize
    shuffle: Shuffle the dataset first
    desc: Description to show while processing.
    map_fn: The map function to use
    map_kwargs: Additional args passed to dataset.map()
    fn_kwargs: Additional args passed to map_fn.
        The default map function passes these args to tokenizer.__call__

    returns: tokenized dataset
    """

    if select_range is not None:
        select_range = normalize_range(len(dataset), select_range)
        dataset = dataset.select(select_range)

    if shuffle:
        dataset = dataset.shuffle()

    tokenized_data = dataset.map(
        map_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc=desc,
        fn_kwargs=dict(tokenizer=tokenizer, feature=feature) | fn_kwargs,
        **map_kwargs,
    )
    return tokenized_data


@main_process_first()
def tokenize_dataset_dict(
    input_dataset_dict: DatasetDict,
    tokenizer,
    splits: Sequence,
    map_fn=default_tokenize_map_fn,
    map_kwargs={},
    fn_kwargs={},
):
    """
    Tokenize multiple splits in a dataset dictionary

    input_dataset_dict: The source dataset-dict
    tokenizer: See tokenize_dataset()
    splits: a sequence of split-specs. See below.
    map_fn: See tokenize_dataset()
    map_kwargs: See tokenize_dataset()
    fn_kwars: See tokenize_dataset()

    ```
    splits = (
        {
            "input_split": "train",
        },
        {
            "input_split": "validation",
            "select_range": 500,
        },
    )
    output_dataset_dict = tokenize_dataset_dict(input_dataset_dict, tokenizer, splits, fn_kwargs=dict(truncation=True))
    ```
    """
    output_dataset_dict = DatasetDict()
    for split in splits:
        split_spec = SplitSpec(**split)
        input_dataset = input_dataset_dict[split_spec.input_split]
        output_dataset = tokenize_dataset(
            input_dataset,
            tokenizer,
            select_range=split_spec.select_range,
            feature=split_spec.feature,
            desc=f"Tokenizing Split '{split_spec.input_split}'",
            map_fn=map_fn,
            map_kwargs=map_kwargs,
            fn_kwargs=fn_kwargs,
        )
        output_dataset_dict[split_spec.output_split] = output_dataset
    return output_dataset_dict
