import os
import datasets
from .distributed import main_process_first


@main_process_first()
def tokenize_dataset(
    dataset,
    tokenizer,
    select: int | float = 1.0,
    shuffle=False,
    **kwargs,
):
    """
    Tokenize a dataset

    tokenizer: The tokenizer to use
    select: If a float, the fraction of the dataset to use. If int, the number of examples
    shuffle: Shuffle the dataset first
    kwargs: Passed to tokenizer __call__

    returns: tokenized dataset
    """

    def map_fn(element):
        outputs = tokenizer(
            element["text"],
            **kwargs,
        )
        return {"input_ids": outputs["input_ids"]}

    if shuffle:
        dataset = dataset.shuffle()

    if isinstance(select, float):
        if select < 1.0:
            dataset = dataset.select(range(0, int(len(dataset) * select)))
    elif isinstance(select, int):
        dataset = dataset.select(range(0, select))

    print("*** Tokenizing Dataset ***")
    tokenized_data = dataset.map(
        map_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized_data
