import os
import datasets
from .distributed import main_process_first


# select: the ratio of the total to tokenize. e.g. 0.1 = 10%
@main_process_first()
def tokenize_dataset(dataset, tokenizer, select=1.0, shuffle=False):
    def map_fn(element, tokenizer):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            # Other common arguments, for reference.
            # padding=True,
            # return_tensors='pt',
            # This can be used to limit the block-size to less than what the model can handle.
            # max_length=block_size,
            # If set to True, if the sequence is truncated, the 'overflowing' tokens will be
            # returned on the next call.
            # return_overflowing_tokens=True,
            # This can we used to get the length of the returned sequences, allowing one to
            # discard short sequences, if desired.
            # return_length=True,
        )
        return {"input_ids": outputs["input_ids"]}

    if shuffle:
        dataset = dataset.shuffle()

    if select < 1.0:
        dataset = dataset.select(range(0, int(len(dataset) * select)))

    tokenized_data = dataset.map(
        map_fn,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs=dict(tokenizer=tokenizer),
    )
    return tokenized_data
