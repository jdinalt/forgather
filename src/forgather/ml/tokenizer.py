import os
import time
from collections.abc import Sequence
from typing import Callable

from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from datasets import Dataset
from forgather.ml.trainer.logging import get_env_type
from tokenizers import Tokenizer


def normalize_range(
    length, select_range: range | int | float | Sequence | None
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


class TokenizerTrainer:
    def __init__(
        self,
        *,
        model,
        normalizer,
        pre_tokenizer,
        post_processor,
        decoder,
        trainer,
        dataset: Callable[[], Dataset],
        feature: str = "text",
        select_range: range | int | float | Sequence | None = None,
        args=None,
        output_dir=None,
    ):

        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pre_tokenizer
        tokenizer.decoder = decoder
        tokenizer.post_processor = post_processor

        self.tokenizer = tokenizer
        self.trainer = trainer
        self.train_dataset = dataset()
        self.args = args
        self.output_dir = output_dir
        self.feature = feature
        
        if select_range is not None:
            select_range = normalize_range(len(self.train_dataset), select_range)
            print(f"Selecting range {select_range} from dataset")
            self.train_dataset = self.train_dataset.select(select_range)

    def __call__(self):
        self.train()
        self.save_model()

    def train(self, batch_size=1000):
        total_samples = len(self.train_dataset)
        steps = total_samples // batch_size
        print("**** Training Tokenizer ****")
        print(f"total_samples: {total_samples}")
        print(f"batch_size: {batch_size}")
        print(f"steps: {steps}")
        print(f"Dataset: {self.train_dataset}")

        if get_env_type() != "file":

            def batch_iterator():
                train_progress_bar = tqdm(
                    total=steps, dynamic_ncols=True, desc="Training Tokenizer"
                )
                try:
                    for i in range(0, total_samples, batch_size):
                        yield self.train_dataset[i : i + batch_size][self.feature]
                        train_progress_bar.update()
                finally:
                    train_progress_bar.close()

        else:

            def batch_iterator():
                for i in range(0, total_samples, batch_size):
                    yield self.train_dataset[i : i + batch_size][self.feature]

        start_time = time.time()

        self.tokenizer.train_from_iterator(
            batch_iterator(), trainer=self.trainer, length=steps
        )

        runtime = round(time.time() - start_time, 2)
        samples_per_second = round(total_samples / runtime, 2)
        print("**** Training Completed ****")
        print(f"runtime: {runtime}")
        print(f"samples_per_second: {samples_per_second}")

    def save_model(self, output_dir: str | os.PathLike = None):
        if output_dir is None:
            output_dir = self.output_dir
        assert output_dir is not None
        tokenizer = self.as_pretrained_tokenizer_fast()
        tokenizer.save_pretrained(output_dir)

    def as_pretrained_tokenizer_fast(self, **kwargs):
        if len(kwargs) == 0:
            kwargs = self.args
        assert kwargs is not None
        return PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            **kwargs,
        )


def train_tokenizer(**kwargs):
    trainer = TokenizerTrainer(**kwargs)
    trainer.train()
    trainer.save_model()
    # Release the resources
    del trainer
