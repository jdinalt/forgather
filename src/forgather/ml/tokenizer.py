from typing import Dict, List
import time
import os
from collections.abc import Sequence

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm
from datasets import Dataset

from .datasets import normalize_range


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
        dataset: Dataset,
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
        self.train_dataset = dataset
        self.args = args
        self.output_dir = output_dir
        self.feature = feature
        select_range
        if select_range is not None:
            select_range = normalize_range(len(self.train_dataset), select_range)
            print(f"Selecting range {select_range} from dataset")
            self.train_dataset = dataset.select(select_range)

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
