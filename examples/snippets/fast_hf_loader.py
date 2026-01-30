import os
from functools import partial
import argparse
from argparse import RawTextHelpFormatter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from forgather.ml.datasets import fast_load_iterable_dataset, preprocess_dataset, default_tokenize_map_fn

class Args(argparse.Namespace):
    tokenizer_id_or_path: str

def main(args: Args):
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.tokenizer_id_or_path)

    fineweb_split = fast_load_iterable_dataset(
        path="HuggingFaceTB/smollm-corpus",
        name="fineweb-edu-dedup",
        split="train[10000:]",
    )

    print(fineweb_split)

    fineweb = preprocess_dataset(
        dataset=fineweb_split,
        tokenizer=tokenizer,
        map_fn=partial(default_tokenize_map_fn, add_eos=True),
        map_kwargs=dict(batch_size=32),
        shard_dataset=dict(num_shards=2, index=0),
        #select_range=":10%",
        #shuffle=False,
    )

    print(fineweb)

    # Load and print the first example
    example = next(iter(fineweb))
    print(tokenizer.decode(example["input_ids"]))

def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Test harness for fast_load_iterable_dataset()",
        epilog=(
            "Examples:\n"
            "\n"
            "    %(prog)s ./path/to/tokenizer\n"
        ),
    )
    parser.add_argument(
        "tokenizer_id_or_path", type=os.path.expanduser, help="Path or ID of tokenizer"
    )
    return parser.parse_args(namespace=Args())


if __name__ == "__main__":
    main(parse_args())