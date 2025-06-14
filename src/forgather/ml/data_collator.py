from typing import Dict, List, Any, Optional
from loguru import logger
import torch


class DataCollatorForCausalLM:
    """
    Transform input features through tokenizer.pad(features, **pad_kwargs) and
        generate labels from input_ids for causal language model.

    This is very similar to transformers.DataCollatorForLanguageModeling, when 'mlm' is set to False, but
    provides more control over padding.

    The padding functionality is like transformers.DataCollatorWithPadding, but this lacks the DataCollatorForLanguageModeling's
    generation of labels.

    Finally, it addressed an annoyance of DataCollatorWithPadding, which is that there is no way to further truncate the
    inputs. While truncation can be controlled at the tokenization phase, changing the value there will cause a cache-miss
    for the tokenized dataset, resutling in the dataset be tokenized again. Depending on the dataset, this can be a lengthy
    process; very frustrating when you are just trying to find an optimal maximum sequence length to train with.

    ignore_index: Replace pad_id with this value, which should match the 'ignore_index' passed
        to torch.nn.CrossEntropyLoss()

    truncation: If true, the sequence dimension of the resulting tensors are sliced to max_length or
        tokenizer.model_max_length, if unspecified. This is useful for quick experiments, where you
        don't want to re-tokenize the dataset each time.
        NOTE: This is not very efficient. Use the tokenizer for truncation, where possible.

    pad_kwargs: Arguments to forward to tokenizer's 'pad' method.

    Summary of 'pad' args from https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3191
        padding: str|bool -- Padding strategy, defaults to True
            True | "longest" : Pad to the longest sequence in the batch
            "max_length" : Pad to "max_length"
            False | "do_not_pad" : Do not add padding
        max_length: int -- Maximum length of output
        pad_to_multiple_of: int -- Can be useful for performance tweaking on some GPUs
        padding_side: str -- Which side to pad, "left" or "right"
        return_attention_mask: bool
        return_tensors: str -- Tensor type to return, 'pt', 'np', 'tf'
        verbose: bool

    """

    def __init__(
        self,
        tokenizer,
        truncation: bool = False,
        ignore_index: int = -100,
        **pad_kwargs,
    ):
        self.tokenizer = tokenizer
        self.max_length = pad_kwargs.get("max_length", tokenizer.model_max_length)
        if self.max_length > tokenizer.model_max_length:
            logger.warning(
                f"{max_length=} is greater than {tokenizer.model_max_length=}"
            )
        self.truncation = truncation
        self.ignore_index = ignore_index
        self.pad_kwargs = pad_kwargs

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.truncation:
            features = self._truncate(features)
        batch = self._pad(features)
        input_ids = batch["input_ids"]
        batch["labels"] = torch.where(
            input_ids == self.tokenizer.pad_token_id, self.ignore_index, input_ids
        )
        return batch

    def _truncate(self, features):
        if self.tokenizer.truncation_side == "right":
            features = [
                {key: value[: self.max_length] for key, value in example.items()}
                for example in features
            ]
        else:
            features = [
                {
                    key: value[len(value) - self.max_length :]
                    for key, value in example.items()
                }
                for example in features
            ]
        return features

    def _pad(self, features):
        # Based upon pad_without_fast_tokenizer_warning() for disabling fast-tokenizer pad warning.
        # https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/data/data_collator.py#L53C5-L53C39
        if not hasattr(self.tokenizer, "deprecation_warnings"):
            return self.tokenizer.pad(features, **self.pad_kwargs)
        warning_state = self.tokenizer.deprecation_warnings.get(
            "Asking-to-pad-a-fast-tokenizer", False
        )
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        try:
            padded = self.tokenizer.pad(features, **self.pad_kwargs)
        finally:
            self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = (
                warning_state
            )

        return padded
