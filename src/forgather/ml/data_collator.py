from typing import Dict, List, Any, Tuple, Optional
import copy
import logging

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_pos_ids_for_packed_sequence(x, token_id, eos: bool = True):
    """
    Get position-ids for packed sequence

    Based on: https://huggingface.co/blog/sirluk/llm-sequence-packing
    """
    B, T = x.shape
    eos_idx = (x.view(-1) == token_id).nonzero(as_tuple=True)[0] + eos
    eos_idx_expanded = torch.cat([eos_idx, torch.arange(0,B*T+1,T)]).unique().sort()[0]
    normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T
    normalized_idx = torch.where(normalized_idx == 0, T, normalized_idx)
    reps = normalized_idx[1:] - normalized_idx[:-1]
    reps = torch.where(reps < 1, normalized_idx[1:], reps)
    
    # get position ids for packed sequence
    pos_ids = (torch.arange(B*T) - torch.repeat_interleave(eos_idx_expanded[:-1], reps)).view(B,T)
    return pos_ids

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
        input_name: str = "input_ids",
        labels_name: str | None = "labels",
        packed_sequences: bool = False,
        **pad_kwargs,
    ) -> dict[str, Tensor] | Tuple[dict[str, Tensor], Tensor]:
        """
        Initializes the data collator with tokenizer and padding/truncation options.
        Args:
            tokenizer: The tokenizer instance used for encoding the data.
            truncation (bool, optional): Whether to truncate sequences to the maximum length. Defaults to False.
            ignore_index (int, optional): The index to ignore in labels during loss computation. Defaults to -100.
            input_name_map (Dict[str, str], optional): Remap dictionary for batch labels
            labels_name: The dictionary key for labels, if None, then returned as second element of tuple
            **pad_kwargs: Additional keyword arguments for padding, such as 'max_length' and 'padding'.
        Notes:
            - If 'max_length' is provided in pad_kwargs and padding is not set to 'max_length', 'max_length' will be ignored.
            - A warning is logged if the specified max_length exceeds the tokenizer's model_max_length.
        """
        # We may need to modify the tokenizer...
        tokenizer = copy.deepcopy(tokenizer)
        if tokenizer.pad_token is None:
            logger.warning("No PAD token defined. Setting pad token to EOS")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if tokenizer.padding_side == "left":
            logger.warning("Padding side set to left; moving it to the right")
            tokenizer.padding_side = "right"

        if tokenizer.truncation_side == "left":
            logger.warning("Truncation side set to left; moving it to the right")
            tokenizer.truncation_side = "right"

        self.tokenizer = tokenizer
        self.max_length = pad_kwargs.get("max_length", tokenizer.model_max_length)
        self.input_name = input_name
        self.labels_name = labels_name
        self.packed_sequences = packed_sequences
        # Supress warning about max_length being ignored when padding is not
        # 'max_length' and max_length is present.
        padding = pad_kwargs.get("padding", None)
        pad_to_max_length = (
            padding and isinstance(padding, str) and padding == "max_length"
        )
        if not pad_to_max_length and "max_length" in pad_kwargs:
            pad_kwargs.pop("max_length")

        if self.max_length > tokenizer.model_max_length:
            raise ValueError(
                f"{self.max_length=} is greater than {tokenizer.model_max_length=}"
            )
        self.truncation = truncation
        self.ignore_index = ignore_index
        self.pad_kwargs = pad_kwargs

    def __repr__(self):
        return (
            f"{type(self).__name__}(tokenizer={self.tokenizer}, truncation={self.truncation}, "
            f"ignore_index={self.ignore_index }), pad_kwargs={self.pad_kwargs}"
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.truncation:
            features = self._truncate(features)
        padded_batch = self._pad(features)
        input_ids: Tensor = padded_batch["input_ids"]
        labels: Tensor|None = padded_batch.get("labels", None)
        if labels is None:
            labels = torch.where(
                input_ids == self.tokenizer.pad_token_id, self.ignore_index, input_ids
            )
        output_dict = {self.input_name: input_ids}
        if not self.labels_name:
            return output_dict, labels

        output_dict["labels"] = labels
        if self.packed_sequences:
            output_dict["position_ids"] = get_pos_ids_for_packed_sequence(input_ids, self.tokenizer.eos_token_id)
        return output_dict

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
        
        
