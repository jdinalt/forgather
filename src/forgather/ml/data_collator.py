import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _pos_ids_from_boundaries(x: Tensor, document_starts: Tensor) -> Tensor:
    """
    Generate position IDs from explicit document boundary positions.

    Args:
        x: Input tensor (B, T) containing token IDs
        document_starts: Tensor (B, max_docs) of document start positions.
                        Padded with -1 for sequences with fewer documents.

    Returns:
        Position IDs tensor (B, T) where position resets at each document boundary.
    """
    B, T = x.shape
    device = x.device

    # Initialize position IDs with sequential values
    pos_ids = torch.zeros((B, T), dtype=torch.long, device=device)

    for batch_idx in range(B):
        # Get valid document starts for this sequence (filter out padding -1s)
        starts = document_starts[batch_idx]
        starts = starts[starts >= 0]  # Remove padding

        if len(starts) == 0:
            # No document boundaries, use sequential positions
            pos_ids[batch_idx] = torch.arange(T, device=device)
        else:
            # Generate position IDs that reset at each boundary
            current_pos = 0
            for i, start in enumerate(starts):
                start_clamped = min(int(start), T)
                # Calculate end position (next boundary or end of sequence)
                end = min(int(starts[i + 1]), T) if i + 1 < len(starts) else T

                # Fill positions from start to end
                doc_length = end - start_clamped
                if doc_length > 0:
                    pos_ids[batch_idx, start_clamped:end] = torch.arange(doc_length, device=device)

    return pos_ids


def _pos_ids_from_tokens(x: Tensor, token_id: int, eos: bool = True) -> Tensor:
    """
    Generate position IDs by detecting boundary tokens (legacy method).

    Args:
        x: Input tensor (B, T) containing token IDs
        token_id: The token ID to use as boundary marker
        eos: If True, reset after token; if False, reset before token

    Returns:
        Position IDs tensor (B, T) where position resets at each boundary token.

    Based on: https://huggingface.co/blog/sirluk/llm-sequence-packing
    """
    B, T = x.shape
    eos_idx = (x.view(-1) == token_id).nonzero(as_tuple=True)[0] + eos
    eos_idx_expanded = (
        torch.cat([eos_idx, torch.arange(0, B * T + 1, T)]).unique().sort()[0]
    )
    normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T
    normalized_idx = torch.where(normalized_idx == 0, T, normalized_idx)
    reps = normalized_idx[1:] - normalized_idx[:-1]
    reps = torch.where(reps < 1, normalized_idx[1:], reps)

    # get position ids for packed sequence
    pos_ids = (
        torch.arange(B * T) - torch.repeat_interleave(eos_idx_expanded[:-1], reps)
    ).view(B, T)
    return pos_ids


def get_pos_ids_for_packed_sequence(
    x: Tensor,
    token_id: Optional[int] = None,
    document_starts: Optional[Tensor] = None,
    eos: bool = True,
) -> Tensor:
    """
    Get position-ids for packed sequence.

    Supports two modes:
    1. Explicit boundaries (preferred): Uses document_starts to determine where to reset positions
    2. Token-based detection (legacy): Uses special tokens (e.g., EOS) to infer boundaries

    Args:
        x: Input tensor (B, T) containing token IDs
        token_id: Token ID to use as boundary marker (for legacy token-based mode)
        document_starts: Tensor of document start positions (for explicit boundary mode)
        eos: If True, reset after token; if False, reset before token (token-based mode only)

    Returns:
        Position IDs tensor (B, T) where position resets at document boundaries.

    Raises:
        ValueError: If neither document_starts nor token_id is provided.
    """
    if document_starts is not None:
        # Preferred: Use explicit boundary information
        return _pos_ids_from_boundaries(x, document_starts)
    elif token_id is not None:
        # Legacy: Infer boundaries from special tokens
        return _pos_ids_from_tokens(x, token_id, eos)
    else:
        raise ValueError(
            "Must provide either document_starts (explicit boundaries) or "
            "token_id (token-based detection)"
        )


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
        packed_sequences: Optional[bool] = None,
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
            packed_sequences (bool | None, optional): Enable position ID generation for packed sequences.
                If None (default), automatically infers from presence of 'document_starts' field in features.
                If True, always generates position IDs. If False, never generates position IDs.
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
            f"ignore_index={self.ignore_index}, pad_kwargs={self.pad_kwargs})"
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # print(f"{type(features)=}")
        # print(f"{len(features)=}")
        # print(f"{type(features[0])=}")
        # for k in features[0].keys():
        #    print(f"{k=}")
        if self.truncation:
            features = self._truncate(features)

        # Extract document_starts before padding (tokenizer.pad won't handle this custom field)
        document_starts_list = None
        if features and "document_starts" in features[0]:
            document_starts_list = [f.pop("document_starts") for f in features]

        padded_batch = self._pad(features)
        input_ids: Tensor = padded_batch["input_ids"]
        labels: Tensor | None = padded_batch.get("labels", None)
        if labels is None:
            labels = torch.where(
                input_ids == self.tokenizer.pad_token_id, self.ignore_index, input_ids
            )
        output_dict = {self.input_name: input_ids}
        if not self.labels_name:
            return output_dict, labels

        output_dict["labels"] = labels

        # Auto-detect packed sequences from presence of document_starts
        # Precedence: explicit setting > auto-detect from document_starts
        use_packed_sequences = self.packed_sequences
        if use_packed_sequences is None:
            # Auto-detect: use packed sequences if document_starts is present
            use_packed_sequences = document_starts_list is not None

        if use_packed_sequences:
            # Use explicit document boundaries if available, otherwise fall back to token-based
            if document_starts_list is not None:
                # Pad document_starts to uniform shape
                document_starts_tensor = self._pad_document_starts(document_starts_list)
                output_dict["position_ids"] = get_pos_ids_for_packed_sequence(
                    input_ids, document_starts=document_starts_tensor
                )
            else:
                # Legacy: Use token-based boundary detection
                output_dict["position_ids"] = get_pos_ids_for_packed_sequence(
                    input_ids, token_id=self.tokenizer.eos_token_id
                )
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

    def _pad_document_starts(self, document_starts_list: List[List[int]]) -> Tensor:
        """
        Pad document_starts lists to uniform shape for batching.

        Args:
            document_starts_list: List of document start lists, one per sequence.
                                 Each inner list contains positions where documents start.

        Returns:
            Tensor (B, max_docs) where each row contains document start positions,
            padded with -1 for sequences with fewer documents.
        """
        if not document_starts_list:
            return torch.tensor([], dtype=torch.long)

        # Find maximum number of documents in any sequence
        max_docs = max(len(starts) for starts in document_starts_list)

        # Pad all sequences to have the same number of document starts
        # Use -1 as padding value to indicate "no document here"
        padded = []
        for starts in document_starts_list:
            padded_starts = starts + [-1] * (max_docs - len(starts))
            padded.append(padded_starts)

        return torch.tensor(padded, dtype=torch.long)

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
