"""
Finish reason determination for generation completion.
"""

from typing import List, Set, Optional
from transformers import PreTrainedTokenizer


class FinishReasonDetector:
    """
    Determines why generation stopped.

    This utility analyzes generation output to determine the reason for stopping,
    which can be: "length" (max tokens reached) or "stop" (EOS token, stop sequence,
    or early stopping).

    Example:
        Basic usage:
        >>> detector = FinishReasonDetector(tokenizer, stop_token_ids={2, 50256})
        >>> token_ids = [1, 3, 4, 5, 6]
        >>> max_tokens = 5
        >>> stopped_by_sequence = False
        >>>
        >>> reason = detector.determine_finish_reason(
        ...     token_ids, max_tokens, stopped_by_sequence
        ... )
        >>> print(reason)  # "length"

        With stop sequence:
        >>> reason = detector.determine_finish_reason(
        ...     [1, 3, 4], 10, stopped_by_sequence=True
        ... )
        >>> print(reason)  # "stop"

        Streaming usage:
        >>> reason = detector.determine_finish_reason_streaming(
        ...     completion_tokens=50,
        ...     max_tokens=100,
        ...     stop_sequences=["STOP"],
        ...     full_response="Text with STOP"
        ... )
        >>> print(reason)  # "stop"
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, stop_token_ids: Set[int]
    ) -> None:
        """
        Initialize finish reason detector.

        Args:
            tokenizer: HuggingFace tokenizer instance
            stop_token_ids: Set of token IDs that trigger stopping
        """
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.stop_token_ids: Set[int] = stop_token_ids

    def determine_finish_reason(
        self,
        generated_token_ids: List[int],
        max_tokens: int,
        stopped_by_sequence: bool,
        ignore_eos: bool = False,
    ) -> str:
        """
        Determine the finish reason for non-streaming generation.

        Args:
            generated_token_ids: List of generated token IDs
            max_tokens: Maximum tokens allowed
            stopped_by_sequence: Whether generation stopped due to a stop sequence
            ignore_eos: Whether EOS tokens were ignored during generation

        Returns:
            Finish reason string: "length" or "stop"
        """
        if len(generated_token_ids) >= max_tokens:
            return "length"
        elif stopped_by_sequence:
            return "stop"
        elif not ignore_eos:
            # Normal EOS handling - only check EOS if not ignoring it
            if (
                self.tokenizer.eos_token_id is not None
                and len(generated_token_ids) > 0
                and generated_token_ids[-1] == self.tokenizer.eos_token_id
            ):
                return "stop"
            elif (
                len(generated_token_ids) > 0
                and generated_token_ids[-1] in self.stop_token_ids
            ):
                return "stop"
        # If we get here, model stopped for unknown reason
        elif len(generated_token_ids) < max_tokens:
            # Stopped early but not due to obvious reasons
            return "stop"
        else:
            return "stop"

    def determine_finish_reason_streaming(
        self,
        completion_tokens: int,
        max_tokens: int,
        stop_sequences: List[str],
        full_response: str,
        ignore_eos: bool = False,
    ) -> str:
        """
        Determine the finish reason for streaming generation.

        Note: For streaming, EOS token detection in decoded text is less reliable.
        The ignore_eos parameter is included for consistency but has limited impact
        since streaming works with text rather than token IDs.

        Args:
            completion_tokens: Number of tokens generated
            max_tokens: Maximum tokens allowed
            stop_sequences: List of stop sequences
            full_response: Full generated response text
            ignore_eos: Whether EOS tokens were ignored during generation

        Returns:
            Finish reason string: "length" or "stop"
        """
        if completion_tokens >= max_tokens:
            return "length"
        elif any(stop_seq in full_response for stop_seq in stop_sequences):
            return "stop"
        else:
            return "stop"
