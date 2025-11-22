from typing import (
    Dict,
    Tuple,
    List,
    Optional,
)
import logging
import yaml
import torch

from transformers import PreTrainedTokenizerBase, PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)

def add_tokens_to_tokenizer(
    tokenizer: PreTrainedTokenizerBase, path_or_config: str | Dict
) -> Tuple[int, Dict[int, str]]:
    """Load additional tokens from YAML and add them to tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer to add tokens to
        add_tokens_path: Path to YAML file containing tokens to add

    Returns:
        Tuple of (num_added, token_inits) where:
        - num_added: Total number of tokens added
        - token_inits: Dict mapping token IDs to initialization strategy ("zero", "mean")

    YAML format (new format with named tokens and init strategies):
        bos_token: "<|begin_of_text|>"  # String format, uses default init (mean)
        eos_token:                       # Dict format with init strategy
            token: "<|end_of_text|>"
            init: "mean"
        pad_token:
            token: "<|pad|>"
            init: "zero"
        unk_token: "<|unknown|>"
        special_tokens:
            - "<|im_start|>"
            - "<|im_end|>"
        regular_tokens:
            - "custom_token"

    Old format (still supported):
        special_tokens:
            - "<|im_start|>"
        regular_tokens:
            - "custom_token"

    Initialization strategies:
        - "zero": Initialize embeddings to zero
        - "mean": Initialize to mean of existing token embeddings
        - Default for BOS/EOS/UNK: "mean"
        - Default for PAD: "zero"
    """
    if isinstance(path_or_config, str):
        with open(path_or_config, "r") as f:
            token_config = yaml.safe_load(f)
    else:
        assert isinstance(path_or_config, dict)
        token_config = path_or_config

    # Define set of named special tokens and their default init strategies
    NAMED_SPECIAL_TOKENS = {"bos_token", "eos_token", "pad_token", "unk_token"}
    DEFAULT_INIT = {
        "bos_token": "mean",
        "eos_token": "mean",
        "pad_token": "zero",
        "unk_token": "mean",
    }

    num_added = 0
    token_inits = {}  # Maps token ID to init strategy

    # Extract named special tokens (bos, eos, pad, unk) with init strategies
    named_tokens = {}
    named_token_inits = {}  # Maps token name to init strategy

    for token_name in NAMED_SPECIAL_TOKENS:
        if token_name in token_config:
            token_entry = token_config[token_name]

            # Support both string and dict format
            if isinstance(token_entry, str):
                # Simple string format: use token string and default init
                token_value = token_entry
                init_strategy = DEFAULT_INIT[token_name]
            elif isinstance(token_entry, dict):
                # Dict format: extract token and init strategy
                token_value = token_entry.get("token")
                if token_value is None:
                    logger.warning(f"Skipping {token_name}: missing 'token' field")
                    continue
                init_strategy = token_entry.get("init", DEFAULT_INIT[token_name])
            else:
                logger.warning(f"Skipping {token_name}: invalid format")
                continue

            old_token = getattr(tokenizer, token_name, None)

            # Log if replacing existing token
            if old_token is not None:
                logger.info(
                    f"Replacing {token_name}: {old_token} -> {token_value} (init: {init_strategy})"
                )
            else:
                logger.info(
                    f"Setting {token_name}: {token_value} (init: {init_strategy})"
                )

            named_tokens[token_name] = token_value
            named_token_inits[token_name] = init_strategy

    # Add named special tokens
    if named_tokens:
        num_named = tokenizer.add_special_tokens(named_tokens)
        logger.info(f"Added {num_named} named special token(s)")
        num_added += num_named

        # Map token IDs to init strategies
        for token_name, init_strategy in named_token_inits.items():
            token_id = getattr(tokenizer, f"{token_name}_id", None)
            if token_id is not None:
                token_inits[token_id] = init_strategy

    # Add additional special tokens
    special_tokens = token_config.get("special_tokens", [])
    if special_tokens:
        num_special = tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        logger.info(
            f"Added {num_special} additional special token(s): {special_tokens}"
        )
        num_added += num_special

    # Add regular tokens
    regular_tokens = token_config.get("regular_tokens", [])
    if regular_tokens:
        num_regular = tokenizer.add_tokens(regular_tokens)
        logger.info(f"Added {num_regular} regular token(s): {regular_tokens}")
        num_added += num_regular

    # Add PAD token if still missing (only if not set via named_tokens)
    if tokenizer.pad_token is None and "pad_token" not in named_tokens:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        logger.info(
            f"Added default PAD token: [PAD] at index {tokenizer.pad_token_id}"
        )
        num_added += 1
        # Default PAD token always uses zero init
        token_inits[tokenizer.pad_token_id] = "zero"

    return num_added, token_inits

def resize_word_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    token_inits: Optional[Dict[int, str]],
):
    """Resize input and output embeddings to match tokenizer vocab size, with optional custom init
    """
    new_vocab_size = len(tokenizer)
    
    # Use HuggingFace's resize_token_embeddings() method
    model.resize_token_embeddings(new_vocab_size, mean_resizing=True)

    # Apply custom initialization strategies for added tokens
    if token_inits:
        with torch.no_grad():
            input_embeddings = model.get_input_embeddings().weight
            output_embeddings = model.get_output_embeddings().weight

            tied_embeddings = input_embeddings is output_embeddings
            for token_id, init_strategy in token_inits.items():
                match init_strategy:
                    case "zero":
                        logger.info(
                            f"Zero-initializing token at index {token_id}"
                        )
                        input_embeddings[token_id].zero_()
                        if not tied_embeddings:
                            output_embeddings[token_id].zero_()
                    case "mean":
                        pass
                    case _:
                        logger.warning(f"Init strategy {init_strategy} is not supported.")
                

def update_config_from_tokenizer(model_config: PretrainedConfig, tokenizer: PreTrainedTokenizerBase):
    """Update model_config tokenizer meta-data from tokenizer
    """
    special_token_ids = [
        "bos_token_id",
        "pad_token_id",
        "eos_token_id",
    ]

    # Transfer special token ids to model config
    for token_id_name in special_token_ids:
        token_id = getattr(tokenizer, token_id_name, None)
        if token_id is not None:
            logger.info(f"Setting {token_id_name} to {token_id}")
            setattr(model_config, token_id_name, token_id)
    
    # Get current vocab size before resizing (for mean calculation)
    old_vocab_size = model_config.vocab_size
    new_vocab_size = len(tokenizer)

    # Update config to reflect new vocabulary size
    if old_vocab_size != new_vocab_size:
        model_config.vocab_size = new_vocab_size
        logger.info(
            f"Updated config vocab sie from {old_vocab_size} to {new_vocab_size}"
        )