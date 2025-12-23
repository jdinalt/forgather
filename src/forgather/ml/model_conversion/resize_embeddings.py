import logging
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Default token configuration for handling missing PAD tokens
DEFAULT_TOKEN_CONFIG = {
    "pad_token": {
        "token": "[PAD]",
        "init": "zero",
        "if_missing": True,
    }
}


def add_tokens_to_tokenizer(
    tokenizer: PreTrainedTokenizerBase, path_or_config: str | Dict
) -> Tuple[int, Dict[int, str]]:
    """Load additional tokens from YAML and add them to tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer to add tokens to
        path_or_config: Path to YAML file or config dict containing tokens to add

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
            if_missing: true             # Only add if not already set
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

    if_missing flag:
        - When true, only add/set the token if it doesn't already exist
        - Useful for ensuring a token exists without forcing replacement
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
                if_missing = False
            elif isinstance(token_entry, dict):
                # Dict format: extract token, init strategy, and if_missing flag
                token_value = token_entry.get("token")
                if token_value is None:
                    logger.warning(f"Skipping {token_name}: missing 'token' field")
                    continue
                init_strategy = token_entry.get("init", DEFAULT_INIT[token_name])
                if_missing = token_entry.get("if_missing", False)
            else:
                logger.warning(f"Skipping {token_name}: invalid format")
                continue

            old_token = getattr(tokenizer, token_name, None)

            # Check if_missing flag
            if if_missing and old_token is not None:
                logger.info(
                    f"Skipping {token_name}: already set to {old_token} (if_missing=true)"
                )
                continue

            # Handle token reassignment when old token exists but differs from new value
            if old_token is not None and old_token != token_value:
                # Get the old token ID before reassignment
                old_token_id = getattr(tokenizer, f"{token_name}_id", None)

                # Check if the new token already exists in vocabulary
                vocab = tokenizer.get_vocab()
                if token_value in vocab:
                    # Token already exists, just reassign the special token pointer
                    new_token_id = vocab[token_value]
                    logger.info(
                        f"Reassigning {token_name}: {old_token} (ID {old_token_id}) -> {token_value} (ID {new_token_id}, already exists)"
                    )
                    named_tokens[token_name] = token_value
                    # No new token added, but we still need to apply init strategy to existing token
                    token_inits[new_token_id] = init_strategy
                else:
                    # New token doesn't exist in vocabulary
                    # We need to add it as a new token
                    # To preserve the semantics, we'll copy the embedding from the old token
                    logger.info(
                        f"Replacing {token_name}: {old_token} (ID {old_token_id}) -> {token_value} (will be added as new token)"
                    )
                    logger.info(
                        f"New token embedding will be copied from old token at ID {old_token_id}"
                    )

                    # Add the new token
                    named_tokens[token_name] = token_value
                    named_token_inits[token_name] = (
                        f"copy:{old_token_id}"  # Special init strategy
                    )
            elif old_token is None:
                logger.info(
                    f"Setting {token_name}: {token_value} (init: {init_strategy})"
                )
                named_tokens[token_name] = token_value
                named_token_inits[token_name] = init_strategy
            else:
                # Token already set to the same value
                logger.info(f"Token {token_name} already set to {token_value}")

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

    # Add additional special tokens (check if they already exist first)
    special_tokens = token_config.get("special_tokens", [])
    if special_tokens:
        # Get existing additional_special_tokens to avoid duplicates
        existing_special_tokens = set(
            getattr(tokenizer, "additional_special_tokens", [])
        )

        # Filter out tokens that already exist
        new_special_tokens = [
            token for token in special_tokens if token not in existing_special_tokens
        ]

        if new_special_tokens:
            num_special = tokenizer.add_special_tokens(
                {"additional_special_tokens": new_special_tokens}
            )
            logger.info(
                f"Added {num_special} additional special token(s): {new_special_tokens}"
            )
            num_added += num_special
        else:
            logger.info(
                f"Skipping additional special tokens: all {len(special_tokens)} token(s) already exist"
            )

    # Add regular tokens (check if they already exist first)
    regular_tokens = token_config.get("regular_tokens", [])
    if regular_tokens:
        # Get existing vocab to avoid duplicates
        existing_vocab = tokenizer.get_vocab()

        # Filter out tokens that already exist
        new_regular_tokens = [
            token for token in regular_tokens if token not in existing_vocab
        ]

        if new_regular_tokens:
            num_regular = tokenizer.add_tokens(new_regular_tokens)
            logger.info(f"Added {num_regular} regular token(s): {new_regular_tokens}")
            num_added += num_regular
        else:
            logger.info(
                f"Skipping regular tokens: all {len(regular_tokens)} token(s) already exist"
            )

    return num_added, token_inits


def resize_word_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    token_inits: Optional[Dict[int, str]],
):
    """Resize input and output embeddings to match tokenizer vocab size, with optional custom init"""
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
                if init_strategy == "zero":
                    logger.info(f"Zero-initializing token at index {token_id}")
                    input_embeddings[token_id].zero_()
                    if not tied_embeddings:
                        output_embeddings[token_id].zero_()
                elif init_strategy == "mean":
                    # Mean initialization is handled by resize_token_embeddings with mean_resizing=True
                    pass
                elif init_strategy.startswith("copy:"):
                    # Copy embedding from another token
                    source_id = int(init_strategy.split(":", 1)[1])
                    logger.info(
                        f"Copying embedding from token ID {source_id} to {token_id}"
                    )
                    input_embeddings[token_id].copy_(input_embeddings[source_id])
                    if not tied_embeddings:
                        output_embeddings[token_id].copy_(output_embeddings[source_id])
                else:
                    logger.warning(f"Init strategy {init_strategy} is not supported.")


def update_config_from_tokenizer(
    model_config: PretrainedConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    update_vocab_size: bool = False,
):
    """Update model_config tokenizer meta-data from tokenizer

    Args:
        model_config: Model configuration to update
        tokenizer: Tokenizer to read metadata from
        update_vocab_size: If True, update vocab_size to match tokenizer (only use when embeddings were resized)
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

    # Update vocab size only if explicitly requested (i.e., when embeddings were actually resized)
    # This prevents corrupting the config when there's a pre-existing mismatch between
    # model vocab_size and tokenizer vocab size
    if update_vocab_size:
        old_vocab_size = model_config.vocab_size
        new_vocab_size = len(tokenizer)

        if old_vocab_size != new_vocab_size:
            model_config.vocab_size = new_vocab_size
            logger.info(
                f"Updated config vocab size from {old_vocab_size} to {new_vocab_size}"
            )
