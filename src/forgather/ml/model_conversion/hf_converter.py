"""Base HuggingFace model converter implementation."""

import os
import logging
from contextlib import ExitStack
from typing import Optional, Dict, Any, Tuple, Callable
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.modeling_utils import no_init_weights as hf_no_init_weights
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from forgather import Project, MetaConfig
from forgather.ml.remap_params import remap_state_dict
from forgather.ml.sharded_checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)
from forgather.ml.utils import default_dtype
from forgather.ml.no_init_weights import no_init_weights
from .base import ModelConverter

logger = logging.getLogger(__name__)


class HFConverter(ModelConverter):
    """Base class for HuggingFace model converters.

    This class implements generic conversion logic for HuggingFace models.
    Model-specific converters should subclass this and override methods
    as needed to handle model-specific details.
    """

    def __init__(self, model_type: str):
        """Initialize HuggingFace converter.

        Args:
            model_type: String identifier for the model type (e.g., "llama", "mistral")
            model_project_dir: Path to the model project directory (e.g., "examples/models/llama")
        """
        super().__init__(model_type)

    def get_hf_config_class(self):
        """Get HuggingFace config class for this model type.

        Returns:
            HuggingFace config class (e.g., LlamaConfig)
        """
        raise NotImplementedError("Subclasses must implement get_hf_config_class()")

    def get_hf_model_class(self):
        """Get HuggingFace model class for this model type.

        Returns:
            HuggingFace model class (e.g., LlamaForCausalLM)
        """
        raise NotImplementedError("Subclasses must implement get_hf_model_class()")

    def get_project_info(
        self,
    ) -> dict[str, Any]:
        """Get path to project and configuration to use

        Returns:
            dict(
                project_dir=PROJECT_DIR,
                config_name=CONFIG_NAME
            )
        """
        raise NotImplementedError("Subclasses must implement get_project_info()")

    def build_model(
        self, src_model_path, src_model_config, output_dir, *, max_length=None, **kwargs
    ) -> Tuple[PretrainedConfig, PreTrainedTokenizer, Callable[[], PreTrainedModel]]:
        """Build model in output directory and return config, tokenizer, and model ctor

        Args:
            src_model_dir: Path to source model
            src_model_config: HuggingFace model configuration
            max_length: Optional max sequence length override
            output_dir: Path to where to construct model
            kwargs: Additional, optional config args. e.g. max_length

        Returns:
            Tuple of PretrainedConfig, PreTrainedTokenizer, and Callable, returning PreTrainedModel
        """
        project_info = self.get_project_info()

        # Translate src config into config args
        config_args = self.create_project_config(src_model_config, max_length)

        proj = Project(
            config_name=project_info["config_name"],
            project_dir=project_info["project_dir"],
            output_dir=output_dir,
            tokenizer_id_or_path=src_model_path,
            **config_args,
        )

        # Dump config for diagnostics
        logger.debug(proj.pp_config)

        return proj("pretrained_config", "pretrained_tokenizer", "model")

    def create_project_config(
        self, src_config: Any, max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create Forgather Project configuration from HuggingFace config.

        Args:
            src_config: HuggingFace model configuration
            max_length: Optional max sequence length override

        Returns:
            Dictionary of parameters to pass to Project() constructor
        """
        raise NotImplementedError("Subclasses must implement create_project_config()")

    def create_hf_config(
        self, src_config: Any, max_length: Optional[int] = None
    ) -> Any:
        """Create HuggingFace config from Forgather config.

        Args:
            src_config: Forgather model configuration
            max_length: Optional max sequence length override

        Returns:
            HuggingFace model configuration
        """
        config_class = self.get_hf_config_class()
        field_mapping = self.get_config_field_mapping("from_forgather")

        max_model_length = src_config.max_sequence_length
        if max_length:
            max_model_length = max_length

        # Build config kwargs from field mapping
        config_kwargs = {}
        for fg_field, hf_field in field_mapping.items():
            if hasattr(src_config, fg_field):
                config_kwargs[hf_field] = getattr(src_config, fg_field)

        # Add max_position_embeddings
        config_kwargs["max_position_embeddings"] = max_model_length

        # Add common defaults that may not be in field mapping
        if "attention_dropout" not in config_kwargs:
            config_kwargs["attention_dropout"] = getattr(
                src_config, "attention_dropout", 0.0
            )
        if "hidden_act" not in config_kwargs:
            config_kwargs["hidden_act"] = "silu"
        if "tie_word_embeddings" not in config_kwargs:
            config_kwargs["tie_word_embeddings"] = False
        if "pad_token_id" not in config_kwargs:
            config_kwargs["pad_token_id"] = getattr(src_config, "pad_token_id", None)
        if "bos_token_id" not in config_kwargs:
            config_kwargs["bos_token_id"] = getattr(src_config, "bos_token_id", 1)
        if "eos_token_id" not in config_kwargs:
            config_kwargs["eos_token_id"] = getattr(src_config, "eos_token_id", 2)

        return config_class(**config_kwargs)

    def _load_and_add_tokens(
        self, tokenizer: AutoTokenizer, add_tokens_path: str
    ) -> Tuple[bool, int, Dict[int, str]]:
        """Load additional tokens from YAML and add them to tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer to add tokens to
            add_tokens_path: Path to YAML file containing tokens to add

        Returns:
            Tuple of (needs_pad_token, num_added, token_inits) where:
            - needs_pad_token: True if PAD token was added
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
        with open(add_tokens_path, "r") as f:
            token_config = yaml.safe_load(f)

        # Define set of named special tokens and their default init strategies
        NAMED_SPECIAL_TOKENS = {"bos_token", "eos_token", "pad_token", "unk_token"}
        DEFAULT_INIT = {
            "bos_token": "mean",
            "eos_token": "mean",
            "pad_token": "zero",
            "unk_token": "mean",
        }

        needs_pad_token = False
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
                    logger.info(f"Replacing {token_name}: {old_token} -> {token_value} (init: {init_strategy})")
                else:
                    logger.info(f"Setting {token_name}: {token_value} (init: {init_strategy})")

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
            logger.info(f"Added {num_special} additional special token(s): {special_tokens}")
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
            logger.info(f"Added default PAD token: [PAD] at index {tokenizer.pad_token_id}")
            needs_pad_token = True
            num_added += 1
            # Default PAD token always uses zero init
            token_inits[tokenizer.pad_token_id] = "zero"

        return needs_pad_token, num_added, token_inits

    def convert_to_forgather(
        self,
        src_model_path: str,
        dst_model_path: str,
        dtype: Optional[str] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Convert HuggingFace model to Forgather format.

        Args:
            src_model_path: Path to source HF model directory
            dst_model_path: Path to destination Forgather model directory
            dtype: Optional dtype for output model
            max_length: Optional max sequence length override
            **kwargs: Additional options (debug_params, prompt, etc.)
        """
        from forgather.ml.construct import torch_dtype

        src_model_path = os.path.abspath(src_model_path)
        dst_model_path = os.path.abspath(dst_model_path)

        # Load source model config and validate
        src_model_config = AutoConfig.from_pretrained(src_model_path)
        self.validate_source_config(src_model_config, "to_forgather")

        # Auto-detect dtype if not specified
        if dtype is None:
            # Try to get dtype from source model config (prefer 'dtype', fall back to 'torch_dtype')
            if (
                hasattr(src_model_config, "dtype")
                and src_model_config.dtype is not None
            ):
                dtype = str(src_model_config.dtype).replace("torch.", "")
                logger.info(f"Auto-detected dtype from source model: {dtype}")
            elif (
                hasattr(src_model_config, "torch_dtype")
                and src_model_config.torch_dtype is not None
            ):
                dtype = str(src_model_config.torch_dtype).replace("torch.", "")
                logger.info(
                    f"Auto-detected dtype from source model (via deprecated torch_dtype): {dtype}"
                )
            else:
                # Default to bfloat16 if not available
                dtype = "bfloat16"
                logger.info(f"No dtype in source model, defaulting to: {dtype}")
        else:
            logger.info(f"Using specified dtype: {dtype}")

        # Setup dtype
        new_dtype = None
        if dtype:
            new_dtype = torch_dtype(dtype)

        logger.info(f"Converting {self.model_type} model from HuggingFace to Forgather")
        logger.info(f"Source: {src_model_path}")
        logger.info(f"Destination: {dst_model_path}")
        logger.info(f"DType: {new_dtype}")

        # Capture original HF model type for reverse conversion
        hf_model_type = src_model_config.model_type
        logger.info(f"Capturing HF model type: {hf_model_type}")

        # Load source model
        logger.info("Loading source HuggingFace model...")
        with ExitStack() as exit_stack:
            if new_dtype:
                exit_stack.enter_context(default_dtype(new_dtype))
            src_model = AutoModelForCausalLM.from_pretrained(src_model_path)
        logger.debug(src_model)

        # Load tokenizer
        src_tokenizer = AutoTokenizer.from_pretrained(src_model_path)

        if kwargs.get("debug_params"):
            self._print_params(src_model, "Source HuggingFace model")

        # Remap state dict
        logger.info("Remapping model weight names...")
        src_state_dict = src_model.state_dict()
        param_mappings = self.get_parameter_mappings("to_forgather")
        mapped_state_dict = remap_state_dict(src_state_dict, param_mappings)

        # Apply model-specific transformations
        mapped_state_dict = self.transform_state_dict(
            mapped_state_dict, "to_forgather", src_model_config, None
        )

        # Create Forgather project configuration
        logger.info("Creating Forgather model...")

        # Get model components
        model_config, tokenizer, model_ctor = self.build_model(
            src_model_path,
            src_model_config,
            dst_model_path,
            max_length=max_length,
        )

        # Update vocab_size to match source model
        # Some models have different vocab sizes in config vs tokenizer
        if src_model_config.vocab_size != model_config.vocab_size:
            logger.info(
                f"Adjusting vocab_size from {model_config.vocab_size} "
                f"to {src_model_config.vocab_size} to match source model"
            )
            model_config.vocab_size = src_model_config.vocab_size

        # Apply chat template if provided
        if kwargs.get("chat_template_path"):
            with open(kwargs["chat_template_path"], "r") as f:
                chat_template = f.read()
            logger.info(f"Setting tokenizer chat template")
            tokenizer.chat_template = chat_template

        # Store HF model type and dtype in config for reverse conversion
        # This allows auto-detection of the correct converter for FG->HF conversion
        # and provides a hint for the expected dtype when loading the model
        model_config.hf_model_type = hf_model_type
        model_config.dtype = dtype
        logger.info(f"Stored hf_model_type={hf_model_type} in Forgather config")
        logger.info(f"Stored dtype={dtype} in Forgather config")

        # Construct model
        logger.info("Constructing Forgather model...")
        with ExitStack() as exit_stack:
            if new_dtype:
                exit_stack.enter_context(default_dtype(new_dtype))
            exit_stack.enter_context(no_init_weights())
            model = model_ctor()
        logger.debug(model)

        if kwargs.get("debug_params"):
            self._print_params(model, "Destination Forgather model")

        # Load state dict
        logger.info("Loading mapped state dictionary...")
        result = model.load_state_dict(mapped_state_dict, strict=False, assign=True)
        logger.debug(f"load_state_dict() result: {result}")

        # Tie weights if needed (must be done after loading state dict)
        if (
            hasattr(model_config, "tie_word_embeddings")
            and model_config.tie_word_embeddings
        ):
            logger.info("Tying word embeddings...")
            model.tie_weights()

        # Add tokens and resize embeddings if requested
        if kwargs.get("add_tokens"):
            logger.info(f"Adding tokens from: {kwargs['add_tokens']}")
            needs_pad_token, num_added, token_inits = self._load_and_add_tokens(
                tokenizer, kwargs["add_tokens"]
            )

            if num_added > 0:
                logger.info(f"Added {num_added} token(s) to vocabulary")
                new_vocab_size = len(tokenizer)
                logger.info(
                    f"Resizing token embeddings from {model_config.vocab_size} to {new_vocab_size}..."
                )

                # Get current vocab size before resizing (for mean calculation)
                old_vocab_size = model_config.vocab_size

                # Use HuggingFace's resize_token_embeddings() method
                # mean_resizing=False uses model's default initialization (normal distribution)
                model.resize_token_embeddings(new_vocab_size, mean_resizing=False)

                # Apply custom initialization strategies for added tokens
                if token_inits:
                    with torch.no_grad():
                        input_embeddings = model.get_input_embeddings().weight
                        output_embeddings = (
                            model.get_output_embeddings().weight
                            if not model_config.tie_word_embeddings
                            else None
                        )

                        for token_id, init_strategy in token_inits.items():
                            if init_strategy == "zero":
                                logger.info(f"Zero-initializing token at index {token_id}")
                                input_embeddings[token_id].zero_()
                                if output_embeddings is not None:
                                    output_embeddings[token_id].zero_()

                            elif init_strategy == "mean":
                                # Initialize to mean of existing (non-added) embeddings
                                logger.info(f"Mean-initializing token at index {token_id}")
                                mean_embedding = input_embeddings[:old_vocab_size].mean(dim=0)
                                input_embeddings[token_id].copy_(mean_embedding)
                                if output_embeddings is not None:
                                    mean_output = output_embeddings[:old_vocab_size].mean(dim=0)
                                    output_embeddings[token_id].copy_(mean_output)

                # Update config to reflect new vocabulary size
                model_config.vocab_size = new_vocab_size
                model_config.pad_token_id = tokenizer.pad_token_id
                logger.info(
                    f"Updated config: vocab_size={new_vocab_size}, pad_token_id={tokenizer.pad_token_id}"
                )

        # Compare logits
        prompt = kwargs.get("prompt", "The old bookstore at the corner of")
        self._compare_logits(
            src_model,
            model,
            src_tokenizer,
            tokenizer,
            prompt,
            "Source HF",
            "Destination Forgather",
        )

        # Save model
        logger.info("Saving Forgather model...")
        model_config.save_pretrained(save_directory=dst_model_path)
        tokenizer.save_pretrained(save_directory=dst_model_path)

        save_checkpoint(
            output_dir=dst_model_path,
            module=model,
            safetensors=False,
            include_param_sharing=True,
        )

        logger.info(f"Conversion complete: {dst_model_path}")

    def convert_from_forgather(
        self,
        src_model_path: str,
        dst_model_path: str,
        dtype: Optional[str] = None,
        max_length: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Convert Forgather model to HuggingFace format.

        Args:
            src_model_path: Path to source Forgather model directory
            dst_model_path: Path to destination HF model directory
            dtype: Optional dtype for output model
            max_length: Optional max sequence length override
            checkpoint_path: Optional specific checkpoint to load
            **kwargs: Additional options (debug_params, prompt, etc.)
        """
        from forgather.ml.construct import torch_dtype

        logger.info(f"Converting {self.model_type} model from Forgather to HuggingFace")
        logger.info(f"Source: {src_model_path}")
        logger.info(f"Destination: {dst_model_path}")

        # Find checkpoint
        if not checkpoint_path:
            logger.info(f"Finding latest checkpoint in {src_model_path}")
            checkpoint_path = find_latest_checkpoint(src_model_path)
            if not checkpoint_path:
                raise ValueError(
                    f"No checkpoints found in {src_model_path}. "
                    "Please provide a valid Forgather model directory."
                )
        elif not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")

        logger.info(f"Using checkpoint: {checkpoint_path}")

        # Load Forgather model config
        src_model_config = AutoConfig.from_pretrained(
            src_model_path, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(src_model_path)

        # Auto-detect dtype if not specified
        if dtype is None:
            # Try to get dtype from source model config (prefer 'dtype', fall back to 'torch_dtype')
            if (
                hasattr(src_model_config, "dtype")
                and src_model_config.dtype is not None
            ):
                dtype = str(src_model_config.dtype).replace("torch.", "")
                logger.info(f"Auto-detected dtype from source model: {dtype}")
            elif (
                hasattr(src_model_config, "torch_dtype")
                and src_model_config.torch_dtype is not None
            ):
                dtype = str(src_model_config.torch_dtype).replace("torch.", "")
                logger.info(
                    f"Auto-detected dtype from source model (via deprecated torch_dtype): {dtype}"
                )
            else:
                # Default to bfloat16 if not available
                dtype = "bfloat16"
                logger.info(f"No dtype in source model, defaulting to: {dtype}")
        else:
            logger.info(f"Using specified dtype: {dtype}")

        # Setup dtype
        new_dtype = None
        if dtype:
            new_dtype = torch_dtype(dtype)

        logger.info(f"DType: {new_dtype}")

        # Load Forgather model
        logger.info("Loading Forgather model...")
        with ExitStack() as exit_stack:
            if new_dtype:
                exit_stack.enter_context(default_dtype(new_dtype))
            exit_stack.enter_context(torch.device("cpu"))
            exit_stack.enter_context(no_init_weights())
            src_model = AutoModelForCausalLM.from_config(
                src_model_config, trust_remote_code=True
            )

        load_checkpoint(checkpoint_path, src_model, device="cpu", strict=True)

        if kwargs.get("debug_params"):
            self._print_params(src_model, "Source Forgather model")

        logger.debug(f"FG Model: {src_model}")

        # Remap state dict
        logger.info("Remapping model weight names to HuggingFace format...")
        src_state_dict = src_model.state_dict()
        param_mappings = self.get_parameter_mappings("from_forgather")
        mapped_state_dict = remap_state_dict(src_state_dict, param_mappings)

        # Apply model-specific transformations
        mapped_state_dict = self.transform_state_dict(
            mapped_state_dict, "from_forgather", src_model_config, None
        )

        # Determine max length
        max_model_length = src_model_config.max_sequence_length
        if max_length:
            max_model_length = max_length

        # Create HF config and model
        logger.info(f"Creating HuggingFace {self.model_type} model...")
        hf_config = self.create_hf_config(src_model_config, max_model_length)

        with ExitStack() as exit_stack:
            if new_dtype:
                exit_stack.enter_context(default_dtype(new_dtype))
            exit_stack.enter_context(torch.device("cpu"))
            exit_stack.enter_context(hf_no_init_weights())
            hf_model_class = self.get_hf_model_class()
            hf_model = hf_model_class(hf_config)

        if kwargs.get("debug_params"):
            self._print_params(
                hf_model, f"Destination HuggingFace {self.model_type} model"
            )
            logger.info("Mapped parameter names:")
            for name in mapped_state_dict.keys():
                logger.info(f"  {name}")

        # Load state dict
        logger.info("Loading mapped state dictionary...")
        result = hf_model.load_state_dict(mapped_state_dict, strict=False, assign=True)
        logger.info(f"load_state_dict() result: {result}")

        # Retie weights
        hf_model.tie_weights()

        logger.debug(f"HF Model: {hf_model}")

        # Validate that unused parameters are only RoPE cached buffers
        if result.unexpected_keys:
            non_rope_unexpected = [
                key
                for key in result.unexpected_keys
                if not (key.endswith(".cos_cached") or key.endswith(".sin_cached"))
            ]
            if non_rope_unexpected:
                logger.warning(
                    f"Warning: Unexpected non-RoPE parameters: {non_rope_unexpected}"
                )
            else:
                logger.info(
                    f"As expected, {len(result.unexpected_keys)} RoPE cached buffers "
                    "were not loaded (they will be recomputed)"
                )

        # Compare logits
        prompt = kwargs.get("prompt", "The old bookstore at the corner of")
        self._compare_logits(
            src_model,
            hf_model,
            tokenizer,
            prompt,
            "Source Forgather",
            f"Destination HuggingFace {self.model_type}",
        )

        # Save model
        logger.info(f"Saving HuggingFace {self.model_type} model...")
        hf_model.save_pretrained(dst_model_path)
        tokenizer.save_pretrained(dst_model_path)

        logger.info(f"Conversion complete: {dst_model_path}")

    def _print_params(self, model, label: str):
        """Print parameter names for debugging."""
        logger.info(f"{label} parameter names:")
        for name in model.state_dict().keys():
            logger.info(f"  {name}")

    def _compare_logits(
        self,
        src_model,
        dst_model,
        src_tokenizer,
        dst_tokenizer,
        prompt: str,
        src_label: str = "Source",
        dst_label: str = "Destination",
        tolerance: float = 1e-5,
    ):
        """Compare logits between source and destination models."""
        src_logits = self._test_forward(src_model, src_tokenizer, prompt)
        dst_logits = self._test_forward(dst_model, dst_tokenizer, prompt)

        # Handle vocab size mismatch
        if src_logits.shape != dst_logits.shape:
            src_vocab = src_logits.shape[-1]
            dst_vocab = dst_logits.shape[-1]
            min_vocab_size = min(src_vocab, dst_vocab)

            logger.info(f"Vocab size mismatch: {src_vocab} vs {dst_vocab}")
            if dst_vocab > src_vocab:
                logger.info(
                    f"Destination has {dst_vocab - src_vocab} additional tokens"
                )
                logger.info(
                    f"Comparing only original {src_vocab} tokens (new tokens not in source)"
                )
            else:
                logger.info(f"Source has {src_vocab - dst_vocab} additional tokens")
                logger.info(f"Comparing only first {min_vocab_size} tokens")

            src_logits = src_logits[..., :min_vocab_size]
            dst_logits = dst_logits[..., :min_vocab_size]

        if not torch.allclose(src_logits, dst_logits, atol=tolerance):
            logger.warning("WARNING: Model logits are dissimilar")
            logger.warning(f"{src_label} Model Logits shape: {src_logits.shape}")
            logger.warning(f"{dst_label} Model Logits shape: {dst_logits.shape}")
            logger.warning(
                f"Max diff: {torch.max(torch.abs(src_logits - dst_logits)).item()}"
            )
            logger.warning(
                f"Mean diff: {torch.mean(torch.abs(src_logits - dst_logits)).item()}"
            )
            logger.warning(
                f"{src_label} logits range: "
                f"[{src_logits.min().item():.6f}, {src_logits.max().item():.6f}]"
            )
            logger.warning(
                f"{dst_label} logits range: "
                f"[{dst_logits.min().item():.6f}, {dst_logits.max().item():.6f}]"
            )
            logger.info(f"{src_label} src logits: {src_logits}")
            logger.info(f"{dst_label} src logits: {dst_logits}")
        else:
            logger.warning("Model logits match.")

    def _test_forward(self, model, tokenizer, prompt: str):
        """Run forward pass and return logits."""
        model.to("cpu")
        model.eval()

        tokenizer_outputs = tokenizer(
            [prompt],
            truncation=False,
            return_length=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        with torch.inference_mode():
            input_ids = tokenizer_outputs["input_ids"].to("cpu")
            outputs = model(input_ids, return_dict=True)
            logits = outputs.logits

        return logits
