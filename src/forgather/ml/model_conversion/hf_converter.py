"""Base HuggingFace model converter implementation."""

import os
import logging
from contextlib import ExitStack
import shutil
from typing import Optional, Dict, Any, Tuple, Callable, override
from abc import abstractmethod
import torch
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
from .resize_embeddings import (
    add_tokens_to_tokenizer,
    resize_word_embeddings,
    update_config_from_tokenizer,
    DEFAULT_TOKEN_CONFIG,
)

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

    @abstractmethod
    def get_hf_config_class(self):
        """Get HuggingFace config class for this model type.

        Returns:
            HuggingFace config class (e.g., LlamaConfig)
        """
        raise NotImplementedError("Subclasses must implement get_hf_config_class()")

    @abstractmethod
    def get_hf_model_class(self):
        """Get HuggingFace model class for this model type.

        Returns:
            HuggingFace model class (e.g., LlamaForCausalLM)
        """
        raise NotImplementedError("Subclasses must implement get_hf_model_class()")

    @abstractmethod
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
        field_mapping = self.get_config_field_mapping("to_forgather")
        # Build config kwargs from field mapping
        config_kwargs = {}
        for hf_field, fg_field in field_mapping.items():
            if hasattr(src_config, hf_field):
                config_kwargs[fg_field] = getattr(src_config, hf_field)

        # Optional override of max_position_embeddings
        if "max_position_embeddings" in config_kwargs and max_length:
            config_kwargs["max_position_embeddings"] = max_length

        return config_kwargs

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

        # Build config kwargs from field mapping
        config_kwargs = {}
        for fg_field, hf_field in field_mapping.items():
            if hasattr(src_config, fg_field):
                config_kwargs[hf_field] = getattr(src_config, fg_field)

        # Optional override of max_position_embeddings
        if "max_position_embeddings" in config_kwargs and max_length:
            config_kwargs["max_position_embeddings"] = max_length

        return config_class(**config_kwargs)

    def convert_to_forgather(
        self,
        src_model_path: str,
        dst_model_path: str,
        dtype: Optional[str] = None,
        max_length: Optional[int] = None,
        test_device: Optional[str] = None,
        dry_run: bool = False,
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

        # Copy generation config
        self._copy_generation_config(src_model_path, dst_model_path)

        # Add tokens and resize embeddings if requested
        # Use default config if no explicit add_tokens provided (unless skip_default_tokens is set)
        token_config = kwargs.get("add_tokens")
        if token_config is None and not kwargs.get("skip_default_tokens", False):
            logger.info("Using default token configuration (adding missing PAD token)")
            token_config = DEFAULT_TOKEN_CONFIG

        if token_config is not None:
            if isinstance(token_config, str):
                logger.info(f"Adding tokens from: {token_config}")
            else:
                logger.info("Adding tokens from provided configuration")

            num_added, token_inits = add_tokens_to_tokenizer(tokenizer, token_config)

            if num_added > 0:
                logger.info(f"Added {num_added} token(s) to vocabulary")
                resize_word_embeddings(model, tokenizer, token_inits)
                # Update vocab size only when embeddings were actually resized
                update_config_from_tokenizer(
                    model_config, tokenizer, update_vocab_size=True
                )
            else:
                # Still update special token IDs even if no tokens were added
                update_config_from_tokenizer(
                    model_config, tokenizer, update_vocab_size=False
                )

        # Merge eos token set
        if hasattr(src_model_config, "eos_token_id") and hasattr(
            model_config, "eos_token_id"
        ):
            eos_set = set()
            eos_set.add(model_config.eos_token_id)
            if isinstance(src_model_config.eos_token_id, list):
                for id in src_model_config.eos_token_id:
                    eos_set.add(id)
            else:
                eos_set.add(src_model_config.eos_token_id)
            if len(eos_set) > 1:
                model_config.eos_token_id = [id for id in eos_set]

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
            test_device=test_device,
        )

        if not dry_run:
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
        else:
            print(model_config)
            shutil.rmtree(dst_model_path)

    def convert_from_forgather(
        self,
        src_model_path: str,
        dst_model_path: str,
        dtype: Optional[str] = None,
        max_length: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        test_device: Optional[str] = None,
        dry_run: bool = False,
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

        # The hf_no_init_weights() context manager disabled tie_weights() in post_init(), so
        # we need to call it explicitly.
        if hasattr(hf_model, "tie_weights"):
            hf_model.tie_weights()

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

        # Copy generation config
        self._copy_generation_config(src_model_path, dst_model_path)

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
            test_device=test_device,
        )

        if not dry_run:
            # Save model
            logger.info(f"Saving HuggingFace {self.model_type} model...")
            hf_model.save_pretrained(dst_model_path)
            tokenizer.save_pretrained(dst_model_path)

            logger.info(f"Conversion complete: {dst_model_path}")
        else:
            print(hf_config)

    @staticmethod
    def _copy_generation_config(src_model_path: str, dst_model_path):
        config_name = "generation_config.json"
        src_config_path = os.path.join(src_model_path, config_name)
        if os.path.isfile(src_config_path):
            dst_config_path = os.path.join(dst_model_path, config_name)
            logger.info(
                f"Copy generation config from {src_config_path} to {dst_config_path}"
            )
            shutil.copyfile(src_config_path, dst_config_path)

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
        test_device: Optional[str] = None,
    ):
        """Compare logits between source and destination models."""
        src_logits = self._test_forward(src_model, src_tokenizer, prompt, test_device)
        dst_logits = self._test_forward(dst_model, dst_tokenizer, prompt, test_device)

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

    def _test_forward(
        self, model, tokenizer, prompt: str, test_device: Optional[str] = None
    ):
        """Run forward pass and return logits."""
        if test_device:
            device = torch.device(test_device)
        else:
            device = torch.device("cpu")
        model.to(device)
        model.eval()

        tokenizer_outputs = tokenizer(
            [prompt],
            truncation=False,
            return_length=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        with torch.inference_mode():
            input_ids = tokenizer_outputs["input_ids"].to(device)
            outputs = model(input_ids, return_dict=True)
            logits = outputs.logits

        model.to("cpu")
        return logits.to("cpu")
