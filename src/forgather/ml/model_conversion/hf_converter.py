"""Base HuggingFace model converter implementation."""

import os
import logging
from contextlib import ExitStack
from typing import Optional, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.modeling_utils import no_init_weights as hf_no_init_weights

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

    def __init__(self, model_type: str, model_project_dir: str):
        """Initialize HuggingFace converter.

        Args:
            model_type: String identifier for the model type (e.g., "llama", "mistral")
            model_project_dir: Path to the model project directory (e.g., "examples/models/llama")
        """
        super().__init__(model_type)
        self.model_project_dir = model_project_dir

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

        # Setup dtype
        new_dtype = None
        if dtype:
            new_dtype = torch_dtype(dtype)

        logger.info(f"Converting {self.model_type} model from HuggingFace to Forgather")
        logger.info(f"Source: {src_model_path}")
        logger.info(f"Destination: {dst_model_path}")
        logger.info(f"DType: {new_dtype}")

        # Load source model config and validate
        src_model_config = AutoConfig.from_pretrained(src_model_path)
        self.validate_source_config(src_model_config, "to_forgather")

        # Capture original HF model type for reverse conversion
        hf_model_type = src_model_config.model_type
        logger.info(f"Capturing HF model type: {hf_model_type}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(src_model_path)

        # Load source model
        print("Loading source HuggingFace model...")
        with ExitStack() as exit_stack:
            if new_dtype:
                exit_stack.enter_context(default_dtype(new_dtype))
            src_model = AutoModelForCausalLM.from_pretrained(src_model_path)
        logger.debug(src_model)

        if kwargs.get("debug_params"):
            self._print_params(src_model, "Source HuggingFace model")

        # Remap state dict
        print("Remapping model weight names...")
        src_state_dict = src_model.state_dict()
        param_mappings = self.get_parameter_mappings("to_forgather")
        mapped_state_dict = remap_state_dict(src_state_dict, param_mappings)

        # Apply model-specific transformations
        mapped_state_dict = self.transform_state_dict(
            mapped_state_dict, "to_forgather", src_model_config, None
        )

        # Create Forgather project configuration
        print("Creating Forgather model...")

        # Get model-specific project config from converter
        config_args = self.create_project_config(src_model_config, max_length)

        proj = Project(
            config_name="",
            project_dir=self.model_project_dir,
            output_dir=dst_model_path,
            tokenizer_id_or_path=src_model_path,
            **config_args,
        )

        # Dump config for diagnostics
        print(proj.pp_config)

        # Validate project type
        proj_meta = proj("meta")
        config_class = proj_meta["config_class"]
        if config_class != "type.model":
            raise TypeError(f"Expected class type.model, found {config_class}")

        # Get model components
        model_config, tokenizer, model_ctor = proj(
            "pretrained_config", "pretrained_tokenizer", "model"
        )

        # Update vocab_size to match source model
        # Some models have different vocab sizes in config vs tokenizer
        if src_model_config.vocab_size != model_config.vocab_size:
            print(
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

        # Store HF model type in config for reverse conversion
        # This allows auto-detection of the correct converter for FG->HF conversion
        model_config.hf_model_type = hf_model_type
        logger.info(f"Stored hf_model_type={hf_model_type} in Forgather config")

        # Construct model
        print("Constructing Forgather model...")
        with ExitStack() as exit_stack:
            if new_dtype:
                exit_stack.enter_context(default_dtype(new_dtype))
            exit_stack.enter_context(no_init_weights())
            model = model_ctor()
        logger.debug(model)

        if kwargs.get("debug_params"):
            self._print_params(model, "Destination Forgather model")

        # Load state dict
        print("Loading mapped state dictionary...")
        result = model.load_state_dict(mapped_state_dict, strict=False, assign=True)
        print(f"load_state_dict() result: {result}")

        # Tie weights if needed (must be done after loading state dict)
        if (
            hasattr(model_config, "tie_word_embeddings")
            and model_config.tie_word_embeddings
        ):
            print("Tying word embeddings...")
            model.tie_weights()

        # Compare logits
        prompt = kwargs.get("prompt", "The old bookstore at the corner of")
        self._compare_logits(
            src_model, model, tokenizer, prompt, "Source HF", "Destination Forgather"
        )

        # Save model
        print("Saving Forgather model...")
        model_config.save_pretrained(save_directory=dst_model_path)
        tokenizer.save_pretrained(save_directory=dst_model_path)

        save_checkpoint(
            output_dir=dst_model_path,
            module=model,
            safetensors=False,
            include_param_sharing=True,
        )

        print(f"Conversion complete: {dst_model_path}")

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

        # Setup dtype
        new_dtype = None
        if dtype:
            new_dtype = torch_dtype(dtype)

        logger.info(f"Converting {self.model_type} model from Forgather to HuggingFace")
        logger.info(f"Source: {src_model_path}")
        logger.info(f"Destination: {dst_model_path}")
        logger.info(f"DType: {new_dtype}")

        # Find checkpoint
        if not checkpoint_path:
            print(f"Finding latest checkpoint in {src_model_path}")
            checkpoint_path = find_latest_checkpoint(src_model_path)
            if not checkpoint_path:
                raise ValueError(
                    f"No checkpoints found in {src_model_path}. "
                    "Please provide a valid Forgather model directory."
                )
        elif not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")

        print(f"Using checkpoint: {checkpoint_path}")

        # Load Forgather model config
        src_model_config = AutoConfig.from_pretrained(
            src_model_path, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(src_model_path)

        # Load Forgather model
        print("Loading Forgather model...")
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

        print(f"FG Model: {src_model}")

        # Remap state dict
        print("Remapping model weight names to HuggingFace format...")
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
        print(f"Creating HuggingFace {self.model_type} model...")
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
            print("Mapped parameter names:")
            for name in mapped_state_dict.keys():
                print(f"  {name}")

        # Load state dict
        print("Loading mapped state dictionary...")
        result = hf_model.load_state_dict(mapped_state_dict, strict=False, assign=True)
        print(f"load_state_dict() result: {result}")

        # Retie weights
        hf_model.tie_weights()

        print(f"HF Model: {hf_model}")

        # Validate that unused parameters are only RoPE cached buffers
        if result.unexpected_keys:
            non_rope_unexpected = [
                key
                for key in result.unexpected_keys
                if not (key.endswith(".cos_cached") or key.endswith(".sin_cached"))
            ]
            if non_rope_unexpected:
                print(f"Warning: Unexpected non-RoPE parameters: {non_rope_unexpected}")
            else:
                print(
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
        print(f"Saving HuggingFace {self.model_type} model...")
        hf_model.save_pretrained(dst_model_path)
        tokenizer.save_pretrained(dst_model_path)

        print(f"Conversion complete: {dst_model_path}")

    def _print_params(self, model, label: str):
        """Print parameter names for debugging."""
        print(f"{label} parameter names:")
        for name in model.state_dict().keys():
            print(f"  {name}")

    def _compare_logits(
        self,
        src_model,
        dst_model,
        tokenizer,
        prompt: str,
        src_label: str = "Source",
        dst_label: str = "Destination",
        tolerance: float = 1e-5,
    ):
        """Compare logits between source and destination models."""
        src_logits = self._test_forward(src_model, tokenizer, prompt)
        dst_logits = self._test_forward(dst_model, tokenizer, prompt)

        # Handle vocab size mismatch
        if src_logits.shape != dst_logits.shape:
            min_vocab_size = min(src_logits.shape[-1], dst_logits.shape[-1])
            print(
                f"Vocab size mismatch: {src_logits.shape[-1]} vs {dst_logits.shape[-1]}"
            )
            print(f"Comparing only first {min_vocab_size} tokens")
            src_logits = src_logits[..., :min_vocab_size]
            dst_logits = dst_logits[..., :min_vocab_size]

        if not torch.allclose(src_logits, dst_logits, atol=tolerance):
            print("WARNING: Model logits are dissimilar")
            print(f"{src_label} Model Logits shape: {src_logits.shape}")
            print(f"{dst_label} Model Logits shape: {dst_logits.shape}")
            print(f"Max diff: {torch.max(torch.abs(src_logits - dst_logits)).item()}")
            print(f"Mean diff: {torch.mean(torch.abs(src_logits - dst_logits)).item()}")
            print(
                f"{src_label} logits range: "
                f"[{src_logits.min().item():.6f}, {src_logits.max().item():.6f}]"
            )
            print(
                f"{dst_label} logits range: "
                f"[{dst_logits.min().item():.6f}, {dst_logits.max().item():.6f}]"
            )
            print(f"{src_label} src logits: {src_logits}")
            print(f"{dst_label} src logits: {dst_logits}")
        else:
            print("Model logits match.")

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
