"""
Inference service core - model loading and infrastructure.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Set, Union, Callable, Any
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    AutoConfig,
)
from jinja2 import Environment, BaseLoader, TemplateError

from forgather.ml.no_init_weights import no_init_weights
from forgather.ml.utils import default_dtype
from forgather.ml.sharded_checkpoint import load_checkpoint, find_latest_checkpoint
from forgather.ml.construct import torch_dtype

from .core import (
    StopSequenceProcessor,
    FinishReasonDetector,
    TokenizerWrapper,
    GenerationLogger,
)
from .models.chat import ChatMessage, ChatCompletionRequest
from .models.completion import CompletionRequest


class InferenceService:
    """
    Core inference service handling model, tokenizer, and generation infrastructure.

    This service manages the model lifecycle, tokenization, and provides utilities
    for generation strategies. It does not perform generation itself - that's
    delegated to strategy classes.

    Example:
        Basic usage with default settings:
        >>> service = InferenceService(
        ...     model_path="./my_model",
        ...     device="cuda:0",
        ...     dtype="bfloat16"
        ... )

        Loading from checkpoint:
        >>> service = InferenceService(
        ...     model_path="./my_model",
        ...     device="cuda:0",
        ...     from_checkpoint=True  # Auto-find latest checkpoint
        ... )

        With custom stop sequences and chat template:
        >>> service = InferenceService(
        ...     model_path="./my_model",
        ...     stop_sequences=["<|im_end|>", "</s>"],
        ...     chat_template_path="./custom_template.jinja"
        ... )

    Attributes:
        model: Loaded HuggingFace model
        tokenizer: Loaded HuggingFace tokenizer
        stop_processor: Utility for trimming at stop sequences
        finish_detector: Utility for determining finish reasons
        tokenizer_wrapper: Utility for tokenization and device placement
        logger: Utility for consistent logging
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        attn_implementation: Optional[str] = None,
        from_checkpoint: bool | str = False,
        chat_template_path: Optional[str] = None,
        dtype: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        compile_args: Optional[dict[str, Any]] = None,
        cache_implementation: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> None:
        """
        Initialize inference service.

        Args:
            model_path: Path to model directory
            device: Device to use (cuda:0, cpu, auto)
            attn_implementation: Attention implementation (eager, sdpa, flash_attention_2, flex_attention)
            from_checkpoint: Load from checkpoint (bool or checkpoint path)
            chat_template_path: Path to custom chat template file
            dtype: Model dtype (float32, float16, bfloat16, etc.)
            stop_sequences: Custom stop sequences

        Raises:
            ValueError: If invalid device, checkpoint path, or dtype specified
        """
        # Create dedicated logger for this application
        self.logger = GenerationLogger(
            logging.getLogger("inference_server"),
            None,  # Will be set after tokenizer is loaded
        )

        self.model_path = model_path
        self.device = device
        self.attn_implementation = attn_implementation
        self.from_checkpoint = from_checkpoint
        self.chat_template_path = chat_template_path
        self.dtype = self._resolve_dtype(dtype)
        self.stop_sequences = stop_sequences or []
        self.chat_template = None
        self.tokenizer = None
        self.model = None
        self.default_generation_config = None
        self.jinja_env = Environment(loader=BaseLoader())
        self.compile_args = compile_args
        self.cache_implementation = cache_implementation
        self.use_cache = use_cache

        # Load model and setup
        self.load_model()
        self.setup_chat_template()
        self._setup_stop_tokens()

        # Initialize core utilities after model/tokenizer are loaded
        self.stop_processor = StopSequenceProcessor(self.tokenizer)
        self.finish_detector = FinishReasonDetector(self.tokenizer, self.stop_token_ids)
        self.tokenizer_wrapper = TokenizerWrapper(self.tokenizer, self.model)

        # Update logger's tokenizer reference
        self.logger.tokenizer = self.tokenizer

    def load_model(self):
        """Load model and tokenizer from directory."""
        self.logger.logger.info(f"Loading model from directory {self.model_path}")

        # This can speed up float32 ops on newer GPUs
        torch.set_float32_matmul_precision("high")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.from_checkpoint:
            if self.device == "auto":
                raise ValueError(
                    "Cannot use 'auto' device with checkpoint loading. Please specify a device explicitly."
                )

            if isinstance(self.from_checkpoint, bool):
                checkpoint_path = find_latest_checkpoint(self.model_path)
                if not checkpoint_path:
                    raise ValueError(
                        f"No checkpoints found in {self.model_path}. Please provide a valid model directory."
                    )
            elif isinstance(self.from_checkpoint, str):
                checkpoint_path = self.from_checkpoint
            else:
                raise ValueError("from_checkpoint must be a boolean or a string path")
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")

            self.logger.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            model_config = AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            # Create model on target device with no_init_weights()
            with (
                torch.device(self.device),
                default_dtype(dtype=self.dtype),
                no_init_weights(),
            ):
                model = AutoModelForCausalLM.from_config(
                    model_config,
                    trust_remote_code=True,
                    attn_implementation=self.attn_implementation,
                )

            # Load checkpoint parameters
            load_checkpoint(checkpoint_path, model, device=self.device, strict=True)
            self.model = model

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=self.dtype,
                device_map=self.device if self.device != "auto" else "auto",
                trust_remote_code=True,
            )

            if self.device != "auto" and torch.cuda.is_available():
                self.model = self.model.to(self.device)

        self.model.eval()

        if self.compile_args is not None:
            if self.compile_args.get("backend", "") == "tensorrt":
                try:
                    import torch_tensorrt
                except Exception as e:
                    logging.warning(
                        "torch_tensor module not available; falling back to default."
                    )
                    self.compile_args.pop("backend")

            self.model.compile(**self.compile_args)

        # Load generation config from model directory if available
        self._load_generation_config()

        self.logger.logger.info(
            f"Model loaded successfully on device: {self.model.device} with dtype: {self.dtype}"
        )
        self.logger.logger.debug(self.model)

    def _load_generation_config(self):
        """Load generation config from model directory if available."""
        try:
            self.default_generation_config = GenerationConfig.from_pretrained(
                self.model_path
            )
            self.logger.logger.info(
                f"Loaded generation config from model directory: {self.model_path}"
            )
            self.logger.logger.info(
                f"Default generation config: {self.default_generation_config}"
            )
        except Exception as e:
            self.logger.logger.info(
                f"No generation config found in model directory or failed to load: {e}"
            )
            # Fallback to model's generation config if available
            if (
                hasattr(self.model, "generation_config")
                and self.model.generation_config is not None
            ):
                self.default_generation_config = self.model.generation_config
                self.logger.logger.info("Using model's built-in generation config")
            else:
                self.default_generation_config = GenerationConfig()
                self.logger.logger.info("Using default GenerationConfig")
        self.logger.logger.info(
            f"Final default generation config: {self.default_generation_config}"
        )
        if (
            self.default_generation_config.temperature != 0.0
            and self.default_generation_config.top_p != 0.0
        ):
            self.logger.logger.warning(
                f"Both temperature ({self.default_generation_config.temperature}) and top_p "
                f"({self.default_generation_config.top_p}) are set != 1 in generation config. "
                "It is recommend to set only one of these to != 1. "
                "See: https://platform.openai.com/docs/api-reference/completions/create"
            )

    def _resolve_dtype(self, dtype_str: Optional[str]) -> torch.dtype:
        """Resolve dtype string to torch.dtype with intelligent defaults."""
        if dtype_str is None:
            # Default to bfloat16 if supported, otherwise float16 on GPU, float32 on CPU
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    return torch.float16
            else:
                return torch.float32

        dtype_str = dtype_str.lower()
        requested_dtype = torch_dtype(dtype_str)

        # Validate bfloat16 support
        if (
            requested_dtype == torch.bfloat16
            and torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()
        ):
            self.logger.logger.warning(
                f"bfloat16 not supported on this GPU, falling back to float16"
            )
            return torch.float16

        return requested_dtype

    def setup_chat_template(self):
        """Setup chat template with priority: custom file > tokenizer > default fallback."""
        if self.chat_template_path and os.path.exists(self.chat_template_path):
            # Use custom template file
            with open(self.chat_template_path, "r") as f:
                self.chat_template = f.read()
            self.logger.logger.info(
                f"Using custom chat template from: {self.chat_template_path}"
            )
        elif hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            # Use tokenizer's built-in template
            self.chat_template = self.tokenizer.chat_template
            self.logger.logger.info("Using tokenizer's built-in chat template")
        else:
            # Use default fallback template
            self.chat_template = self.get_default_chat_template()
            self.logger.logger.info("Using default fallback chat template")
        self.logger.logger.info(f"Chat template loaded: {repr(self.chat_template)}")

    def _setup_stop_tokens(self):
        """Setup stop token IDs from stop sequences."""
        self.stop_token_ids: Set[int] = set()

        # Always include native EOS token
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)

        # Add custom stop sequences
        for sequence in self.stop_sequences:
            try:
                token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
                if len(token_ids) == 1:
                    # Single token - can use as direct stopping criterion
                    self.stop_token_ids.add(token_ids[0])
                    self.logger.logger.info(
                        f"Added single-token stop sequence: {repr(sequence)} -> token ID {token_ids[0]}"
                    )
                else:
                    # Multi-token sequence - will need post-processing
                    self.logger.logger.info(
                        f"Added multi-token stop sequence: {repr(sequence)} -> token IDs {token_ids}"
                    )
            except Exception as e:
                self.logger.logger.warning(
                    f"Failed to tokenize stop sequence {repr(sequence)}: {e}"
                )

        self.logger.logger.info(f"Stop token IDs: {sorted(self.stop_token_ids)}")

    def _build_generation_config(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> GenerationConfig:
        """Build a GenerationConfig from request parameters."""
        # Base parameters that are always used
        max_tokens = getattr(request, "max_new_tokens", None) or getattr(
            request, "max_tokens", 16
        )

        # Start with the loaded default generation config
        if self.default_generation_config is not None:
            # Create a copy to avoid modifying the default
            generation_config = GenerationConfig(
                **self.default_generation_config.to_dict()
            )
        else:
            generation_config = GenerationConfig()

        # Core parameters
        generation_config.max_new_tokens = max_tokens
        if request.temperature is not None:
            generation_config.temperature = request.temperature

        if request.top_p is not None:
            generation_config.top_p = request.top_p
        generation_config.do_sample = (
            request.temperature is None or request.temperature > 0
        )
        if not hasattr(generation_config, "pad_token_id"):
            generation_config.pad_token_id = self.tokenizer.pad_token_id

        if not hasattr(generation_config, "eos_token_id"):
            generation_config.eos_token_id = self.tokenizer.eos_token_id

        if not hasattr(generation_config, "bos_token_id"):
            generation_config.bos_token_id = self.tokenizer.bos_token_id

        generation_config.return_dict_in_generate = True
        generation_config.output_scores = False

        # Set early_stopping properly - only use with beam search (num_beams > 1)
        early_stopping_value = getattr(request, "early_stopping", None)
        if early_stopping_value is not None:
            generation_config.early_stopping = early_stopping_value

        # Add HuggingFace specific parameters if they are not None
        hf_params = [
            "repetition_penalty",
            "length_penalty",
            "no_repeat_ngram_size",
            "encoder_no_repeat_ngram_size",
            "bad_words_ids",
            "min_length",
            "num_beams",
            "num_beam_groups",
            "diversity_penalty",
            "temperature_last_layer",
            "top_k",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
            "guidance_scale",
        ]

        for param in hf_params:
            value = getattr(request, param, None)
            if value is not None:
                setattr(generation_config, param, value)

        # Handle special cases
        if hasattr(request, "seed") and request.seed is not None:
            # Set random seed for reproducibility
            torch.manual_seed(request.seed)

        if self.use_cache is not None:
            generation_config.use_cache = self.use_cache

        if self.cache_implementation is not None:
            generation_config.cache_implementation = self.cache_implementation

        # If using beam search, adjust sampling and early_stopping
        if generation_config.num_beams and generation_config.num_beams > 1:
            generation_config.do_sample = False  # Beam search doesn't use sampling
            # Only enable early_stopping with beam search if not explicitly set
            if early_stopping_value is None:
                generation_config.early_stopping = True

        return generation_config

    def get_default_chat_template(self) -> str:
        """Return a reasonable default chat template as Jinja2."""
        return """{%- for message in messages %}
    {%- if message['role'] == 'system' -%}
        System: {{ message['content'] }}\\n\\n
    {%- elif message['role'] == 'user' -%}
        User: {{ message['content'] }}\\n\\n
    {%- elif message['role'] == 'assistant' -%}
        Assistant: {{ message['content'] }}\\n\\n
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    Assistant:
{%- endif -%}"""

    def format_messages(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a single prompt string using Jinja2 template."""
        try:
            # Prepare message data for template
            message_data = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            # Create Jinja2 template and render
            template = self.jinja_env.from_string(self.chat_template)
            formatted = template.render(
                messages=message_data,
                bos_token=self.tokenizer.bos_token,
                eos_token=self.tokenizer.eos_token,
                add_generation_prompt=True,
            )
            return formatted

        except TemplateError as e:
            self.logger.logger.error(f"Template error: {e}")
            # Fallback to simple formatting if template fails
            return self._fallback_format_messages(messages)
        except Exception as e:
            self.logger.logger.error(f"Unexpected error in template rendering: {e}")
            return self._fallback_format_messages(messages)

    def _fallback_format_messages(self, messages: List[ChatMessage]) -> str:
        """Simple fallback formatting if template fails."""
        formatted = ""
        for message in messages:
            if message.role == "system":
                formatted += f"System: {message.content}\n\n"
            elif message.role == "user":
                formatted += f"User: {message.content}\n\n"
            elif message.role == "assistant":
                formatted += f"Assistant: {message.content}\n\n"

        formatted += "Assistant: "
        return formatted
