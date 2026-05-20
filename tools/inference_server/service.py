"""
Inference service core - model loading and infrastructure.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set, Union

import torch
from jinja2 import BaseLoader, Environment, TemplateError
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from forgather.ml.construct import torch_dtype
from forgather.ml.no_init_weights import no_init_weights
from forgather.ml.sharded_checkpoint import find_latest_checkpoint, load_checkpoint
from forgather.ml.utils import default_dtype

from .core import (
    FinishReasonDetector,
    GenerationLogger,
    StopSequenceProcessor,
    TokenizerWrapper,
)
from .models.chat import ChatCompletionRequest, ChatMessage
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
        ignore_eos: bool = False,
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
            compile_args: Arguments for torch.compile
            cache_implementation: KV cache implementation
            use_cache: Whether to use KV cache
            ignore_eos: Server-level default for ignoring EOS tokens (can be overridden per-request)

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
        self.ignore_eos = ignore_eos

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
                attn_implementation=self.attn_implementation,
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
        if (
            not hasattr(generation_config, "pad_token_id")
            or generation_config.pad_token_id is None
        ):
            generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Handle EOS token - conditionally disable if ignore_eos is True
        # Check request-level ignore_eos, fall back to server-level default
        request_ignore_eos = getattr(request, "ignore_eos", None)
        ignore_eos = (
            request_ignore_eos if request_ignore_eos is not None else self.ignore_eos
        )
        if ignore_eos:
            # Set to -1 (impossible token ID) to prevent HF from stopping on EOS
            # Note: Setting to None doesn't work because HuggingFace fills it from model defaults
            generation_config.eos_token_id = -1
        else:
            # Normal behavior: ensure eos_token_id is set
            if (
                not hasattr(generation_config, "eos_token_id")
                or generation_config.eos_token_id is None
            ):
                generation_config.eos_token_id = self.tokenizer.eos_token_id

        if (
            not hasattr(generation_config, "bos_token_id")
            or generation_config.bos_token_id is None
        ):
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
            "min_new_tokens",
            "num_beams",
            "num_beam_groups",
            "diversity_penalty",
            "temperature_last_layer",
            "top_k",
            "typical_p",
            "min_p",
            "epsilon_cutoff",
            "eta_cutoff",
            "guidance_scale",
            "penalty_alpha",
            "presence_penalty",
            "frequency_penalty",
        ]

        for param in hf_params:
            value = getattr(request, param, None)
            if value is not None:
                setattr(generation_config, param, value)

        # Explicit do_sample override wins over the temperature-derived
        # default set above. Lets the user force greedy decoding even when
        # temperature is set, or force sampling when temperature is 0.
        do_sample_override = getattr(request, "do_sample", None)
        if do_sample_override is not None:
            generation_config.do_sample = do_sample_override

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

    def format_messages(
        self,
        messages: List[ChatMessage],
        next_role: Optional[str] = None,
        add_generation_prompt: Optional[bool] = None,
        continue_final_message: bool = False,
    ) -> str:
        """Convert chat messages to a single prompt string using Jinja2 template.

        ``next_role`` selects which role the model generates as. ``"user"``
        enables "impersonate": the template is rendered twice, the
        assistant role-opener is extracted, and ``assistant`` is
        substituted with ``user`` so the model generates inside an
        opened user-role span. Works with any chat template that spells
        the literal role name in its role marker (ChatML, Llama 3, Qwen,
        Mistral, Gemma, …) — i.e., effectively all of them — without
        depending on ``continue_final_message`` being honored.

        ``add_generation_prompt`` / ``continue_final_message`` mirror
        the vLLM /tokenize flags. ``next_role`` overrides them when set
        to ``"user"``. When ``add_generation_prompt`` is None, defaults
        to True (the chat-completion path) — this preserves the
        existing call sites that don't pass it.
        """
        try:
            message_data = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]
            template = self.jinja_env.from_string(self.chat_template)
            role = (next_role or "assistant").lower()
            render_kwargs = dict(
                bos_token=self.tokenizer.bos_token,
                eos_token=self.tokenizer.eos_token,
            )

            if role == "user":
                closed = template.render(
                    messages=message_data,
                    add_generation_prompt=False,
                    continue_final_message=False,
                    **render_kwargs,
                )
                with_prompt = template.render(
                    messages=message_data,
                    add_generation_prompt=True,
                    continue_final_message=False,
                    **render_kwargs,
                )
                if (
                    with_prompt.startswith(closed)
                    and len(with_prompt) > len(closed)
                    and "assistant" in with_prompt[len(closed) :]
                ):
                    opener = with_prompt[len(closed) :]
                    return closed + opener.replace("assistant", "user")
                # The template doesn't expose a usable assistant opener
                # (doesn't honor add_generation_prompt, normalizes
                # whitespace, or uses a non-literal role marker). Log
                # and fall through to the standard path so the caller
                # at least gets a working completion — impersonate
                # silently degrades to a normal assistant turn.
                self.logger.logger.warning(
                    "format_messages: cannot synthesize user-role opener from "
                    "this chat template; falling back to standard assistant turn."
                )

            return template.render(
                messages=message_data,
                add_generation_prompt=(
                    True if add_generation_prompt is None else add_generation_prompt
                ),
                continue_final_message=continue_final_message,
                **render_kwargs,
            )

        except TemplateError as e:
            self.logger.logger.error(f"Template error: {e}")
            # Fallback to simple formatting if template fails
            return self._fallback_format_messages(messages)
        except Exception as e:
            self.logger.logger.error(f"Unexpected error in template rendering: {e}")
            return self._fallback_format_messages(messages)

    def score_prompt(self, text: str, top_k: int = 10) -> dict:
        """Score an input string with per-token causal-LM logprobs.

        Runs a single forward pass (no ``generate()``) and returns the
        OpenAI legacy-completions ``logprobs`` structure plus a
        Forgather extension field:

            {
              "tokens": [...],
              "token_logprobs": [None, lp1, lp2, ...],
              "top_logprobs": [None, {tok: lp, ...}, ...],
              "text_offset": [0, off1, off2, ...],
              "token_entropies": [None, h1, h2, ...],   # Forgather extension
            }

        Position 0 is ``None`` because a causal LM has no prediction
        for the first token. Matches what vLLM returns for the same
        request shape (``echo=true, logprobs=K, max_tokens=0``); the
        ``token_entropies`` field is non-standard (the full-vocab
        Shannon entropy in nats at each prediction position) and only
        Forgather returns it — clients should treat it as optional.
        """
        # Tokenize without padding so the returned length equals the
        # actual token count — needed for accurate alignment.
        enc = self.tokenizer_wrapper.tokenize_and_move_to_device(
            text,
            max_length=2048,
            padding=False,
            truncation=True,
        )
        input_ids = enc["input_ids"]  # (1, N)
        n = int(input_ids.shape[1])
        if n == 0:
            return {
                "tokens": [],
                "token_logprobs": [],
                "top_logprobs": [],
                "text_offset": [],
                "token_entropies": [],
            }

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
        # Logits in float32 for numerically stable log_softmax; bf16
        # log_softmax across a 32k+ vocab loses too much precision in
        # the tail and breaks the top-K ranking.
        logits = outputs.logits[0].float()  # (N, V)
        logprobs_all = torch.log_softmax(logits, dim=-1)  # (N, V)

        ids = input_ids[0].tolist()
        # Decode each id individually so the returned strings line up
        # 1:1 with the token positions. ``batch_decode`` over a list of
        # single-element lists keeps the alignment but does the decode
        # work in one tokenizer call — measurably faster than N Python
        # round-trips for long inputs.
        tokens = self.tokenizer.batch_decode([[tid] for tid in ids])

        text_offset: List[int] = []
        acc = 0
        for tok in tokens:
            text_offset.append(acc)
            acc += len(tok)

        token_logprobs: list = [None]
        top_logprobs: list = [None]
        token_entropies: list = [None]

        if n > 1:
            # Position i (i >= 1) is predicted by logits at position i-1.
            target_ids = input_ids[0, 1:]  # (N-1,)
            pred = logprobs_all[:-1]  # (N-1, V)
            actual_lp = pred.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            # Shannon entropy (nats) of the full vocabulary distribution
            # at each predicting position. Reuses the log_softmax we
            # already have: H = -sum(p * log_p) = -sum(exp(lp) * lp).
            # Sliced to the predicting positions (0..N-2) — the final
            # position would predict a token past the input, which we
            # don't score.
            entropies = -(pred.exp() * pred).sum(dim=-1)  # (N-1,)

            k = min(int(top_k), pred.shape[-1])
            top_vals, top_idx = torch.topk(pred, k=k, dim=-1)

            actual_lp_list = actual_lp.tolist()
            entropies_list = entropies.tolist()
            top_vals_list = top_vals.tolist()
            top_idx_list = top_idx.tolist()

            # Batch-decode every top-K id in one tokenizer call instead
            # of (N-1)*k individual decodes — for a 2k-token input with
            # k=10 that's 20k Python round-trips vs. one C-side batch.
            flat_top_ids = [tid for row in top_idx_list for tid in row]
            flat_top_strs = self.tokenizer.batch_decode(
                [[tid] for tid in flat_top_ids]
            )

            for i in range(n - 1):
                token_logprobs.append(actual_lp_list[i])
                token_entropies.append(entropies_list[i])
                top_dict: dict = {}
                row_off = i * k
                for j in range(k):
                    tok_str = flat_top_strs[row_off + j]
                    # Distinct vocab ids can decode to the same string
                    # (e.g. byte-fallback aliases). Keep the higher of
                    # the colliding logprobs so the rendered entry
                    # reflects the most-probable instance. ``topk``
                    # already returns descending order so the first
                    # write for a given string is the maximum — but
                    # check anyway in case a future call passes
                    # unsorted top-K.
                    existing = top_dict.get(tok_str)
                    if existing is None or top_vals_list[i][j] > existing:
                        top_dict[tok_str] = top_vals_list[i][j]
                top_logprobs.append(top_dict)

        return {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "text_offset": text_offset,
            "token_entropies": token_entropies,
        }

    def tokenize(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Tokenize a raw string with the loaded tokenizer.

        Helper for the ``/tokenize`` endpoint. Returns a flat list of
        token IDs without padding/truncation; the caller is responsible
        for length-checking against ``max_model_len`` if needed.
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=None,
            padding=False,
            truncation=False,
        )
        ids = encoded["input_ids"]
        # ``return_tensors=None`` gives a plain list (or list-of-list
        # for batched input). We always pass a single string.
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    def get_max_model_len(self) -> int:
        """Best-effort max sequence length for /tokenize responses.

        ``tokenizer.model_max_length`` defaults to a sentinel
        (``int(1e30)``) when the tokenizer doesn't know — fall back to
        the model config's ``max_position_embeddings`` in that case,
        and a conservative 2048 if neither source has a real value.
        """
        max_len = getattr(self.tokenizer, "model_max_length", 0) or 0
        if max_len > 10**9 or max_len <= 0:
            cfg = getattr(self.model, "config", None)
            max_len = getattr(cfg, "max_position_embeddings", 0) or 2048
        return int(max_len)

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
