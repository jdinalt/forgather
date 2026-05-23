"""
Inference service core - model loading and infrastructure.

Supports multiple models per server. Each ``ModelEntry`` holds the per-model
state (tokenizer, weights, generation config, chat template, stop tokens,
per-model utility objects). ``InferenceService`` owns the entries dict, the
swap lock, and the server-wide defaults (device, ignore_eos, from_checkpoint
policy, the shared logger).

At any moment at most one entry is on GPU (``state == "gpu"``). Others are
either ``"cpu"`` (loaded, weights parked in CPU memory) or ``"unloaded"``
(never loaded since startup). Requests acquire the active entry via
``InferenceService.acquire(name)``, which serializes via an ``asyncio.Lock``
and performs swap-to-CPU / lazy load as needed.

Strategy classes still reach into the service via ``service.model``,
``service.tokenizer``, ``service.stop_processor`` etc. — those are now
``@property`` shims that route to the active entry, so no strategy code
changed.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

import torch
from jinja2 import BaseLoader, Environment, TemplateError
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
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

_MODULE_LOGGER = logging.getLogger("inference_server")


class ModelNotFoundError(LookupError):
    """Raised by :meth:`InferenceService._resolve_entry` when the OpenAI
    ``model`` field on a request doesn't match any registered entry on
    a multi-model server.

    Domain exception — the FastAPI route layer translates this to a
    404 in :mod:`routes`. The service itself stays transport-agnostic
    so unit tests can drive it without spinning up the framework.
    """

    def __init__(self, requested: Optional[str], available: List[str]) -> None:
        self.requested = requested
        self.available = available
        super().__init__(f"model not found: {requested!r}. Available: {available}")


def resolve_dtype(dtype_str: Optional[str]) -> torch.dtype:
    """Resolve a dtype string to ``torch.dtype`` with intelligent defaults.

    Module-level so callers (server CLI, YAML loader) can resolve dtypes
    when building model entries without instantiating the service.
    """
    if dtype_str is None:
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    requested = torch_dtype(dtype_str.lower())
    if (
        requested == torch.bfloat16
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        _MODULE_LOGGER.warning(
            "bfloat16 not supported on this GPU, falling back to float16"
        )
        return torch.float16
    return requested


@dataclass
class ModelEntry:
    """Per-model state: weights, tokenizer, derived utilities, lifecycle.

    Created by the CLI/YAML layer (server.py) and registered with the
    service. ``load()`` populates the tokenizer/model and downstream
    utility objects on demand.
    """

    name: str
    model_path: str
    dtype: torch.dtype
    attn_implementation: Optional[str] = None
    chat_template_path: Optional[str] = None
    stop_sequences: List[str] = field(default_factory=list)
    compile_args: Optional[Dict[str, Any]] = None
    cache_implementation: Optional[str] = None
    use_cache: Optional[bool] = None

    # Populated by load()
    tokenizer: Optional[PreTrainedTokenizer] = None
    model: Optional[PreTrainedModel] = None
    default_generation_config: Optional[GenerationConfig] = None
    chat_template: Optional[str] = None
    stop_token_ids: Optional[Set[int]] = None
    stop_processor: Optional[StopSequenceProcessor] = None
    finish_detector: Optional[FinishReasonDetector] = None
    tokenizer_wrapper: Optional[TokenizerWrapper] = None

    # Lifecycle: "unloaded" | "cpu" | "gpu"
    state: str = "unloaded"

    def load(
        self,
        device: str,
        from_checkpoint: Union[bool, str],
        default_chat_template_factory: Callable[[], str],
    ) -> None:
        """Load tokenizer + model + utilities, leaving the entry on ``device``.

        Sets ``state = "gpu"`` (or whatever ``device`` happens to be; the
        registry always promotes via this path or via ``to_device()``).
        """
        _MODULE_LOGGER.info(
            "[%s] loading model from %s (device=%s, from_checkpoint=%r)",
            self.name,
            self.model_path,
            device,
            from_checkpoint,
        )
        torch.set_float32_matmul_precision("high")

        self._load_tokenizer()
        self._load_model(device, from_checkpoint)
        self._load_generation_config()
        self._setup_chat_template(default_chat_template_factory)
        self._setup_stop_tokens()
        self._setup_utilities()

        self.state = "gpu" if str(device).startswith("cuda") else "cpu"
        _MODULE_LOGGER.info(
            "[%s] model loaded on device %s with dtype %s",
            self.name,
            self.model.device,
            self.dtype,
        )

    def to_device(self, device: str) -> None:
        """Move the (already-loaded) model between CPU and GPU."""
        if self.model is None:
            raise RuntimeError(f"entry {self.name!r} is not loaded")
        self.model.to(device)
        self.state = "gpu" if str(device).startswith("cuda") else "cpu"

    def _load_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self, device: str, from_checkpoint: Union[bool, str]) -> None:
        if from_checkpoint:
            if device == "auto":
                raise ValueError(
                    "Cannot use 'auto' device with checkpoint loading. "
                    "Specify an explicit device."
                )

            if isinstance(from_checkpoint, bool):
                checkpoint_path = find_latest_checkpoint(self.model_path)
                if not checkpoint_path:
                    raise ValueError(
                        f"No checkpoints found in {self.model_path}. "
                        "Provide a valid model directory."
                    )
            else:
                checkpoint_path = from_checkpoint
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")

            _MODULE_LOGGER.info(
                "[%s] loading from checkpoint: %s", self.name, checkpoint_path
            )
            model_config = AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            with (
                torch.device(device),
                default_dtype(dtype=self.dtype),
                no_init_weights(),
            ):
                self.model = AutoModelForCausalLM.from_config(
                    model_config,
                    trust_remote_code=True,
                    attn_implementation=self.attn_implementation,
                )
            load_checkpoint(checkpoint_path, self.model, device=device, strict=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=self.dtype,
                device_map=device if device != "auto" else "auto",
                attn_implementation=self.attn_implementation,
                trust_remote_code=True,
            )
            if device != "auto" and torch.cuda.is_available():
                self.model = self.model.to(device)

        self.model.eval()

        if self.compile_args is not None:
            args = dict(self.compile_args)
            if args.get("backend", "") == "tensorrt":
                try:
                    import torch_tensorrt  # noqa: F401
                except Exception:
                    _MODULE_LOGGER.warning(
                        "torch_tensorrt not available; falling back to default backend."
                    )
                    args.pop("backend")
            self.model.compile(**args)

    def _load_generation_config(self) -> None:
        try:
            self.default_generation_config = GenerationConfig.from_pretrained(
                self.model_path
            )
            _MODULE_LOGGER.info(
                "[%s] loaded generation config from %s", self.name, self.model_path
            )
        except Exception as e:
            _MODULE_LOGGER.info(
                "[%s] no generation config in model dir (%s); using model defaults",
                self.name,
                e,
            )
            built_in = getattr(self.model, "generation_config", None)
            if built_in is not None:
                self.default_generation_config = built_in
                _MODULE_LOGGER.info(
                    "[%s] using model's built-in generation config", self.name
                )
            else:
                self.default_generation_config = GenerationConfig()
                _MODULE_LOGGER.info("[%s] using default GenerationConfig", self.name)

        if (
            self.default_generation_config.temperature != 0.0
            and self.default_generation_config.top_p != 0.0
        ):
            _MODULE_LOGGER.warning(
                "[%s] both temperature (%s) and top_p (%s) are set != 1 in "
                "generation config; recommend setting only one. See "
                "https://platform.openai.com/docs/api-reference/completions/create",
                self.name,
                self.default_generation_config.temperature,
                self.default_generation_config.top_p,
            )

    def _setup_chat_template(self, default_factory: Callable[[], str]) -> None:
        if self.chat_template_path and os.path.exists(self.chat_template_path):
            with open(self.chat_template_path, "r") as f:
                self.chat_template = f.read()
            _MODULE_LOGGER.info(
                "[%s] using custom chat template from: %s",
                self.name,
                self.chat_template_path,
            )
        elif hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            self.chat_template = self.tokenizer.chat_template
            _MODULE_LOGGER.info(
                "[%s] using tokenizer's built-in chat template", self.name
            )
        else:
            self.chat_template = default_factory()
            _MODULE_LOGGER.info("[%s] using default fallback chat template", self.name)

    def _setup_stop_tokens(self) -> None:
        ids: Set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        for sequence in self.stop_sequences:
            try:
                token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
                if len(token_ids) == 1:
                    ids.add(token_ids[0])
                    _MODULE_LOGGER.info(
                        "[%s] added single-token stop sequence: %r -> %d",
                        self.name,
                        sequence,
                        token_ids[0],
                    )
                else:
                    _MODULE_LOGGER.info(
                        "[%s] added multi-token stop sequence: %r -> %s",
                        self.name,
                        sequence,
                        token_ids,
                    )
            except Exception as e:
                _MODULE_LOGGER.warning(
                    "[%s] failed to tokenize stop sequence %r: %s",
                    self.name,
                    sequence,
                    e,
                )
        self.stop_token_ids = ids
        _MODULE_LOGGER.info("[%s] stop token IDs: %s", self.name, sorted(ids))

    def _setup_utilities(self) -> None:
        self.stop_processor = StopSequenceProcessor(self.tokenizer)
        self.finish_detector = FinishReasonDetector(self.tokenizer, self.stop_token_ids)
        self.tokenizer_wrapper = TokenizerWrapper(self.tokenizer, self.model)


class InferenceService:
    """
    Multi-model inference service.

    Holds a registry of ``ModelEntry`` objects plus server-wide defaults.
    Requests resolve to an entry via the OpenAI ``model`` field; the
    ``acquire(name)`` async context manager handles lazy loading and
    CPU↔GPU swap under a single lock.

    Strategy classes touch the service through ``service.model``,
    ``service.tokenizer``, etc. — these are properties that route to the
    currently-active entry. Strategy code is unchanged.

    Example (single model, eager-load like the old behavior):

        >>> entry = ModelEntry(
        ...     name="my_model",
        ...     model_path="./my_model",
        ...     dtype=resolve_dtype("bfloat16"),
        ... )
        >>> service = InferenceService(
        ...     entries=[entry], device="cuda:0", ignore_eos=False
        ... )

    Multiple models (lazy):

        >>> a = ModelEntry(name="a", model_path="./a", dtype=torch.bfloat16)
        >>> b = ModelEntry(name="b", model_path="./b", dtype=torch.bfloat16)
        >>> service = InferenceService(entries=[a, b], device="cuda:0")
        >>> async with service.acquire("a"):
        ...     ...   # 'a' lazily loads to GPU on first use
    """

    def __init__(
        self,
        entries: List[ModelEntry],
        device: str,
        from_checkpoint: Union[bool, str] = False,
        ignore_eos: bool = False,
        keep_on_gpu: bool = False,
        eager_load: bool = False,
    ) -> None:
        if not entries:
            raise ValueError("InferenceService requires at least one ModelEntry")

        names = [e.name for e in entries]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate model names in entries: {names}")

        if len(entries) > 1:
            if isinstance(from_checkpoint, str):
                raise ValueError(
                    "Specific checkpoint path (-c <path>) is not supported "
                    "with multiple models; use -c with no path to load each "
                    "model's latest checkpoint."
                )
            if device == "auto":
                raise ValueError(
                    "device='auto' is not supported with multiple models; "
                    "specify an explicit device (e.g. cuda:0)."
                )

        self.device = device
        self.from_checkpoint = from_checkpoint
        self.ignore_eos = ignore_eos
        # ``keep_on_gpu``: never demote inactive models to CPU. They load
        # to GPU on first request and stay there. Only useful when total
        # GPU memory > sum(model sizes); the operator opts in. Default
        # off — swap-on-demand is the safer behavior for the typical
        # multi-model setup.
        self.keep_on_gpu = keep_on_gpu
        self.jinja_env = Environment(loader=BaseLoader())

        # Logger reads the current active tokenizer lazily so log decodes
        # follow whichever model is on GPU.
        self.logger = GenerationLogger(
            logging.getLogger("inference_server"),
            tokenizer_factory=lambda: self._active_or_none("tokenizer"),
        )

        self.entries: Dict[str, ModelEntry] = {e.name: e for e in entries}
        self.active: Optional[ModelEntry] = None
        # asyncio.Lock created lazily on first acquire() so the service can
        # be constructed outside a running event loop.
        self._swap_lock: Optional[asyncio.Lock] = None

        # Preserve "fail fast at startup" behavior for single-model setups
        # (which is what every existing call site does today). Multi-model
        # mode lazy-loads unless ``eager_load`` is set; eager-loading every
        # entry forces broken checkpoints / bad paths to surface at startup
        # rather than waiting for the first request to land on them.
        if len(entries) == 1 or eager_load:
            entry_list = list(self.entries.values())
            for entry in entry_list:
                # In eager-load mode every entry loads; in keep_on_gpu mode
                # they all stay on GPU. Without keep_on_gpu the last one
                # eager-loaded ends up on GPU and earlier ones get
                # demoted to CPU — same end-state the swap protocol
                # would produce after one round of requests.
                if not keep_on_gpu and self.active is not None and len(entry_list) > 1:
                    self._move_active_to_cpu()
                entry.load(
                    self.device,
                    self.from_checkpoint,
                    self.get_default_chat_template,
                )
                self.active = entry

    # ---------- Registry lifecycle ----------

    def list_entries(self) -> List[ModelEntry]:
        """Return all entries in registration order (for ``/v1/models``)."""
        return list(self.entries.values())

    @asynccontextmanager
    async def acquire(self, requested_name: Optional[str]):
        """Acquire a model by name; lazy-load or swap as needed.

        Holds the lock for the full duration of the request — preserves
        the existing one-request-at-a-time semantics and makes mid-flight
        swap impossible by construction.
        """
        entry = self._resolve_entry(requested_name)
        if self._swap_lock is None:
            self._swap_lock = asyncio.Lock()
        async with self._swap_lock:
            if entry.state == "unloaded":
                # First-time load. In swap mode, demote the previous
                # active first to free GPU memory; in keep_on_gpu mode,
                # leave it where it is and stack the new model alongside.
                if (
                    not self.keep_on_gpu
                    and self.active is not None
                    and self.active is not entry
                ):
                    self._move_active_to_cpu()
                entry.load(
                    self.device,
                    self.from_checkpoint,
                    self.get_default_chat_template,
                )
                self.active = entry
            elif self.active is not entry:
                # Re-targeting an already-loaded entry. In swap mode,
                # demote the previous active and promote the chosen one
                # if it had been demoted earlier. In keep_on_gpu mode,
                # both stay on GPU; if the chosen entry was somehow on
                # CPU (e.g. keep_on_gpu was flipped at runtime — not
                # currently exposed, but defensively handled), promote
                # it without demoting the other.
                if not self.keep_on_gpu and self.active is not None:
                    self._move_active_to_cpu()
                if entry.state != "gpu":
                    entry.to_device(self.device)
                self.active = entry
            yield entry

    def _move_active_to_cpu(self) -> None:
        if self.active is None:
            return
        if self.active.state == "gpu":
            _MODULE_LOGGER.info("[%s] moving to CPU", self.active.name)
            self.active.to_device("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _resolve_entry(self, name: Optional[str]) -> ModelEntry:
        """Pick which entry a request targets.

        Empty/missing name + single-model server → the sole entry
        (legacy permissive behavior). Empty name + multi-model server
        → raises :class:`ModelNotFoundError`. Name matches an entry →
        that entry. Otherwise raises :class:`ModelNotFoundError`.

        Raises a domain exception (not ``HTTPException``) so the
        service module stays free of web-framework imports. The route
        layer translates it to a 404.
        """
        if name and name in self.entries:
            return self.entries[name]
        if len(self.entries) == 1:
            return next(iter(self.entries.values()))
        raise ModelNotFoundError(name, list(self.entries.keys()))

    # ---------- Active-entry property shims (strategy/code touch these) ----------

    def _active_attr(self, attr: str) -> Any:
        if self.active is None:
            raise RuntimeError(
                f"No active model; cannot access service.{attr}. "
                "Wrap the call site in `async with service.acquire(name):`."
            )
        return getattr(self.active, attr)

    def _active_or_none(self, attr: str) -> Any:
        if self.active is None:
            return None
        return getattr(self.active, attr)

    @property
    def model(self) -> PreTrainedModel:
        return self._active_attr("model")

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._active_attr("tokenizer")

    @property
    def default_generation_config(self) -> Optional[GenerationConfig]:
        return self._active_attr("default_generation_config")

    @property
    def chat_template(self) -> Optional[str]:
        return self._active_attr("chat_template")

    @property
    def stop_sequences(self) -> List[str]:
        return self._active_attr("stop_sequences")

    @property
    def stop_token_ids(self) -> Set[int]:
        return self._active_attr("stop_token_ids")

    @property
    def stop_processor(self) -> StopSequenceProcessor:
        return self._active_attr("stop_processor")

    @property
    def finish_detector(self) -> FinishReasonDetector:
        return self._active_attr("finish_detector")

    @property
    def tokenizer_wrapper(self) -> TokenizerWrapper:
        return self._active_attr("tokenizer_wrapper")

    @property
    def use_cache(self) -> Optional[bool]:
        return self._active_attr("use_cache")

    @property
    def cache_implementation(self) -> Optional[str]:
        return self._active_attr("cache_implementation")

    @property
    def model_path(self) -> str:
        """Path of the currently-active model.

        Kept for legacy compatibility (``/v1/models`` listed
        ``model_path.split("/")[-1]`` before multi-model support). New
        code should iterate ``list_entries()`` instead.
        """
        return self._active_attr("model_path")

    # ---------- Generation config and per-request helpers ----------

    def _build_generation_config(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> GenerationConfig:
        """Build a GenerationConfig from request parameters + active model defaults."""
        max_tokens = getattr(request, "max_new_tokens", None) or getattr(
            request, "max_tokens", 16
        )

        if self.default_generation_config is not None:
            generation_config = GenerationConfig(
                **self.default_generation_config.to_dict()
            )
        else:
            generation_config = GenerationConfig()

        generation_config.max_new_tokens = max_tokens
        if request.temperature is not None:
            generation_config.temperature = request.temperature
        if request.top_p is not None:
            generation_config.top_p = request.top_p
        generation_config.do_sample = (
            request.temperature is None or request.temperature > 0
        )

        tokenizer = self.tokenizer
        if (
            not hasattr(generation_config, "pad_token_id")
            or generation_config.pad_token_id is None
        ):
            generation_config.pad_token_id = tokenizer.pad_token_id

        request_ignore_eos = getattr(request, "ignore_eos", None)
        ignore_eos = (
            request_ignore_eos if request_ignore_eos is not None else self.ignore_eos
        )
        if ignore_eos:
            # Setting eos_token_id to None doesn't work — HF refills it from
            # model defaults. Use -1 (impossible token ID) instead.
            generation_config.eos_token_id = -1
        else:
            if (
                not hasattr(generation_config, "eos_token_id")
                or generation_config.eos_token_id is None
            ):
                generation_config.eos_token_id = tokenizer.eos_token_id

        if (
            not hasattr(generation_config, "bos_token_id")
            or generation_config.bos_token_id is None
        ):
            generation_config.bos_token_id = tokenizer.bos_token_id

        generation_config.return_dict_in_generate = True
        generation_config.output_scores = False

        early_stopping_value = getattr(request, "early_stopping", None)
        if early_stopping_value is not None:
            generation_config.early_stopping = early_stopping_value

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

        do_sample_override = getattr(request, "do_sample", None)
        if do_sample_override is not None:
            generation_config.do_sample = do_sample_override

        if hasattr(request, "seed") and request.seed is not None:
            torch.manual_seed(request.seed)

        if self.use_cache is not None:
            generation_config.use_cache = self.use_cache
        if self.cache_implementation is not None:
            generation_config.cache_implementation = self.cache_implementation

        if generation_config.num_beams and generation_config.num_beams > 1:
            generation_config.do_sample = False
            if early_stopping_value is None:
                generation_config.early_stopping = True

        return generation_config

    def get_default_chat_template(self) -> str:
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
        """Render chat messages with the active model's template.

        ``next_role`` selects which role the model generates as. ``"user"``
        enables "impersonate": the template is rendered twice, the
        assistant role-opener is extracted, and ``assistant`` is
        substituted with ``user`` so the model generates inside an
        opened user-role span.
        """
        tokenizer = self.tokenizer
        chat_template = self.chat_template
        try:
            message_data = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]
            template = self.jinja_env.from_string(chat_template)
            role = (next_role or "assistant").lower()
            render_kwargs = dict(
                bos_token=tokenizer.bos_token,
                eos_token=tokenizer.eos_token,
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
            return self._fallback_format_messages(messages)
        except Exception as e:
            self.logger.logger.error(f"Unexpected error in template rendering: {e}")
            return self._fallback_format_messages(messages)

    def score_prompt(self, text: str, top_k: int = 10, max_length: int = 2048) -> dict:
        """Score an input string with per-token causal-LM logprobs.

        Runs a single forward pass (no ``generate()``) and returns the
        OpenAI legacy-completions ``logprobs`` structure plus a Forgather
        extension (``token_entropies``).

        ``max_length`` caps the tokenized prompt; surfaced as the webui's
        Analyze "Maximum length" field so a giant paste can be scored in
        bigger chunks without redeploying.
        """
        enc = self.tokenizer_wrapper.tokenize_and_move_to_device(
            text,
            max_length=max(1, int(max_length)),
            padding=False,
            truncation=True,
        )
        input_ids = enc["input_ids"]
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
        # log_softmax in float32 — bf16 across a 32k+ vocab loses too much
        # precision in the tail and breaks the top-K ranking.
        logits = outputs.logits[0].float()
        logprobs_all = torch.log_softmax(logits, dim=-1)

        tokenizer = self.tokenizer
        ids = input_ids[0].tolist()
        # batch_decode over [[tid]] keeps 1:1 alignment with positions but
        # does the decode in one C-side call rather than N round-trips.
        tokens = tokenizer.batch_decode([[tid] for tid in ids])

        text_offset: List[int] = []
        acc = 0
        for tok in tokens:
            text_offset.append(acc)
            acc += len(tok)

        token_logprobs: list = [None]
        top_logprobs: list = [None]
        token_entropies: list = [None]

        if n > 1:
            target_ids = input_ids[0, 1:]
            pred = logprobs_all[:-1]
            actual_lp = pred.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            entropies = -(pred.exp() * pred).sum(dim=-1)

            k = min(int(top_k), pred.shape[-1])
            top_vals, top_idx = torch.topk(pred, k=k, dim=-1)

            actual_lp_list = actual_lp.tolist()
            entropies_list = entropies.tolist()
            top_vals_list = top_vals.tolist()
            top_idx_list = top_idx.tolist()

            flat_top_ids = [tid for row in top_idx_list for tid in row]
            flat_top_strs = tokenizer.batch_decode([[tid] for tid in flat_top_ids])

            for i in range(n - 1):
                token_logprobs.append(actual_lp_list[i])
                token_entropies.append(entropies_list[i])
                top_dict: dict = {}
                row_off = i * k
                for j in range(k):
                    tok_str = flat_top_strs[row_off + j]
                    # Distinct vocab ids can decode to the same string;
                    # keep the higher of the colliding logprobs.
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
        """Tokenize a raw string with the active model's tokenizer."""
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=None,
            padding=False,
            truncation=False,
        )
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    def get_max_model_len(self) -> int:
        """Best-effort max sequence length for the active model."""
        max_len = getattr(self.tokenizer, "model_max_length", 0) or 0
        if max_len > 10**9 or max_len <= 0:
            cfg = getattr(self.model, "config", None)
            max_len = getattr(cfg, "max_position_embeddings", 0) or 2048
        return int(max_len)

    def _fallback_format_messages(self, messages: List[ChatMessage]) -> str:
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
