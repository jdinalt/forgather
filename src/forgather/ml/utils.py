import types
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pprint import pformat
from typing import Callable

import torch


def alt_repr(obj):
    """
    Alternative __repr__ implementation for objects which have
    not implemented it for their class.

    Some object are relatively opaque. This should expose more info.
    """
    # If __repr__ is a wrapper, they have not implemented it.
    if isinstance(obj.__repr__, types.MethodWrapperType):
        attrs = {}
        for key in dir(obj):
            # Ignore protected, private, etc.
            if key.startswith("_"):
                continue
            value = getattr(obj, key)

            # Ignore methods and other callables.
            if isinstance(value, Callable):
                continue

            attrs[key] = value
        return pformat(attrs)
    else:
        return repr(obj)


class ConversionDescriptor:
    """
    A descriptor for automatically converting types.

    Example:
    ```
    class Color(Enum):
        RED = "red"
        BLUE = "blue"

    @dataclass
    class Data:
        color: ConversionDescriptor = ConversionDescriptor(Color, default=Color.RED)

    data = Data(color="blue")
    print(data)
    > Data(color=<Color.BLUE: 'blue'>)
    ```
    """

    def __init__(self, cls, *, default):
        self._cls = cls
        self._default = default

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        if obj is None:
            return self._default
        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value):
        setattr(obj, self._name, self._cls(value))


class DiagnosticEnum(Enum):
    """
    An extension to Enum which provides better diagnostic info when an invalid value is set.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}; choose one of {cls._value2member_map_.keys()}"
        )


def fmt_si(v: int) -> str:
    """Format an integer with SI suffix (K, M, G)."""
    v = int(v)
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.3g}G"
    elif v >= 1_000_000:
        return f"{v / 1_000_000:.3g}M"
    elif v >= 1_000:
        return f"{v / 1_000:.3g}K"
    return str(v)


@dataclass(frozen=True, slots=True)
class ModelParameterStats:
    """Structured parameter statistics for a PyTorch model.

    All counts are raw integers (number of scalar parameters).
    """

    total: int
    trainable: int
    embedding: int
    non_embedding: int
    trainable_non_embedding: int
    tied_embeddings: bool

    @property
    def chinchilla_optimal_tokens(self) -> int:
        """Chinchilla-optimal token count: ~20 tokens per non-embedding parameter.

        Reference: Hoffmann et al. "Training Compute-Optimal Large Language Models" (2022).
        """
        return 20 * self.non_embedding

    @property
    def flops_per_token(self) -> float:
        """Estimated FLOPs per token for forward + backward pass (C = 6N).

        N = non-embedding parameters.  Counts each multiply-accumulate as 2 FLOPs,
        consistent with hardware FLOP specs (NVIDIA TFLOP/s).
        Forward = 2N, backward ~= 4N, total = 6N.

        References:
            Kaplan et al. "Scaling Laws for Neural Language Models" (2020).
            Hoffmann et al. "Training Compute-Optimal Large Language Models" (2022).
        """
        return 6.0 * self.non_embedding


def _safe_get_module(model, method_name):
    """Safely call a model method (e.g. get_input_embeddings), returning None on failure."""
    getter = getattr(model, method_name, None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None


def count_parameters(model) -> ModelParameterStats:
    """Compute structured parameter statistics for a PyTorch model.

    Uses get_input_embeddings()/get_output_embeddings() (HuggingFace PreTrainedModel
    interface) to identify embedding parameters.  Models without these methods
    report embedding=0.

    PyTorch's model.parameters() deduplicates shared parameters by tensor id,
    so tied embedding weights are counted once in total/trainable.
    """
    # Collect embedding parameter ids (deduplicated via set)
    embedding_param_ids: set[int] = set()
    for method_name in ("get_input_embeddings", "get_output_embeddings"):
        module = _safe_get_module(model, method_name)
        if module is not None:
            for p in module.parameters():
                embedding_param_ids.add(id(p))

    # Detect tied embeddings
    tied = False
    inp = _safe_get_module(model, "get_input_embeddings")
    out = _safe_get_module(model, "get_output_embeddings")
    if inp is not None and out is not None:
        iw = getattr(inp, "weight", None)
        ow = getattr(out, "weight", None)
        if iw is not None and ow is not None:
            tied = iw is ow

    # Single pass over parameters
    total = trainable = embedding = trainable_emb = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        is_emb = id(p) in embedding_param_ids
        if p.requires_grad:
            trainable += n
            if is_emb:
                trainable_emb += n
        if is_emb:
            embedding += n

    return ModelParameterStats(
        total=total,
        trainable=trainable,
        embedding=embedding,
        non_embedding=total - embedding,
        trainable_non_embedding=trainable - trainable_emb,
        tied_embeddings=tied,
    )


@contextmanager
def default_dtype(dtype: torch.dtype):
    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(prev_dtype)
