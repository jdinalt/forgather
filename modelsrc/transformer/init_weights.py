import math
import re
from functools import partial
from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor, nn


def has_local_state(module: nn.Module) -> bool:
    """Returns True if module has any direct parameters or buffers"""
    # Check for direct parameters
    has_params = sum(1 for _ in module.parameters(recurse=False))
    # Check for direct buffers
    has_buffers = sum(1 for _ in module.buffers(recurse=False))
    return has_params or has_buffers


def _get_callable_name(fn: Callable) -> str:
    """Extract function name from callable for debug output."""
    if isinstance(fn, partial):
        return fn.func.__name__
    return fn.__name__


@torch.no_grad()
def simple_weight_init(
    module: nn.Module,
    scale_rsqrt_d_model: bool = False,
) -> None:
    """
    Defaults to module's reset_parameters() method, excepting nn.Embedding

    nn.Embedding lacks a ways to specify the initialization std, which we need, if scaling by the rsqrt(d_model)
    """
    if not has_local_state(module):
        return

    if scale_rsqrt_d_model and isinstance(module, nn.Embedding):
        init_embeddings(
            module.weight,
            module.padding_idx,
            scale_rsqrt_d_model=True,
        )
        return

    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
        return

    raise ValueError(
        f"Module of type '{type(module)}' has parameters, but lacks a 'reset_parameters()' method"
    )


@torch.no_grad()
def init_weights_by_regex(
    module: torch.nn.Module,
    regex_list: List[Tuple[str, Callable]],
    debug: bool = False,
) -> None:
    r"""
    Initialize module parameters with optional regex-based overrides.

    This function provides a flexible initialization system that combines PyTorch's
    standard `reset_parameters()` convention with regex-based parameter selection
    for fine-grained control.

    Initialization Strategy
    -----------------------
    1. **Skip modules without state**: Modules with no parameters or buffers are skipped
    2. **Regex-based override** (optional): If module has `init_prefix` attribute:
       - Construct pseudo-FQN: `init_prefix + '.' + param_name`
       - Search regex_list in order, apply first match
       - **All-or-nothing**: If ANY parameter matches, ALL must match (prevents partial init)
       - Skip `reset_parameters()` if successful
    3. **Fallback to reset_parameters()**: Call module's `reset_parameters()` if available
    4. **Error**: Raise exception if module has parameters but no initialization method

    The `init_prefix` Pattern
    --------------------------
    Modules can set a semantic prefix to enable regex-based initialization:

        setattr(self.query_linear, "init_prefix", "attn.query")
        setattr(self.up_proj, "init_prefix", "ff.up_proj")

    This creates pseudo-FQNs like "attn.query.weight" and "ff.up_proj.weight" that
    can be matched with regex patterns, without requiring actual tree traversal.

    Standard init_prefix Conventions
    ---------------------------------
    The following prefixes are used consistently across Forgather models:

    - **Attention modules**:
        - `attn.query` - Query projection
        - `attn.key` - Key projection
        - `attn.value` - Value projection
        - `attn.output` - Output projection

    - **Feedforward modules**:
        - `ff.up_proj` - Up projection (GLU variants)
        - `ff.gate_proj` - Gate projection (GLU variants)
        - `ff.down_proj` - Down projection (GLU variants)
        - `ff.linear1` - First linear layer (standard FFN)
        - `ff.linear2` - Second linear layer (standard FFN)

    - **Embeddings**:
        - `embedding` - Input token embeddings
        - `lm_head` - Output language model head

    These semantic names are implementation-independent, allowing the same regex
    patterns to work across different model architectures.

    Regex Pattern Conventions
    --------------------------
    **Dots are used unescaped** in regex patterns (e.g., `'attn.query.weight'` not
    `'attn\.query\.weight'`) for readability and error resistance. Since init_prefix
    values are controlled and use dots as hierarchical separators, the risk of false
    matches is negligible. The unescaped dot makes patterns visually match the semantic
    structure of parameter names and avoids the error-prone `'.\` typo.

    Examples:
        - `'attn.query.weight'` - Matches query projection weights
        - `'ff.up_proj.weight|ff.gate_proj.weight'` - Matches multiple FFN weights
        - `'bias'` - Matches all bias parameters (no prefix needed)

    Args:
        module: The module to initialize (typically called per-module by _init_weights)
        regex_list: List of (pattern, init_function) tuples
            - pattern: Regular expression to match against pseudo-FQN
            - init_function: Callable taking a parameter tensor (e.g., torch.nn.init.*)
            Patterns are tested in order; first match wins.
        debug: If True, print which initialization function is applied to each parameter

    Raises:
        ValueError: If partial initialization detected (some params matched, others didn't)
        ValueError: If module has parameters but no `reset_parameters()` and no regex matches

    Example:
        from functools import partial

        regex_list = [
            (r"bias", torch.nn.init.zeros_),
            (r"embedding.weight", partial(init_embeddings, padding_idx=0)),
            (r"ff.up_proj.weight|ff.down_proj.weight",
             partial(torch.nn.init.xavier_normal_, gain=1.0)),
            (r"attn..*weight", partial(torch.nn.init.trunc_normal_, std=0.02)),
        ]

        # Apply to entire model (called by _init_weights for each module)
        for module in model.modules():
            init_weights_by_regex(module, regex_list, debug=True)

    Note:
        Most PyTorch modules (Linear, Embedding, LayerNorm, etc.) have `reset_parameters()`
        and will initialize correctly without regex overrides. Use regex patterns only
        when you need custom initialization beyond PyTorch defaults.
    """
    if not has_local_state(module):
        return

    # Is the module tagged with an init_prefix?
    init_prefix = getattr(module, "init_prefix", None)
    if init_prefix is not None:
        n_params = 0
        uninitialized_params = []
        for name, param in module.named_parameters(
            init_prefix, recurse=False, remove_duplicate=True
        ):
            n_params += 1
            for regex, init_fn in regex_list:
                m = re.search(regex, name)
                if m is not None:
                    if debug:
                        print(f"Init: {_get_callable_name(init_fn)}({name})")
                    init_fn(param)
                    break
            else:
                # Keep track of unmatched parameter names
                uninitialized_params.append(name)
        # If any parameters were initialized, ensure that all parameters were initialized
        n_init_params = n_params - len(uninitialized_params)
        if n_init_params:
            if n_init_params != n_params:
                raise ValueError(
                    f"Not all parameters in {init_prefix} were initialized: {uninitialized_params} Check model's init config."
                )
            else:
                return
        # Try next init method

    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
        return

    raise ValueError(
        f"Module of type '{type(module)}' has parameters, but lacks a 'reset_parameters()' method"
    )


def init_torch_linear_default(weight: Tensor, gain: float = 1.0):
    """
    This is the torch default for nn.Linear()
    This appears to be a He init variant, which always uses
    fan-in and does not try to compensate for the activation.

    You can use the "gain" parameter for compensating for the
    activation.

    Note: torch has a rather unusual implementation. See:
    https://github.com/pytorch/pytorch/issues/57109
    https://soumith.ch/files/20141213_gplus_nninit_discussion.htm
    """
    in_features = weight.shape[1]
    uniform_range = gain / math.sqrt(in_features)
    torch.nn.init.uniform_(weight, -uniform_range, uniform_range)


@torch.no_grad()
def init_embeddings(
    weight: Tensor,
    padding_index: int = None,
    std: float = 1.0,
    scale_rsqrt_d_model: bool = True,
):
    """
    Simple embedding init.

    Zeros out the pad embedding.
    If scale_rsqrt_d_model == True, then std = 1 / sqrt(d_model), as per
        per Attention is All you Need.
    Note: When scale_rsqrt_d_model == True, the input encoder should scale the embedding outputs
        by sqrt(d_model).
    """
    if scale_rsqrt_d_model:
        std = 1.0 / math.sqrt(weight.shape[1])
    nn.init.normal_(weight, std=std)

    # If pad index, zero that embedding.
    if padding_index is not None:
        nn.init.zeros_(weight[padding_index])
