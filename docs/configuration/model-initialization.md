# Model Parameter Initialization

Forgather provides a flexible parameter initialization system that combines PyTorch's standard `reset_parameters()` convention with regex-based pattern matching for fine-grained control. This system is compatible with HuggingFace Transformers 5.0+ and supports custom initialization schemes across different model architectures.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [The init_prefix Pattern](#the-init_prefix-pattern)
- [Standard Conventions](#standard-conventions)
- [Configuration Guide](#configuration-guide)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

## Overview

### The Problem

When training transformer models, you often need different initialization strategies for different parameter types:
- Embeddings might use normal distribution with `std = 1/√d_model`
- Attention projections might use truncated normal with small std
- Feedforward layers might use Xavier initialization with depth-dependent scaling
- Biases are typically initialized to zero

Additionally, HuggingFace Transformers 5.0+ requires models to properly integrate with the `from_pretrained()` loading system, which uses meta device construction and flag-based initialization tracking.

### The Solution

Forgather's initialization system provides:
1. **Dependency injection** - the init function is a configurable callable (`_init_weights_fn`)
2. **Built-in implementations** - `init_weights_by_regex` (regex-based) and `simple_weight_init` (reset_parameters-based)
3. **Automatic fallback** - both implementations fall back to PyTorch's `reset_parameters()` for standard modules
4. **HF compatibility** - properly overrides `_init_weights()` to preserve initialization flags
5. **Extensible** - custom init functions can be injected via the configuration system

## How It Works

### The _init_weights_fn Injection Point

The HF template's `_init_weights(module)` delegates to `self.causal_lm._init_weights_fn(module)`.
This callable is set from the model configuration — it is not hardcoded. Any function with
the signature `(module: nn.Module) -> None` can be used.

Forgather provides two built-in implementations in `modelsrc/transformer/init_weights.py`:

- **`init_weights_by_regex`** — regex-based initialization with `init_prefix` support
  (used by Llama, Mistral, Qwen3, and most models)
- **`simple_weight_init`** — delegates to `reset_parameters()` with optional
  embedding scaling (useful for simpler models or quick prototyping)

Both follow the same contract: skip modules without local state, fall back to
`reset_parameters()`, and raise on modules that can't be initialized. Custom
init functions can follow the same pattern.

### init_weights_by_regex Strategy

When `init_weights_by_regex(module)` is called for each module:

1. **Skip empty modules** - Modules without parameters or buffers are skipped
2. **Regex-based override** (if `init_prefix` is set):
   - Construct pseudo-FQN: `init_prefix + '.' + param_name`
   - Search regex patterns in order, apply first match
   - **All-or-nothing validation**: If ANY parameter matches, ALL must match (prevents partial initialization)
   - Skip `reset_parameters()` if successful
3. **Fallback to reset_parameters()** - Call PyTorch's standard initialization
4. **Error** - Raise exception if module has parameters but no initialization method

### Example Flow

```python
# During model construction, modules are tagged:
self.query_linear = nn.Linear(d_model, d_model)
setattr(self.query_linear, "init_prefix", "attn.query")

# Later, during initialization:
# 1. _init_weights(query_linear) is called
# 2. Finds init_prefix = "attn.query"
# 3. Constructs name "attn.query.weight" from query_linear.weight
# 4. Matches against regex pattern 'attn.query.weight'
# 5. Applies custom initialization function
```

## The init_prefix Pattern

### What is init_prefix?

`init_prefix` is a string attribute attached to modules that serves as a semantic identifier for regex matching. It creates pseudo-FQNs (Fully Qualified Names) without requiring actual tree traversal.

```python
# In module constructor
self.up_proj = nn.Linear(d_model, d_feedforward, bias=False)
setattr(self.up_proj, "init_prefix", "ff.up_proj")

# During initialization, this becomes:
# Parameter: up_proj.weight
# Pseudo-FQN: "ff.up_proj.weight"  (init_prefix + '.' + param_name)
```

### Why Use Semantic Names?

Semantic names (like `attn.query`, `ff.up_proj`) are **implementation-independent**:
- Same regex patterns work across different model architectures
- Easy to understand: `attn.query` clearly means "attention query projection"
- Decoupled from Python variable names: renaming `query_linear` to `q_proj` doesn't break initialization
- Consistent across codebase: all attention modules use the same conventions

## Standard Conventions

Forgather uses consistent `init_prefix` values across all models:

### Attention Modules
```python
setattr(self.query_linear, "init_prefix", "attn.query")
setattr(self.key_linear, "init_prefix", "attn.key")
setattr(self.value_linear, "init_prefix", "attn.value")
setattr(self.output_linear, "init_prefix", "attn.output")
```

### Feedforward Modules

**GLU Variants (Llama, Mistral, etc.)**:
```python
setattr(self.up_proj, "init_prefix", "ff.up_proj")
setattr(self.gate_proj, "init_prefix", "ff.gate_proj")
setattr(self.down_proj, "init_prefix", "ff.down_proj")
```

**Standard Feedforward**:
```python
setattr(self.linear1, "init_prefix", "ff.linear1")
setattr(self.linear2, "init_prefix", "ff.linear2")
```

### Embeddings
```python
setattr(self.embedding, "init_prefix", "embedding")
setattr(self.lm_head, "init_prefix", "lm_head")
```

## Configuration Guide

### Basic Template Structure

In your model configuration YAML, define initialization in the `[init_weights]` section:

```yaml
[init_weights]
    [init_regex_list]
.define: &init_regex_list !dlist
    # Each entry is [regex_pattern, init_function]
    zeros:
        - 'bias'
        - !partial:torch.nn.init:zeros_

    trunc_normal:
        - 'ff.up_proj.weight|ff.gate_proj.weight'
        - !partial:.llama_init:trunc_normal_magic

    embedding:
        - 'embedding.weight'
        - !partial:.init_weights:init_embeddings
            padding_index: !var "pad_token_id"

    [init_function]
.define: &init_weights !partial:.init_weights:init_weights_by_regex@init_weights
    regex_list: *init_regex_list
    debug: False  # Set to True to see which init function is applied to each parameter
```

### Regex Pattern Syntax

**Important**: Dots are used **unescaped** in patterns for readability:

```yaml
# ✅ Recommended - readable and clear
- 'attn.query.weight'
- 'ff.up_proj.weight|ff.gate_proj.weight'

# ❌ Not recommended - harder to read, error-prone
- 'attn\.query\.weight'
- 'ff\.up_proj\.weight|ff\.gate_proj\.weight'
```

Since `init_prefix` values are controlled and use dots as hierarchical separators, the risk of false matches is negligible. The unescaped dot makes patterns visually match the semantic structure.

### Pattern Matching Rules

1. **First match wins** - Patterns are tested in order
2. **Full regex syntax** - Use `|` for OR, `.*` for wildcards, etc.
3. **No prefix needed for universal patterns** - `'bias'` matches all biases
4. **All-or-nothing** - If any parameter matches, all must match (prevents partial init)

## Examples

### Example 1: Llama-Style Initialization

```yaml
[init_regex_list]
.define: &init_regex_list !dlist
    zeros:
        - 'bias'
        - !partial:torch.nn.init:zeros_

    # Attention and up projection: small std
    trunc_normal_magic:
        - 'ff.up_proj.weight|attn.query.weight|attn.key.weight|attn.value.weight'
        - !partial:.llama_init:trunc_normal_magic  # std=0.02

    # Gate/down projections and output: depth-scaled std
    trunc_normal:
        - 'ff.gate_proj.weight|ff.down_proj.weight|attn.output.weight'
        - !partial:.llama_init:trunc_normal
            std: !call:.llama_init:llama_std [ !var "num_hidden_layers" ]

    # LM head: inverse sqrt scaling
    lm_head:
        - 'lm_head.weight'
        - !partial:.llama_init:init_output_layer
            d_model: !var "hidden_size"
```

### Example 2: Custom Initialization

```yaml
[init_regex_list]
.define: &init_regex_list !dlist
    # Standard bias initialization
    zeros:
        - 'bias'
        - !partial:torch.nn.init:zeros_

    # Embeddings with custom std
    embedding:
        - 'embedding.weight'
        - !partial:.init_weights:init_embeddings
            padding_index: !var "pad_token_id"
            scale_rsqrt_d_model: True

    # All attention weights: Xavier uniform
    attention:
        - 'attn.*.weight'
        - !partial:torch.nn.init:xavier_uniform_
            gain: 1.0

    # Feedforward with custom logic
    feedforward:
        - 'ff.*.weight'
        - !partial:my_custom_init:special_init
            alpha: 0.1
```

### Example 3: Deepnet Initialization

[DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)

```yaml
[init_regex_list]
.define: &init_regex_list !dlist
    # Standard bias initialization
    zeros: 
        - 'bias'
        - !partial:torch.nn.init:zeros_

    # Layers using DeepNet initialization, with `deepnet_beta` computed for n_layers
    deepnet:
        - '^ff\.|attn.value.weight|attn.output.weight'
        - !partial:torch.nn.init:xavier_uniform_
            gain: !call:.deepnet:deepnet_beta [ !var "num_hidden_layers", 0 ]
    
    # Init remaining attention layers with xavier_uniform_
    linear:
        - 'attn.key.weight|attn.query.weight'
        - !partial:torch.nn.init:xavier_uniform_ [ gain: 1.0 ]
    
[layer_factory]
# For completeness, replace layers with DeepNorm layers
.define: &layer_factory !partial:.deepnet:DeepnetLayer@layer_factory
    feedforward_factory: *feedforward_factory
    attention_factory: *attention_factory
    norm_factory: *layer_norm_factory
    alpha: !call:.deepnet:deepnet_alpha [ !var "num_hidden_layers", 0 ]

[layer_stack]
# DeepNorm layers are a from of 'post-norm.' Remove redundant final norm layer from
# stack, if converting a pre-norm layers.
    == super()
    post_norm_factory: null
```

### Example 4: Debug Mode

Enable debug mode to see exactly which initialization function is applied to each parameter:

```yaml
[init_function]
.define: &init_weights !partial:.init_weights:init_weights_by_regex@init_weights
    regex_list: *init_regex_list
    debug: True  # Enable debug output
```

Output during initialization:
```
Init: zeros_(attn.query.bias)
Init: trunc_normal_magic(attn.query.weight)
Init: zeros_(attn.key.bias)
Init: trunc_normal_magic(attn.key.weight)
...
```

## Troubleshooting

### Error: "Not all parameters in X were initialized: [...] Check model's init config.""

**Cause**: Some parameters matched a regex pattern but others didn't (partial initialization).

**Solution**: Check your regex patterns cover all parameters with that `init_prefix`:
```yaml
# ❌ Incomplete - missing value projection
- 'attn.query.weight|attn.key.weight'

# ✅ Complete - covers all attention projections
- 'attn.query.weight|attn.key.weight|attn.value.weight|attn.output.weight'
```

Note that the diagnostic message will list the uninitialized layers.

### Error: "Module of type 'X' has parameters, but lacks a 'reset_parameters()' method"

**Cause**: Module has parameters but:
1. No `init_prefix` set (so regex matching skipped)
2. No `reset_parameters()` method (so fallback failed)

**Solution**: Either add `init_prefix` to the module (and a matching regex) or implement `reset_parameters()`:
```python
class CustomModule(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
```

## Technical Details

### HuggingFace Integration

The initialization system integrates with HuggingFace Transformers v5's
`PreTrainedModel` weight initialization pipeline. Understanding this call chain
is important for debugging initialization issues.

#### The Call Chain

When a model is constructed (either directly or via `from_pretrained`), HF's
`post_init()` triggers weight initialization:

```
DynamicCasualLM.__init__(config)
  |
  +-- self.causal_lm = CasualLM(...)     # Build model (all tensors on meta or target device)
  |
  +-- self.post_init()                    # HF PreTrainedModel hook
        |
        +-- self.init_weights()           # PreTrainedModel method
              |
              +-- self.initialize_weights()
                    |
                    +-- self.smart_apply(self._initialize_weights)
                          |
                          +-- for each module in model.modules():
                                |
                                +-- self._initialize_weights(module)
                                      |
                                      +-- check _is_hf_initialized flag
                                      |   (skip if already initialized from checkpoint)
                                      |
                                      +-- self._init_weights(module)      # OUR OVERRIDE
                                            |
                                            +-- self.causal_lm._init_weights_fn(module)
                                                  |
                                                  +-- (injected init function)
                                                  |   e.g. init_weights_by_regex
                                                  |   or   simple_weight_init
                                                  |   or   custom function
                                                  |
                                                  +-- has_local_state(module)?
                                                  |   (skip if no params or buffers)
                                                  |
                                                  +-- (implementation-specific logic)
                                                  |
                                                  +-- fallback: module.reset_parameters()
```

Key points:

- **`_init_weights(module)`** is HF's per-module hook. Its signature is
  `(self, module)` where `self` is the top-level `PreTrainedModel` and `module`
  is the specific sub-module being initialized. Our template overrides this to
  delegate to the injected `_init_weights_fn`.

- **`_init_weights_fn`** is a callable stored on `CasualLM`, set from the
  configuration. This is the dependency injection point -- different models
  can use different initialization strategies. It is typically a ``partial``
  with pre-bound arguments (e.g., ``partial(init_weights_by_regex,
  regex_list=...)``).

- **Built-in implementations** (``init_weights_by_regex``,
  ``simple_weight_init``) both follow the same contract: skip modules without
  local state, apply implementation-specific logic, fall back to
  ``reset_parameters()``. Custom init functions should follow the same pattern.

- **`_is_hf_initialized`** is a flag set by HF on parameters loaded from a
  checkpoint. The `_initialize_weights` wrapper checks this flag and skips
  modules that were already loaded. The `torch.nn.init.*` functions are also
  patched to respect this flag as an additional safety net.

#### Buffer-Only Modules (e.g., RotaryEmbedding)

Modules with only non-persistent buffers (no parameters) still participate
in initialization. `has_local_state()` checks both parameters and buffers,
so `RotaryEmbedding` (which has an `inv_freq` buffer) is not skipped.

Since `RotaryEmbedding` has no `init_prefix` and no parameters for regex
matching, the init function falls through to `reset_parameters()`.
This is where the buffer values are computed:

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, ...):
        # Allocate empty buffer -- NOT computed here
        self.register_buffer("inv_freq", torch.empty(d_head // 2), persistent=False)

    def reset_parameters(self):
        # Compute actual values -- called by init system
        if self.inv_freq.is_meta:
            return  # On meta device, wait for real device
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2, ...) / d_head))
        self.inv_freq.copy_(inv_freq)
```

The `persistent=False` flag means `inv_freq` is excluded from `state_dict()`,
so it is never saved to or loaded from checkpoints. It is always recomputed
by `reset_parameters()`.

#### Meta Device Construction (Pipeline Parallel, from_pretrained)

When a model is constructed on the meta device (for pipeline parallel or
`from_pretrained`), tensors have shape and dtype but no data:

```
1. with torch.device("meta"):
       model = DynamicCasualLM(config)
   # All tensors (params and buffers) are on meta device
   # post_init() calls _init_weights for each module
   # reset_parameters() sees inv_freq.is_meta -> returns early (no-op)

2. Load state_dict (for from_pretrained)
   # Persistent parameters are loaded from checkpoint
   # Non-persistent buffers (inv_freq) remain on meta

3. HF calls _move_missing_keys_from_meta_to_device()
   # Replaces meta tensors with torch.empty_like(..., device=target_device)
   # inv_freq is now an empty tensor on the real device

4. HF calls _initialize_missing_keys() -> init_weights() for affected modules
   # _init_weights(rotary_emb) is called again
   # reset_parameters() now runs: inv_freq.is_meta is False
   # Computes and fills inv_freq on the real device
```

This two-phase initialization (allocate in `__init__`, compute in
`reset_parameters`) is what allows the same code to work for both direct
construction and meta-device construction.

### Initialization Order

During model construction:
1. Model is constructed (on meta device for `from_pretrained`, or target device directly)
2. `post_init()` triggers `_init_weights(module)` for each module
3. For each module, the injected init function (e.g., `init_weights_by_regex`) runs:
   - Skip if no local state (no parameters or buffers)
   - Apply implementation-specific logic (e.g., regex matching via `init_prefix`)
   - Fall back to `reset_parameters()`
4. If loading from checkpoint: `_is_hf_initialized` flags prevent re-initialization of loaded parameters
5. For meta device: steps 3-4 happen again after buffers are moved to real device

### Module Construction Patterns

**Modules with parameters** (e.g., attention, feedforward): Tag sub-modules with
`init_prefix` for regex-based initialization. PyTorch's built-in modules
(`nn.Linear`, `nn.Embedding`, etc.) already have `reset_parameters()` as a
fallback, so no override is needed for standard cases.

```python
class MyAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        # ...

        # Tag with semantic init_prefix for regex matching
        setattr(self.query_linear, "init_prefix", "attn.query")
        setattr(self.key_linear, "init_prefix", "attn.key")
        # ...
```

**Modules with only buffers** (e.g., RotaryEmbedding): Allocate empty buffers
in `__init__` and compute values in `reset_parameters()`. This two-phase
pattern supports meta device construction.

```python
class MyModule(nn.Module):
    def __init__(self, dim, device=None):
        super().__init__()
        self.dim = dim
        # Allocate but don't compute -- reset_parameters() handles that
        self.register_buffer("my_buffer", torch.empty(dim, device=device), persistent=False)

    def reset_parameters(self):
        if self.my_buffer.is_meta:
            return  # Meta device: wait for real device
        values = self._compute_buffer()
        self.my_buffer.copy_(values)
```

The `persistent=False` flag excludes the buffer from `state_dict()`, so it is
always recomputed (never loaded from checkpoints). The `is_meta` guard allows
`reset_parameters()` to be safely called during meta-device construction,
deferring actual computation until the buffer has been moved to a real device.

### Performance Considerations

- Regex matching is performed once during model initialization (not during training)
- The `has_local_state()` check is fast (just counts parameters/buffers)
- Debug mode adds minimal overhead (just prints)

## See Also

- [Syntax Reference](syntax-reference.md) - Full YAML configuration syntax
- [Writing a Config](writing-a-config.md) - General configuration guide
- [Template Inheritance](inheritance.md) - How to extend base templates
- `modelsrc/transformer/init_weights.py` - Built-in init functions (`init_weights_by_regex`, `simple_weight_init`)
- `modelsrc/transformer/llama_init.py` - Llama-specific initialization functions
- `templatelib/examples/models/transformers/dynamic_llama.yaml` - Complete example
