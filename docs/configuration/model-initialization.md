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
1. **Semantic naming** via `init_prefix` - modules are tagged with implementation-independent names
2. **Regex-based overrides** - match parameter patterns and apply custom initialization
3. **Automatic fallback** - uses PyTorch's `reset_parameters()` for standard modules
4. **HF compatibility** - properly overrides `_init_weights()` to preserve initialization flags

## How It Works

### Three-Tier Initialization Strategy

When `_init_weights(module)` is called for each module during model initialization:

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

Forgather models override `_init_weights(module)` in the `DynamicCasualLM` class:

```python
@override
def _init_weights(self, module: torch.nn.Module):
    self.causal_lm.init_weights(module)
```

This preserves the `@init.guard_torch_init_functions()` decorator from HuggingFace's base class, which:
1. Patches `torch.nn.init.*` functions to check `_is_hf_initialized` flags
2. Prevents re-initialization of parameters loaded from checkpoints
3. Enables proper meta device construction in `from_pretrained()`

### Initialization Order

During model construction:
1. Model is constructed on meta device (transformers 5.0+)
2. If loading checkpoint: weights are loaded, `param._is_hf_initialized = True` is set
3. `_init_weights(module)` is called for each module, where `module._is_hf_initialized == False`
4. For each module:
   - Check if has `init_prefix`
   - If yes: try regex matching
   - If no/no matches: call `reset_parameters()`
   - Patched `torch.nn.init.*` functions check flags and skip already-initialized params

### Module Construction Pattern

When creating custom modules, follow this pattern:

```python
class MyAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        # Create linear layers
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        # Tag with semantic init_prefix
        setattr(self.query_linear, "init_prefix", "attn.query")
        setattr(self.key_linear, "init_prefix", "attn.key")
        setattr(self.value_linear, "init_prefix", "attn.value")
        setattr(self.output_linear, "init_prefix", "attn.output")

    # No need to override reset_parameters()
    # PyTorch's Linear already has reset_parameters() as fallback
```

### Performance Considerations

- Regex matching is performed once during model initialization (not during training)
- The `has_local_state()` check is fast (just counts parameters/buffers)
- Debug mode adds minimal overhead (just prints)

## See Also

- [Syntax Reference](syntax-reference.md) - Full YAML configuration syntax
- [Writing a Config](writing-a-config.md) - General configuration guide
- [Template Inheritance](inheritance.md) - How to extend base templates
- `modelsrc/transformer/init_weights.py` - Implementation source code
- `modelsrc/transformer/llama_init.py` - Llama initialization functions
- `templatelib/examples/models/transformers/dynamic_llama.yaml` - Complete example
