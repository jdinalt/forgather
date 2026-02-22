# Model Architecture: modelsrc/transformer

The `modelsrc/transformer/` directory contains reusable, composable modules for building transformer-based causal language models. These modules are the building blocks from which Forgather's code generator assembles complete, self-contained model implementations.

## How Models Are Assembled

Forgather models are not imported at runtime. Instead, the code generator copies module source files into a model's `output_models/` directory alongside a main model file derived from `modelsrc/templates/hf_causal.py`. The result is a standalone HuggingFace-compatible model that can be loaded with `AutoModelForCausalLM`.

```
output_models/my_model/
├── modeling_my_model.py     # Main file (from hf_causal.py template)
├── causal_lm.py             # Copied from modelsrc/transformer/
├── glu_feedforward.py       # Copied from modelsrc/transformer/
├── rotary_embeddings.py     # Copied from modelsrc/transformer/
├── pre_ln_layer.py          # ...
└── config.json              # HuggingFace config
```

**Import constraint**: The main model file can import from other files in the same directory. However, those imported files cannot themselves use local imports (a HuggingFace limitation -- only one level of local imports is resolved when copying model files). Each module in `modelsrc/transformer/` is therefore self-contained, depending only on PyTorch, HuggingFace Transformers, and optionally pip-installed packages like `triton` or `liger-kernel`.

## Module Inventory

### Top-Level Model

| File | Class | Description |
|------|-------|-------------|
| `causal_lm.py` | `CasualLM` | Core model: chains InputEncoder -> LayerStack -> output. HF-compatible forward with KV cache support. |
| `causal_loss.py` | `CausalLoss` | Standard next-token cross-entropy loss with label shifting. |

`CasualLM` is an orchestrator. It accepts an `input_encoder`, `layer_stack`, `init_weights` function, and `attn_mask_fn`, all passed as callables/factories from the configuration system. Its forward pass:

1. Converts `input_ids` to embeddings via `InputEncoder`
2. Creates the attention mask via `attn_mask_fn`
3. Passes hidden states through the `LayerStack`
4. Returns `BaseModelOutputWithPast` (hidden states + optional KV cache)

The HF template (`modelsrc/templates/hf_causal.py`) wraps `CasualLM` inside a `PreTrainedModel` subclass that adds the language model head (`lm_head`), loss computation, generation support, and vLLM pipeline/tensor parallelism plans.

### Input Stage

| File | Class | Description |
|------|-------|-------------|
| `input_encoder.py` | `InputEncoder` | Embedding lookup + optional scaling (sqrt(d_model)) + positional encoding + dropout. |

`InputEncoder` delegates positional encoding to a pluggable `positional_encoder` module (any of the PE classes below, or None).

### Positional Encodings

| File | Class | Interface | Description |
|------|-------|-----------|-------------|
| `rotary_embeddings.py` | `RotaryPE` | `(q, k, position_ids) -> (q, k)` | Real-valued RoPE. Applied to Q/K in the attention module, not in InputEncoder. Supports cached and on-demand modes, Llama3 frequency scaling, Liger kernel acceleration, and fused Triton rotation. |
| `complex_rotary_embeddings.py` | `ComplexRotaryPE` | Same | Complex-valued RoPE (experimental, ~11% slower than real-valued). |
| `sinusoidal_pe.py` | `SinusoidalPE` | `(x, position_ids) -> x` | Original Transformer absolute positional encoding. Added to embeddings in InputEncoder. |
| `null_pe.py` | `NullPE` | `(x, position_ids) -> x` | Identity -- used when positional information comes from attention (e.g., ALiBi). |

**Two interfaces exist**: Absolute PEs (SinusoidalPE, NullPE) are added to embeddings in InputEncoder. Relative PEs (RotaryPE, ComplexRotaryPE) modify Q/K tensors inside the attention module.

RoPE is the dominant choice (~90% of configs). Key implementation details:
- **Cached mode** (default): Precomputes cos/sin for `max_sequence_length` positions once. Best for inference with dynamic KV cache (~135 tok/s on Llama 1B).
- **On-demand mode**: Recomputes per forward pass. Supports `torch.export()`. Optionally compiled via `torch.compile()` (~115 tok/s compiled, ~92 without).
- **`rotate_half(x)`**: Splits tensor into halves, negates, and concatenates. The fused Triton kernel eliminates this entirely.
- **`use_liger`**: Optional Liger kernel acceleration (external dependency).
- **`use_triton`**: Fused Triton rotation kernel (3.7-6x speedup over PyTorch, no external dependency beyond triton).

### Layer Stacking

| File | Class | Description |
|------|-------|-------------|
| `layer_stack.py` | `LayerStack` | Sequential stack of N identical layers from a factory. Optional post-norm. |
| `checkpoint_layer_stack.py` | `LayerStack` | Same, with activation checkpointing. Configurable `checkpoint_stride` (1=all, 2=every other, etc.). |
| `explicit_layer_stack.py` | `ExplicitLayerStack` | Accepts a list of distinct layer factories. For heterogeneous architectures. |

`LayerStack` uses `nn.ModuleDict` (not `nn.ModuleList`) keyed by string index. The `checkpoint_layer_stack` variant integrates with HuggingFace's `gradient_checkpointing_enable()`.

### Transformer Layers

All layer types share the same constructor signature: `feedforward_factory`, `attention_factory`, `norm_factory`, `dropout`, `residual_dropout`, plus `**kwargs` which are forwarded to the sub-factories (including `layer_idx`).

| File | Class | Architecture | Description |
|------|-------|-------------|-------------|
| `pre_ln_layer.py` | `PreLNLayer` | `x + attn(LN(x))`, `x + ff(LN(x))` | Pre-normalization (modern default). Better training stability at scale. |
| `post_ln_layer.py` | `PostLNLayer` | `LN(x + attn(x))`, `LN(x + ff(x))` | Post-normalization (original Transformer). |
| `deepnet.py` | `DeepnetLayer` | `LN(alpha*x + attn(x))` | DeepNet residual scaling for 100+ layer models. Includes `deepnet_alpha()` and `deepnet_beta()` helpers. |

All three support **residual dropout** (independent dropout on the residual path, per [Residual Dropout paper](https://aclanthology.org/2024.sigul-1.35.pdf)).

### Attention

| File | Class/Function | Description |
|------|---------------|-------------|
| `causal_multihead_attn.py` | `CausalMultiheadAttn` | Main attention module. Separate Q/K/V/O linear projections, GQA support, optional QK normalization (Qwen3-style), pluggable position encoder and attention backend. |
| `causal_alibi_attn.py` | `CausalAlibiAttn` | ALiBi attention. Adds position-dependent biases to attention scores instead of modifying Q/K. Supports trainable slopes and multiple backends. |
| `attention_interface.py` | (functions) | Four attention backends conforming to the [HF Attention Interface](https://huggingface.co/docs/transformers/main/attention_interface). |
| `eager_attention.py` | `eager_scaled_dot_product_attention` | Reference SDPA implementation for learning/debugging. |
| `causal_mask.py` | `causal_mask()` | Attention mask generation. Wraps HF's `create_causal_mask` / `create_sliding_window_causal_mask`. |

**`CausalMultiheadAttn` data flow:**
```
hidden_states [B, seq, d_model]
  -> query_linear -> [B, seq, num_heads, d_head]
  -> key_linear   -> [B, seq, num_kv_heads, d_head]
  -> value_linear -> [B, seq, num_kv_heads, d_head]
  -> (optional) q_norm, k_norm  (per-head LayerNorm over d_head)
  -> (optional) pos_encoder(q, k, position_ids)  (RoPE rotation)
  -> transpose to [B, heads, seq, d_head]
  -> (optional) KV cache update
  -> attn_fn(q, k, v, mask, ...)
  -> reshape to [B, seq, d_model]
  -> output_linear -> [B, seq, d_model]
```

**Attention backends** (selected via `attn_implementation` string):

| Backend | Key | Performance | Memory | Notes |
|---------|-----|-----------|--------|-------|
| Eager | `"eager"` | Baseline | O(seq^2) | Reference implementation. Supports ALiBi. |
| SDPA | `"sdpa"` | Good | O(seq^2) | PyTorch native. Uses `is_causal` flag to skip mask allocation. ALiBi support. |
| Flex Attention | `"flex_attention"` | Good | Medium | PyTorch 2.x. Native ALiBi via `score_mod`. Optional `torch.compile`. |
| Flash Attention 2 | `"flash_attention_2"` | Best | O(seq) | Requires `flash-attn` package. ALiBi via slopes. Sliding window support. |

Backends are passed to attention modules via an `attn_functions` dict and looked up by name. Falls back to the HF global registry (`ALL_ATTENTION_FUNCTIONS`) if not found in the dict.

### Feedforward

| File | Class | Architecture | Description |
|------|-------|-------------|-------------|
| `feedforward_layer.py` | `FeedforwardLayer` | `linear1 -> dropout -> activation -> linear2` | Standard two-layer FFN. Default activation: ReLU. |
| `glu_feedforward.py` | `GLUFeedforwardLayer` | `(up_proj(x) * activation(gate_proj(x))) -> dropout -> down_proj` | Gated Linear Unit variant (3 projections). Default activation: SiLU. Supports fused Triton kernels. |

`GLUFeedforwardLayer` is the dominant choice for modern models (Llama, Mistral, Qwen, etc.). The gating mechanism (`up * silu(gate)`) is the single most expensive memory-bandwidth operation in the model.

- **`use_triton=True`**: Fuses `activation(gate) * up` into a single kernel (1.67x forward speedup). Supports SiLU and GELU activations. Falls back to PyTorch for unsupported activations.
- All projections are `bias=False` (standard for modern LLMs).
- `init_prefix` attributes on projections enable regex-based initialization (see below).

### Weight Initialization

| File | Function/Class | Description |
|------|---------------|-------------|
| `init_weights.py` | `init_weights_by_regex()` | Regex-based parameter initialization. Matches against semantic `init_prefix` attributes on modules. |
| `init_weights.py` | `simple_weight_init()` | Fallback: calls `reset_parameters()`. |
| `llama_init.py` | Various | Llama-specific init strategies: `trunc_normal_magic`, `llama_std`, `hf_llama_weight_init`, etc. |

**`init_prefix` convention**: Modules set `init_prefix` on their sub-modules to enable semantic matching:
- `"attn.query"`, `"attn.key"`, `"attn.value"`, `"attn.output"` -- attention projections
- `"ff.up_proj"`, `"ff.gate_proj"`, `"ff.down_proj"` -- GLU feedforward projections
- `"ff.linear1"`, `"ff.linear2"` -- standard feedforward projections (via `init_prefix`, not parameter name)
- `"embedding"` -- input embeddings
- `"lm_head"` -- output projection (set in hf_causal.py template)

`init_weights_by_regex()` constructs a pseudo-FQN from these prefixes (e.g., `"attn.query.weight"`) and matches against a user-provided regex list. It enforces all-or-nothing semantics: if any parameter in a module matches, all must match.

## Composition Pattern

All modules use the **factory pattern**. Constructors accept `*_factory` callables rather than concrete instances. This enables the configuration system to wire components together:

```yaml
# In a Forgather config template:
layer_factory: !partial:pre_ln_layer:PreLNLayer
    feedforward_factory: !partial:glu_feedforward:GLUFeedforwardLayer
        d_feedforward: 2048
        activation_factory: !partial:torch.nn:SiLU
        use_triton: true
    attention_factory: !partial:causal_multihead_attn:CausalMultiheadAttn
        num_heads: 8
        pos_encoder: !call:rotary_embeddings:RotaryPE
            use_triton: true
    norm_factory: !partial:torch.nn:RMSNorm
```

The `layer_idx` parameter is automatically threaded through: `LayerStack` passes it to each layer factory, and layers forward it to attention (needed for KV cache indexing and vLLM).

## Performance Optimizations

The modules support several optimization strategies, enabled via constructor flags:

| Optimization | Flag | Where | Effect |
|-------------|------|-------|--------|
| Triton fused SiLU/GELU*up | `use_triton=True` | `GLUFeedforwardLayer` | 1.67x forward, 1.50x backward |
| Triton fused RoPE rotation | `use_triton=True` | `RotaryPE` | 3.7-6x forward, 4.5x backward |
| Liger RoPE kernel | `use_liger=True` | `RotaryPE` | External library acceleration |
| Flash Attention 2 | `attn_implementation="flash_attention_2"` | Attention modules | O(seq) memory, fastest |
| Flex Attention + compile | `attn_implementation="flex_attention"` | Attention modules | Good with torch.compile |
| Gradient checkpointing | `enable_checkpoint=True` | `LayerStack` (checkpoint variant) | Memory/compute tradeoff |
| Compiled RoPE | `compile_on_demand=True` | `RotaryPE` (on-demand mode) | ~15% speedup via torch.compile |

All optimizations fall back gracefully when dependencies are unavailable.

## File Reference

```
modelsrc/
├── templates/
│   └── hf_causal.py                  # HF PreTrainedModel template (Jinja2)
└── transformer/
    ├── causal_lm.py                  # CasualLM (core model)
    ├── input_encoder.py              # InputEncoder (embeddings + PE + dropout)
    │
    ├── rotary_embeddings.py          # RotaryPE (real-valued, recommended)
    ├── complex_rotary_embeddings.py  # ComplexRotaryPE (experimental)
    ├── sinusoidal_pe.py              # SinusoidalPE (absolute, original Transformer)
    ├── null_pe.py                    # NullPE (identity)
    │
    ├── layer_stack.py                # LayerStack (sequential)
    ├── checkpoint_layer_stack.py     # LayerStack (with activation checkpointing)
    ├── explicit_layer_stack.py       # ExplicitLayerStack (heterogeneous layers)
    │
    ├── pre_ln_layer.py               # PreLNLayer (modern default)
    ├── post_ln_layer.py              # PostLNLayer (original Transformer)
    ├── deepnet.py                    # DeepnetLayer (100+ layer scaling)
    │
    ├── causal_multihead_attn.py      # CausalMultiheadAttn (Q/K/V/O projections + GQA)
    ├── causal_alibi_attn.py          # CausalAlibiAttn (ALiBi positional biases)
    ├── attention_interface.py        # Attention backends (eager, SDPA, flex, flash)
    ├── eager_attention.py            # Reference SDPA implementation
    ├── causal_mask.py                # Attention mask generation
    │
    ├── feedforward_layer.py          # FeedforwardLayer (standard 2-layer FFN)
    ├── glu_feedforward.py            # GLUFeedforwardLayer (gated, 3 projections)
    │
    ├── init_weights.py               # Regex-based weight initialization
    ├── llama_init.py                 # Llama-specific initialization
    └── causal_loss.py                # CausalLoss (shifted cross-entropy)
```
