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

### Schema provenance in `config.json`

Newly generated `config.json` files carry two fields that identify the
architecture and the schema version of the Forgather sources at the
time of save:

| Field | Purpose |
|-------|---------|
| `forgather_arch` | Converter registry key (e.g. `"llama"`). Defaults to `ns.model_short_name` from the model template. |
| `forgather_arch_version` | PEP 440 schema version string (e.g. `"1"`, `"1.2"`, `"2.3.1"`). Defaults to `"1"`. Bump the **major** in the arch's converter when you make a non-backwards-compatible change to parameter FQNs or config fields; bump minor / patch when the change is forwards-compatible (the existing saved schema still loads under the new code). |

These are read by [`forgather update`](model-update.md) when migrating
an existing saved model to a newer source layout, so the user does not
have to remember which converter / schema version the model was built
under. They can be overridden in a project template by setting
`ns.forgather_arch` / `ns.forgather_arch_version` explicitly.

## Module Inventory

### Top-Level Model

| File | Class | Description |
|------|-------|-------------|
| `causal_lm.py` | `CasualLM` | Core model: chains InputEncoder -> LayerStack -> output. HF-compatible forward with KV cache support. |
| `causal_loss.py` | `CausalLoss` | Standard next-token cross-entropy loss with label shifting. |

`CasualLM` is an orchestrator. It accepts an `input_encoder`, `layer_stack`, `init_weights` function, `attn_mask_fn`, and an optional `rotary_emb` module, all passed as callables/factories from the configuration system. Its forward pass:

1. Converts `input_ids` to embeddings via `InputEncoder`
2. Creates the attention mask via `attn_mask_fn`
3. Computes rotary position embeddings via `rotary_emb` (if present)
4. Passes hidden states + `position_embeddings` through the `LayerStack`
5. Returns `BaseModelOutputWithPast` (hidden states + optional KV cache)

The HF template (`modelsrc/templates/hf_causal.py`) wraps `CasualLM` inside a `PreTrainedModel` subclass that adds the language model head (`lm_head`), loss computation, generation support, and vLLM pipeline/tensor parallelism plans.

### Input Stage

| File | Class | Description |
|------|-------|-------------|
| `input_encoder.py` | `InputEncoder` | Embedding lookup + optional scaling (sqrt(d_model)) + positional encoding + dropout. |

`InputEncoder` delegates positional encoding to a pluggable `positional_encoder` module (any of the PE classes below, or None).

### Positional Encodings

| File | Class | Interface | Description |
|------|-------|-----------|-------------|
| `rotary_embeddings.py` | `RotaryEmbedding` | `(x, position_ids) -> (cos, sin)` | RoPE module (`nn.Module`). Computes (cos, sin) once per forward pass in `CasualLM`. Supports Llama3 frequency scaling. |
| `rotary_embeddings.py` | `apply_rotary_pos_emb` | `(q, k, position_embeddings, **kw) -> (q, k)` | Standalone rotation function. Injected into attention as `pos_encoder`. |
| `sinusoidal_pe.py` | `SinusoidalPE` | `(x, position_ids) -> x` | Original Transformer absolute positional encoding. Added to embeddings in InputEncoder. |
| `null_pe.py` | `NullPE` | `(x, position_ids) -> x` | Identity -- used when positional information comes from attention (e.g., ALiBi). |

**Two interfaces exist**: Absolute PEs (SinusoidalPE, NullPE) are added to embeddings in InputEncoder. Relative PEs (RoPE) modify Q/K tensors inside the attention module.

RoPE is the dominant choice (~90% of configs). The architecture follows Transformers v5.x:
- **`RotaryEmbedding`** is an `nn.Module` owned by `CasualLM`. It computes `(cos, sin)` once per forward pass.
- **`position_embeddings`**: The `(cos, sin)` tuple flows through kwargs from `CasualLM` -> `LayerStack` -> `PreLNLayer` -> `CausalMultiheadAttn` -> `pos_encoder`.
- **`apply_rotary_pos_emb`**: A stateless function that extracts `position_embeddings` from kwargs and applies the rotation to Q/K.
- **Meta device support**: `inv_freq` is a non-persistent buffer (`persistent=False`), allocated empty in `__init__` and computed by `reset_parameters()`. The weight initialization system (`init_weights_by_regex`) calls `reset_parameters()` as its fallback for modules without `init_prefix`. On meta device, `reset_parameters()` is a no-op; values are computed after buffers are moved to a real device. See `docs/configuration/model-initialization.md` for the full call chain.
- **torch.compile**: No graph breaks. The `float32_output` flag is a compile-time guard (constant-folded after first trace). No Triton kernels needed -- torch.compile fuses the element-wise rotation ops automatically.

#### RoPE Numerical Precision

RoPE precision has two distinct aspects: **frequency computation** (inv_freq, cos, sin) and **rotation application** (q*cos + rotate_half(q)*sin). The requirements differ.

**Frequency computation -- float32 is mandatory.** The inverse frequencies are computed as `1 / (theta^(2i/d))`, producing values that span several orders of magnitude. Cos/sin of these frequencies at large position indices amplify small rounding errors into large absolute errors. Known incidents:

- HF Transformers #29301: autocast silently downcast the `inv_freq @ position_ids` matmul to bf16 despite explicit `.float()` calls. Fixed by wrapping with `torch.autocast(enabled=False)`.
- GPT-NeoX bug: cos/sin tables computed in fp16 produced errors of ~0.78 vs float32 at position 1857. The periodic functions turn small relative rounding into large absolute errors.
- Research has shown that bf16 frequency computation breaks RoPE's position-shift invariance property -- the mathematical guarantee that `A(i+d)(j+d) == A(i)(j)` fails under bf16 rounding.

Our implementation forces float32 for all frequency computation via `torch.autocast(enabled=False)`.

**Rotation -- float32 is recommended but optional.** The rotation `q*cos + rotate_half(q)*sin` involves element-wise multiplications. In bf16, this introduces rounding at each step. The impact depends on context length: short contexts (<4K) show negligible differences; long contexts (>8K) can show measurable degradation.

What major implementations do:

| Implementation | Frequency computation | Rotation |
|---|---|---|
| HF Transformers v5 | float32 (autocast disabled) | Model dtype |
| Flash Attention (Triton) | float32 | float32 (hardcoded) |
| torchtitan | float32 | Model dtype |
| vLLM | float32 | Model dtype |

The `RotaryEmbedding.float32_output` flag controls which approach to use:
- `float32_output=False` (default): matches HF/torchtitan/vLLM. Sufficient for most training.
- `float32_output=True`: matches Flash Attention. Recommended for long-context training or when investigating precision-related instability.

Under **AMP** (`torch.autocast`): matmul is a "fast" op that gets cast to lower precision, which is why autocast must be explicitly disabled for the frequency computation. Element-wise ops (the rotation) follow their input dtype and are not auto-promoted. This means AMP does not help with rotation precision -- the `float32_output` flag is the only control.

**`torch.set_float32_matmul_precision`**: Settings like `"high"` (TF32) or `"medium"` reduce precision in the multiply-accumulate stage of matmuls. The frequency computation uses `inv_freq @ position_ids` where the inner (contraction) dimension is 1 -- each output element is a single multiply with no accumulation. Therefore TF32/medium produce identical results to full float32 for this operation, and the setting is not a concern.

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
  -> (optional) pos_encoder(q, k, **kwargs)  (RoPE rotation via position_embeddings kwarg)
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
        pos_encoder: !partial:rotary_embeddings:apply_rotary_pos_emb
    norm_factory: !partial:torch.nn:RMSNorm
```

The `layer_idx` parameter is automatically threaded through: `LayerStack` passes it to each layer factory, and layers forward it to attention (needed for KV cache indexing and vLLM).

## Performance Optimizations

The modules support several optimization strategies, enabled via constructor flags:

| Optimization | Flag | Where | Effect |
|-------------|------|-------|--------|
| Triton fused SiLU/GELU*up | `use_triton=True` | `GLUFeedforwardLayer` | 1.67x forward, 1.50x backward |
| Float32 RoPE rotation | Always on for bf16/fp16 | `apply_rotary_pos_emb` | Numerical stability for half-precision training |
| Flash Attention 2 | `attn_implementation="flash_attention_2"` | Attention modules | O(seq) memory, fastest |
| Flex Attention + compile | `attn_implementation="flex_attention"` | Attention modules | Good with torch.compile |
| Gradient checkpointing | `enable_checkpoint=True` | `LayerStack` (checkpoint variant) | Memory/compute tradeoff |
| torch.compile | Model-level | All modules including RoPE | Fuses element-wise ops; no graph breaks in RoPE path |

All optimizations fall back gracefully when dependencies are unavailable.

## Project-Specific modelsrc

Model projects can define their own custom modules alongside the shared `modelsrc/transformer/` components. These live in a `modelsrc/` directory within the model project:

```
examples/models/llama_canon/
├── modelsrc/                        # Project-specific model modules
│   ├── canon_layer.py               # Custom Canon layer with Triton kernels
│   ├── canon_pre_ln_layer.py        # Canon-enhanced transformer layer
│   ├── canon_causal_multihead_attn.py
│   └── canon_glu_feedforward.py
└── templates/
    └── configs/
        └── default.yaml             # Adds modelsrc/ to submodule search path
```

The model config registers these modules via `[model_submodule_searchpath]`:

```yaml
[model_submodule_searchpath]
    - "{{ joinpath(project_dir, 'modelsrc') }}"
    == super()
```

The **import constraint** applies equally to project-specific modules: each file must be self-contained (no local imports), depending only on PyTorch, HF Transformers, and pip-installed packages.

### Cross-Project Inheritance Caveat

The `project_dir` variable resolves to the **current** project's directory, not the template's originating directory. When a child project extends a model project that has its own `modelsrc/`, the search path will point to the wrong location.

For example, if `examples/tiny_experiments/canon/models/` extends `examples/models/llama_canon/`, the `project_dir` in the inherited template resolves to `.../canon/models/`, which has no `modelsrc/` directory. The code generator will fail with `ModuleNotFoundError` when trying to import the custom modules.

**Fix**: Override `[model_submodule_searchpath]` in the model sub-project's baseline config to explicitly reference the base model's modelsrc directory:

```yaml
[model_submodule_searchpath]
    - "{{ joinpath(ns.forgather_dir, 'examples/models/llama_canon/modelsrc') }}"
    == super()
```

This override must be placed in an **inline model template** (the section after the `#--- config.*.model ---` separator), not in the main config section, because `[model_submodule_searchpath]` is defined within the model template inheritance chain.

**Recommended project structure**: Separate model definitions into a `models/` sub-project within your experiment directory. This isolates the model template search path from the training template search path. See `examples/tiny_experiments/canon/models/` for a working example, and the CLAUDE.md section "Creating an Experiment Project That Extends a Model Project" for the full pattern.

## File Reference

```
modelsrc/
├── templates/
│   └── hf_causal.py                  # HF PreTrainedModel template (Jinja2)
└── transformer/
    ├── causal_lm.py                  # CasualLM (core model)
    ├── input_encoder.py              # InputEncoder (embeddings + PE + dropout)
    │
    ├── rotary_embeddings.py          # RotaryEmbedding (nn.Module) + apply_rotary_pos_emb
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
