# Integrating Fused Linear+Loss with PyTorch Pipeline Parallel

## Problem Summary

Training Qwen3 1.7B (vocab_size=151936) in pipeline parallel mode shows:
- Last stage (with output layer): **20.5 GB peak memory**
- Middle stages: **~2 GB peak memory**

Memory profiling reveals:
- Logits tensor per microbatch: **1.2 GB** (batch_size=1, seq_len=4096, bfloat16)
- With 3-4 microbatches in flight: **3.6-4.8 GB just for logits**
- Backward pass temporary allocations: **additional ~2-3 GB**
- **Total peak: 10-12 GB** for the output computation alone

## PyTorch Pipeline Parallel API Flow

### Forward Pass (ScheduleGPipe._step_microbatches)

```python
for i in range(n_microbatches):
    # 1. Model forward (last stage)
    output = stage.forward_one_chunk(i, args[i], kwargs[i])
    #    → Returns: logits [batch, seq, vocab_size]
    #    → Stored in: stage.output_chunks (for final merge)
    #    → Stored in: stage.fwd_cache[i] (for backward)

    # 2. Loss computation (if last stage)
    _maybe_compute_loss(stage, output, targets, i)
    #    → Calls: loss_fn(output, target)
    #    → Stores: loss in _internal_losses[i]
```

### Backward Pass

```python
for i in range(n_microbatches):
    # 1. Retrieve loss for this microbatch
    loss = _maybe_get_loss(stage, i)  # From _internal_losses

    # 2. Backward through loss
    stage.backward_one_chunk(i, loss=loss, ...)
    #    → For last stage: uses 'loss' as the starting gradient
    #    → Calls: torch.autograd.backward(loss)
```

## The Memory Problem

The issue is in the **forward pass storage**:

1. **Model returns logits** → stored in `stage.output_chunks` and `stage.fwd_cache`
2. **Loss is computed** → creates computational graph linking loss ← logits
3. **Logits must stay in memory** until backward completes (part of autograd graph)
4. With **3-4 microbatches in flight**: 3-4× logits in memory simultaneously

Even though loss is computed immediately, **the logits tensor cannot be freed** because:
- It's in the autograd graph between model output and loss
- Backward needs to propagate gradients through: loss → logits → hidden_states

## Why Chunked Loss Doesn't Help

`ChunkedCausalLoss(logits, labels)` receives the **already-materialized** logits:

```python
# Model forward
logits = output_layer(hidden_states)  # ← 1.2 GB allocated here!

# Loss (chunked or not)
loss = loss_fn(logits, labels)  # ← Too late, logits already exist
```

The chunked loss only helps during the loss computation itself, but logits are already in memory.

## Solution: Fused Linear + Cross-Entropy

`FusedLinearCrossEntropy` computes loss **directly from hidden states**, never materializing logits:

```python
# Instead of:
logits = linear(hidden)      # 1.2 GB
loss = cross_entropy(logits, labels)

# We do:
loss = fused_layer(hidden, labels)  # No logits! Chunked internally
```

Autograd graph becomes: loss → hidden_states (skipping the 1.2 GB logits intermediary)

## Integration Challenges

### Challenge 1: Pipeline API Contract

The pipeline expects:
```python
output = model.forward(inputs)              # Model returns something
loss = loss_fn(output, targets)             # Loss fn receives that something
```

But `FusedLinearCrossEntropy` needs:
```python
loss = fused_layer(hidden_states, targets)  # Needs both inputs!
```

The pipeline doesn't pass `targets` to the model, only to the loss function.

### Challenge 2: Model Output Requirements

For last stage:
- **Training**: Need to return loss (or something to compute loss from)
- **Inference**: Need to return logits for generation/evaluation

The fused layer combines both, but pipeline needs separation.

## Potential Solutions

### Option 1: Custom Loss Function Wrapper (RECOMMENDED)

Create a loss function that wraps both the output layer and loss computation:

```python
class PipelineFusedLoss:
    def __init__(self, output_layer, chunk_size=4096):
        # Move output layer INTO the loss function
        self.output_layer = output_layer
        self.chunk_size = chunk_size

    def __call__(self, hidden_states, labels):
        # hidden_states is what model returns (not logits!)
        # Compute fused output + loss
        return self._fused_linear_cross_entropy(
            hidden_states, labels,
            self.output_layer.weight,
            self.output_layer.bias
        )

    def _fused_linear_cross_entropy(self, hidden, labels, weight, bias):
        # Same chunked implementation as FusedLinearCrossEntropy
        ...
```

Model modification:
```python
class CausalLM:
    def forward(self, ..., labels=None):
        ...
        if self.output_decoder:
            if labels is not None and self.pipeline_mode:
                # For pipeline training: return hidden states, not logits!
                return hidden_states
            else:
                # For inference: return logits as normal
                logits = self.output_decoder(hidden_states)
                return logits
```

Configuration:
```yaml
# Create the fused loss with embedded output layer
model: ...
  # Don't include output_decoder in model for pipeline training

loss_fn: !call:PipelineFusedLoss
  output_layer: !call:nn.Linear [2048, 151936]
  chunk_size: 4096
```

**Pros:**
- Works within pipeline API constraints
- Model returns hidden_states (16 MB vs 1.2 GB)
- Loss function handles the heavy computation
- Clean separation: model = transformer, loss = output+cross_entropy

**Cons:**
- Model needs to know about "pipeline mode"
- Output layer lives in loss function (slightly unusual)
- Need separate handling for eval/inference

### Option 2: Modify Model to Return Multiple Outputs

Have model return `(hidden_states, logits)` tuple:

```python
def forward(self, ...):
    ...
    if self.output_decoder:
        logits = self.output_decoder(hidden_states)
        return (hidden_states, logits)  # Return both
```

Custom loss function receives both:
```python
def fused_loss_fn(outputs, labels):
    hidden_states, logits = outputs
    # Use hidden_states for fused computation
    # Ignore logits (they exist but won't be used)
    return fused_linear_ce(hidden_states, labels, ...)
```

**Pros:**
- Model doesn't need mode flags
- Always returns consistent format

**Cons:**
- Still materializes logits (defeats the purpose!)
- Wastes memory unless we avoid computing logits

### Option 3: Custom Pipeline Schedule (COMPLEX)

Fork `ScheduleGPipe` to pass targets to model forward:

```python
class ScheduleGPipeWithTargets(ScheduleGPipe):
    def _step_microbatches(self, ...):
        for i in range(n_microbatches):
            if stage.is_last:
                # Pass targets to model!
                output = stage.forward_one_chunk(
                    i, args[i],
                    {**kwargs[i], 'labels': targets[i]}
                )
                # Model returns loss directly
                _internal_losses.append(output)
            else:
                output = stage.forward_one_chunk(i, args[i], kwargs[i])
```

**Pros:**
- Most flexible, model has full control
- Clean model API

**Cons:**
- Requires maintaining forked pipeline code
- Breaks compatibility with standard PyTorch pipeline
- Complex to maintain

### Option 4: Lazy Logits Wrapper (HACKY)

Wrap hidden_states in a container that computes logits on-demand:

```python
class LazyLogits:
    def __init__(self, hidden_states, output_layer):
        self.hidden = hidden_states
        self.layer = output_layer
        self._logits = None

    def materialize(self):
        if self._logits is None:
            self._logits = self.layer(self.hidden)
        return self._logits

# Model returns LazyLogits
# Standard loss: calls .materialize() → gets logits
# Fused loss: uses .hidden directly
```

**Pros:**
- Backwards compatible with standard loss
- Flexible for different loss types

**Cons:**
- Complex, fragile
- Doesn't actually save memory if standard loss is used

## Recommended Approach: Option 1

The **PipelineFusedLoss** wrapper is the most practical solution:

1. **Model changes**: Minimal - just return hidden_states instead of logits when in pipeline mode
2. **Pipeline changes**: None - uses existing API
3. **Performance**: Optimal - no logits materialization
4. **Compatibility**: Can coexist with standard training path

### Implementation Plan

1. Create `PipelineFusedLoss` class in `src/forgather/ml/loss.py`
2. Add `pipeline_training_mode` flag to `CausalLM`
3. Modify model forward to return hidden_states when flag is set
4. Update pipeline trainer to:
   - Set model to pipeline_training_mode
   - Use PipelineFusedLoss with embedded output layer
5. Keep standard training path unchanged

### Memory Impact

Expected reduction (based on profiling):
- Current: **20.5 GB peak** (with logits)
- With fusion: **~11-12 GB peak** (43% reduction from profiling)
- Savings: **~9 GB** → enough to increase batch size or use on smaller GPUs

## Open Questions for Research Review

1. Are there established patterns for this in other frameworks? (DeepSpeed, Megatron-LM?)
2. What do recent papers recommend for large vocabulary models?
3. Is there a better way to handle the model/loss separation in pipeline parallel?

