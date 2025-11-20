# Fused Loss Integration: Trainer API Design

## Problem Statement

For large vocabulary models (e.g., Qwen3 with 151K tokens), the standard training pattern materializes massive logits tensors:

```python
# Current pattern
outputs = model(input_ids)  # Model returns logits
logits = outputs.logits     # [batch, seq, 151936] - 5GB+ tensor!
loss = loss_fn(logits, labels)
```

This causes severe memory issues, especially in pipeline parallel where multiple microbatches are in flight.

**Solution**: Fused linear + cross-entropy that never materializes logits, computing loss directly from hidden states.

## Design Requirements

1. **No special cases**: Keep trainer code general, not tied to specific implementations
2. **Dependency injection**: Use Callable pattern for flexibility
3. **Model interface compatibility**: Work with HuggingFace `PreTrainedModel.get_output_embeddings()`
4. **Pipeline parallel support**: Must work within PyTorch pipeline API constraints
5. **Backward compatibility**: Don't break existing training code

## Proposed API

### 1. Fused Loss Protocol

Define a protocol that fused loss implementations must follow:

```python
# In src/forgather/ml/loss.py

from typing import Protocol, Optional
import torch
from torch import Tensor, nn

class FusedOutputLoss(Protocol):
    """
    Protocol for fused output layer + loss computation.

    Implementations compute loss directly from hidden states without
    materializing full logits tensor, reducing memory for large vocabularies.
    """

    def __call__(
        self,
        hidden_states: Tensor,  # [batch, seq, hidden_dim]
        labels: Tensor,         # [batch, seq]
    ) -> Tensor:
        """Compute loss from hidden states and labels."""
        ...

    def forward_logits(
        self,
        hidden_states: Tensor,  # [batch, seq, hidden_dim]
    ) -> Tensor:
        """
        Inference mode: materialize logits for generation.

        Returns:
            logits: [batch, seq, vocab_size]
        """
        ...
```

### 2. Wrapper for Apple CCE

```python
# In src/forgather/ml/loss.py

class LinearCrossEntropyLoss:
    """
    Wrapper for fused linear + cross-entropy implementations.

    This wrapper makes CCE (or any linear_cross_entropy implementation)
    compatible with the trainer's expected interface.

    Args:
        output_embeddings: The output layer from model.get_output_embeddings()
                          Can be nn.Linear or any module with .weight and optional .bias
        impl: Implementation to use ("cce", "torch_compile", "pytorch")
        chunk_size: For pytorch impl, chunk size for vocabulary
        **kwargs: Additional arguments passed to linear_cross_entropy
    """

    def __init__(
        self,
        output_embeddings: nn.Module,
        impl: str = "cce",  # "cce" | "torch_compile" | "pytorch"
        chunk_size: int = 4096,
        **kwargs
    ):
        self.output_embeddings = output_embeddings
        self.impl = impl
        self.chunk_size = chunk_size
        self.kwargs = kwargs

        # Get weight and bias from output layer
        self.weight = output_embeddings.weight
        self.bias = getattr(output_embeddings, 'bias', None)

        # Select implementation
        if impl == "cce":
            try:
                from cut_cross_entropy import linear_cross_entropy
                self._compute = self._compute_cce
                self.linear_cross_entropy = linear_cross_entropy
            except ImportError:
                logger.warning("cut-cross-entropy not installed, falling back to pytorch")
                self._compute = self._compute_pytorch
        elif impl == "torch_compile":
            from cut_cross_entropy import linear_cross_entropy
            self.linear_cross_entropy = linear_cross_entropy
            self._compute = self._compute_cce
        else:  # pytorch
            self._compute = self._compute_pytorch

    def __call__(self, hidden_states: Tensor, labels: Tensor) -> Tensor:
        """Compute fused loss from hidden states."""
        return self._compute(hidden_states, labels)

    def _compute_cce(self, hidden_states: Tensor, labels: Tensor) -> Tensor:
        """Use Apple's CCE implementation."""
        return self.linear_cross_entropy(
            hidden_states,
            self.weight,
            labels,
            bias=self.bias,
            shift=1,  # Causal LM shifting
            impl=self.impl,
            **self.kwargs
        )

    def _compute_pytorch(self, hidden_states: Tensor, labels: Tensor) -> Tensor:
        """Use pure PyTorch chunked implementation."""
        # Shift for causal prediction
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
        flat_labels = shift_labels.view(-1)

        # Use our FusedLinearCrossEntropy implementation
        return _fused_linear_cross_entropy(
            flat_hidden, flat_labels,
            self.weight, self.bias,
            chunk_size=self.chunk_size
        )

    def forward_logits(self, hidden_states: Tensor) -> Tensor:
        """Inference mode: materialize logits."""
        return torch.nn.functional.linear(
            hidden_states, self.weight, self.bias
        )
```

### 3. Model Modification Flag

Add a simple flag to tell models whether to return hidden states or logits:

```python
# In model configuration or trainer args
return_hidden_for_loss: bool = False  # New flag
```

When `True`, model's forward returns hidden states instead of logits for loss computation.

### 4. Trainer Integration

Modify `_forward_backward_step` to support fused loss:

```python
# In src/forgather/ml/trainer/trainer.py

def _forward_backward_step(
    self, input_dict: dict[str, Tensor], labels: Tensor
) -> Tensor:
    if self.train_loss_fn:
        # Check if loss function is fused (has forward_logits method)
        if hasattr(self.train_loss_fn, 'forward_logits'):
            # Fused loss: model returns hidden states
            outputs = self.model(**input_dict, return_hidden_for_loss=True)
            hidden_states = self._extract_hidden_states(outputs)
            loss = self.train_loss_fn(hidden_states, labels)
        else:
            # Standard loss: model returns logits
            outputs = self.model(**input_dict)
            logits = logits_from_outputs(outputs)
            loss = self.train_loss_fn(logits, labels)
    else:
        # Model computes loss internally
        outputs = self.model(labels=labels, **input_dict)
        loss = loss_from_outputs(outputs)

    self._backward(loss)
    return loss.detach()

def _extract_hidden_states(self, outputs):
    """Extract hidden states from model outputs."""
    # Handle different output types
    if isinstance(outputs, Tensor):
        return outputs
    elif hasattr(outputs, 'hidden_states'):
        return outputs.hidden_states
    elif hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state
    elif isinstance(outputs, tuple):
        return outputs[0]
    else:
        raise ValueError(f"Cannot extract hidden states from {type(outputs)}")
```

### 5. Pipeline Parallel Adaptation

For pipeline parallel, the pattern is slightly different - the model MUST return hidden states (not logits):

```python
# In src/forgather/ml/trainer/pipeline/pipeline_trainer.py

def _prepare_model(self):
    ...
    # For pipeline, always use return_hidden_for_loss=True
    # because loss is computed externally by pipeline scheduler
    if hasattr(self.model, 'config'):
        self.model.config.return_hidden_for_loss = True
    ...
```

## Usage Example

### Configuration

```yaml
# Training script configuration

# Model (Qwen3 1.7B)
model: !call:transformers.AutoModelForCausalLM.from_pretrained
  - "Qwen/Qwen-1.7B"

# Extract output embeddings from model
output_embeddings: !ref model.get_output_embeddings()

# Fused loss using Apple CCE
loss_fn: !call:LinearCrossEntropyLoss
  output_embeddings: !ref output_embeddings
  impl: "cce"  # or "torch_compile" or "pytorch"
  chunk_size: 4096

# Trainer
trainer: !call:Trainer
  model: !ref model
  args: !ref training_args
  loss_fn: !ref loss_fn  # Pass as compute_loss_func
  train_dataset: !ref train_dataset
```

### Python Code

```python
from forgather.ml.trainer import Trainer, TrainingArguments
from forgather.ml.loss import LinearCrossEntropyLoss
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1.7B")

# Create fused loss
fused_loss = LinearCrossEntropyLoss(
    output_embeddings=model.get_output_embeddings(),
    impl="cce",  # Apple's optimized kernels
    chunk_size=4096
)

# Train with fused loss
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_loss_func=fused_loss,  # Use fused loss
)

trainer.train()
```

## Implementation Plan

1. **Add `LinearCrossEntropyLoss` wrapper** to `src/forgather/ml/loss.py`
   - Support CCE, torch_compile, and pure PyTorch backends
   - Handle output embeddings extraction
   - Implement `forward_logits()` for inference

2. **Update `Trainer._forward_backward_step()`**
   - Detect fused loss via `hasattr(loss_fn, 'forward_logits')`
   - Extract hidden states from model outputs
   - Pass hidden states to fused loss

3. **Update `PipelineTrainer`**
   - Ensure model returns hidden states
   - Loss function receives hidden states from scheduler

4. **Add helper utilities**
   - `_extract_hidden_states()` for different output types
   - Auto-detection of fused loss capability

5. **Documentation and examples**
   - Update training examples
   - Document memory savings
   - Provide migration guide

## Benefits

1. **No trainer modifications** for specific implementations
2. **Clean separation**: Loss function owns output layer
3. **Flexible**: Works with CCE, custom implementations, future optimizations
4. **Backward compatible**: Existing code continues to work
5. **Inference support**: `forward_logits()` for generation
6. **Memory efficient**: 83% reduction with CCE

## Alternative Considered: Factory Pattern

Instead of passing loss function directly, could use a factory:

```python
def create_fused_loss(model):
    """Factory that extracts output embeddings and creates fused loss."""
    output_embeddings = model.get_output_embeddings()
    return LinearCrossEntropyLoss(output_embeddings, impl="cce")

trainer = Trainer(
    model=model,
    loss_factory=create_fused_loss,  # Factory instead of loss directly
    ...
)
```

**Decision**: Direct injection is simpler and more explicit. Factory pattern doesn't provide enough benefit to justify complexity.

