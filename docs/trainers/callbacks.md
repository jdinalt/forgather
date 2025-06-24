# Trainer Callbacks

The Forgather trainer system includes a sophisticated callback system that is fully compatible with HuggingFace Transformers callbacks while providing additional functionality.

## Callback System Overview

### Design Principles

- **Event-Driven**: Callbacks respond to specific training events
- **HuggingFace Compatible**: Same event interface as `transformers.TrainerCallback`
- **Extensible**: Easy to create custom callbacks
- **Configurable**: Full integration with template-based configuration

### Event Lifecycle

Training events are dispatched in this order:

1. `on_init_end` - After trainer initialization
2. `on_train_begin` - Before training starts
3. For each epoch:
   - `on_epoch_begin` - At start of epoch
   - For each batch:
     - `on_step_begin` - Before forward pass
     - `on_pre_optimizer_step` - Before optimizer step
     - `on_optimizer_step` - After optimizer step
     - `on_substep_end` - After substep (for gradient accumulation)
     - `on_step_end` - After complete step
     - `on_log` - When logging occurs (periodic)
     - `on_evaluate` - When evaluation occurs (periodic)
     - `on_save` - When saving occurs (periodic)
   - `on_epoch_end` - At end of epoch
4. `on_train_end` - After training completes

## Built-in Callbacks

### ProgressCallback

**Location**: `forgather.ml.default_callbacks.ProgressCallback`

Provides TQDM progress bars for training progress.

**Features:**
- Training step progress bar
- Evaluation progress bar
- Automatic formatting with metrics
- Multi-process safe

**Configuration:**
```yaml
# Included by default in Trainer
# No additional configuration needed
```

**Usage:**
```python
from forgather.ml.default_callbacks import ProgressCallback

trainer.add_callback(ProgressCallback())
```

### InfoCallback

**Location**: `forgather.ml.default_callbacks.InfoCallback`

Displays training information and configuration.

**Features:**
- Model parameter count
- Training configuration summary
- Dataset statistics
- Device information

**Configuration:**
```yaml
# Included by default in Trainer
# No additional configuration needed
```

### TBLogger (TensorBoard Logger)

**Location**: `forgather.ml.tb_logger.TBLogger`

Logs training metrics and configuration to TensorBoard.

**Features:**
- Scalar metric logging (loss, learning rate, etc.)
- Configuration serialization
- Experiment metadata tracking
- Multi-run comparison support

**Configuration:**
```yaml
summary_writer: &summary_writer !singleton:torch.utils.tensorboard:SummaryWriter
    - "./runs/experiment_001"

experiment_info: &experiment_info !dict:@experiment_info
    name: "My Experiment"
    description: "Testing new architecture"
    date: "2024-01-01"

trainer_callbacks: &trainer_callbacks !list:@trainer_callbacks
    - !singleton:forgather.ml.tb_logger:TBLogger
        args: [*summary_writer]
        kwargs:
            <<: *experiment_info
```

**Usage:**
```python
from torch.utils.tensorboard import SummaryWriter
from forgather.ml.tb_logger import TBLogger

writer = SummaryWriter("./runs/my_experiment")
callback = TBLogger(writer, name="My Experiment")
trainer.add_callback(callback)
```

### JsonLogger

**Location**: `forgather.ml.json_logger.JsonLogger`

Logs all training events and metrics to JSON format.

**Features:**
- Complete training log in structured format
- Metric history preservation
- Easy post-processing and analysis
- Configuration snapshot included

**Configuration:**
```yaml
trainer_callbacks: &trainer_callbacks !list:@trainer_callbacks
    - !singleton:forgather.ml.json_logger:JsonLogger
        name: "My Experiment"
        description: "Experiment description"
        config: !var "pp_config"  # Preprocessed configuration
```

**Output Format:**
```json
{
    "event": "on_log",
    "timestamp": "2024-01-01T12:00:00",
    "global_step": 100,
    "epoch": 0.5,
    "metrics": {
        "loss": 2.34,
        "learning_rate": 1e-4
    },
    "meta": {
        "name": "My Experiment",
        "config": {...}
    }
}
```

### TextgenCallback

**Location**: `forgather.ml.textgen_callback.TextgenCallback`

Generates text samples during training for monitoring progress.

**Features:**
- Periodic text generation
- Multiple prompt evaluation
- Generation parameter control
- TensorBoard integration

**Configuration:**
```yaml
test_prompts: &test_prompts !list:@test_prompts
    - "The weather today is"
    - "Machine learning is"
    - "In the future, we will"

generation_config: &generation_config !dict:@generation_config
    do_sample: true
    max_new_tokens: 50
    temperature: 0.7
    top_p: 0.9

trainer_callbacks: &trainer_callbacks !list:@trainer_callbacks
    - !singleton:forgather.ml.textgen_callback:TextgenCallback
        summary_writer: *summary_writer
        prompts: *test_prompts
        generation_config: *generation_config
        generation_steps: 1000  # Generate every 1000 steps
```

**Usage:**
```python
from forgather.ml.textgen_callback import TextgenCallback

callback = TextgenCallback(
    summary_writer=writer,
    prompts=["Hello, world!", "The future of AI is"],
    generation_config={"max_new_tokens": 30, "temperature": 0.7},
    generation_steps=500
)
trainer.add_callback(callback)
```

### GradLogger (Gradient Logger)

**Location**: `forgather.ml.grad_logger.GradLogger`

Monitors gradient statistics during training.

**Features:**
- Gradient norm tracking
- Per-layer gradient analysis
- Gradient flow visualization
- Vanishing/exploding gradient detection

**Configuration:**
```yaml
trainer_callbacks: &trainer_callbacks !list:@trainer_callbacks
    - !singleton:forgather.ml.grad_logger:GradLogger
        summary_writer: *summary_writer
        log_frequency: 100
        track_layers: ["layer.0", "layer.11"]  # Specific layers
```

**Metrics Logged:**
- `grad_norm_total` - Total gradient norm
- `grad_norm_max` - Maximum layer gradient norm
- `grad_norm_min` - Minimum layer gradient norm
- `grad_norm_mean` - Mean layer gradient norm

## Creating Custom Callbacks

### Basic Callback Structure

```python
from forgather.ml.trainer_types import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

class MyCustomCallback(TrainerCallback):
    def __init__(self, custom_param=None):
        self.custom_param = custom_param
        self.step_count = 0
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                       control: TrainerControl, **kwargs):
        print(f"Training started with {state.max_steps} steps")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        self.step_count += 1
        
        # Custom logic here
        if self.step_count % 100 == 0:
            print(f"Completed {self.step_count} steps")
        
        # Control training flow
        if self.step_count >= 1000:
            control.should_training_stop = True
        
        return control
    
    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        if logs and 'loss' in logs:
            loss = logs['loss']
            if loss < 0.1:
                print("Target loss reached!")
                control.should_training_stop = True
        
        return control
```

### Advanced Callback Example

```python
import torch
from pathlib import Path
import json

class ModelCheckpointCallback(TrainerCallback):
    """Save model when validation loss improves."""
    
    def __init__(self, output_dir, metric="eval_loss", mode="min"):
        self.output_dir = Path(output_dir)
        self.metric = metric
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_model_path = None
        
    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, metrics=None, **kwargs):
        if not metrics or self.metric not in metrics:
            return
        
        current_metric = metrics[self.metric]
        is_better = (
            (self.mode == 'min' and current_metric < self.best_metric) or
            (self.mode == 'max' and current_metric > self.best_metric)
        )
        
        if is_better:
            self.best_metric = current_metric
            
            # Save model
            model = kwargs.get('model')
            if model:
                checkpoint_dir = self.output_dir / f"best_model_step_{state.global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model state
                model.save_pretrained(checkpoint_dir)
                
                # Save metrics
                with open(checkpoint_dir / "metrics.json", "w") as f:
                    json.dump({
                        "step": state.global_step,
                        "epoch": state.epoch,
                        self.metric: current_metric,
                        "all_metrics": metrics
                    }, f, indent=2)
                
                self.best_model_path = checkpoint_dir
                print(f"New best model saved at {checkpoint_dir}")
                
            # Trigger save in trainer
            control.should_save = True
        
        return control
```

### Integration with Configuration

```yaml
# Custom callback in template
trainer_callbacks: &trainer_callbacks !list:@trainer_callbacks
    - !singleton:my_module.callbacks:ModelCheckpointCallback
        output_dir: "./best_models"
        metric: "eval_loss"
        mode: "min"
    
    - !singleton:my_module.callbacks:EarlyStoppingCallback
        patience: 3
        threshold: 0.001
```

## Callback Management

### Adding Callbacks

```python
# Add callback instance
trainer.add_callback(MyCustomCallback(custom_param="value"))

# Add callback class (will be instantiated)
trainer.add_callback(MyCustomCallback)

# Add multiple callbacks
callbacks = [
    TBLogger(summary_writer),
    JsonLogger(name="experiment"),
    TextgenCallback(prompts=test_prompts)
]
for callback in callbacks:
    trainer.add_callback(callback)
```

### Removing Callbacks

```python
# Remove specific callback instance
callback_instance = MyCustomCallback()
trainer.add_callback(callback_instance)
trainer.pop_callback(callback_instance)

# Remove by class
trainer.pop_callback(MyCustomCallback)

# Remove without returning
trainer.remove_callback(TBLogger)
```

### Callback Priority and Ordering

Callbacks are executed in the order they were added. For specific ordering:

```python
# Add high-priority callbacks first
trainer.add_callback(ModelCheckpointCallback())  # Execute first
trainer.add_callback(TBLogger(writer))           # Execute second
trainer.add_callback(TextgenCallback(prompts))   # Execute last
```

## Common Callback Patterns

### Conditional Logic

```python
class ConditionalCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Only act on specific steps
        if state.global_step % 500 == 0:
            # Custom logic every 500 steps
            pass
        
        # Only act in specific epochs
        if state.epoch >= 2:
            # Logic for later epochs
            pass
        
        return control
```

### External Service Integration

```python
import wandb

class WandBCallback(TrainerCallback):
    def __init__(self, project_name, run_name=None):
        wandb.init(project=project_name, name=run_name)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs, step=state.global_step)
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        wandb.finish()
```

### Multi-Process Safety

```python
class MultiProcessCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        # Only execute on main process
        if not state.is_world_process_zero:
            return control
        
        # Main process logic here
        print("Logging from main process only")
        return control
```

## Debugging Callbacks

### Callback Execution Logging

```python
class DebugCallback(TrainerCallback):
    def __init__(self, events_to_log=None):
        self.events_to_log = events_to_log or [
            'on_train_begin', 'on_step_end', 'on_log', 'on_evaluate'
        ]
    
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        
        if name.startswith('on_') and name in self.events_to_log:
            def logged_method(*args, **kwargs):
                print(f"Callback event: {name}")
                return attr(*args, **kwargs)
            return logged_method
        
        return attr
```

### Callback State Inspection

```python
class StateInspectionCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        print(f"Step {state.global_step}/{state.max_steps}")
        print(f"Epoch {state.epoch:.2f}")
        print(f"Recent logs: {state.log_history[-1] if state.log_history else 'None'}")
        
        # Inspect model if available
        model = kwargs.get('model')
        if model:
            print(f"Model device: {next(model.parameters()).device}")
        
        return control
```

## Best Practices

### 1. Keep Callbacks Focused

Each callback should have a single, clear responsibility:

```python
# Good: Single purpose
class LossMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only monitor loss
        pass

# Avoid: Multiple responsibilities
class EverythingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Logs metrics, saves models, generates text, etc.
        pass
```

### 2. Handle Multi-Process Training

```python
class SafeCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        # Check if this is the main process
        if state.is_world_process_zero:
            # Only execute on main process
            pass
        return control
```

### 3. Use Configuration for Flexibility

```python
class ConfigurableCallback(TrainerCallback):
    def __init__(self, frequency=100, enabled=True, **kwargs):
        self.frequency = frequency
        self.enabled = enabled
        self.config = kwargs
    
    def on_step_end(self, args, state, control, **kwargs):
        if not self.enabled:
            return control
        
        if state.global_step % self.frequency == 0:
            # Callback logic
            pass
        
        return control
```

### 4. Error Handling

```python
class RobustCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            # Callback logic that might fail
            self.risky_operation(logs)
        except Exception as e:
            print(f"Callback error: {e}")
            # Continue training despite error
        
        return control
```

## Migration from HuggingFace

Forgather callbacks are fully compatible with HuggingFace callbacks:

```python
# HuggingFace callback works directly
from transformers import EarlyStoppingCallback

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

# Forgather-specific enhancements
trainer.add_callback(TBLogger(writer))  # Additional functionality
```