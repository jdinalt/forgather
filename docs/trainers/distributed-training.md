# Distributed Training

Forgather provides multiple distributed training strategies to handle different scales and types of parallelism, from simple multi-GPU data parallelism to advanced pipeline parallelism for very large models.

## Overview

### Available Strategies

1. **AccelTrainer**: Data parallelism using HuggingFace Accelerate
2. **PipelineTrainer**: Pipeline parallelism using PyTorch Distributed
3. **Trainer**: Single GPU/CPU (baseline)

### Choosing the Right Strategy

| Model Size | Memory per GPU | Recommended Strategy |
|------------|----------------|---------------------|
| < 1B params | 8GB+ | Trainer (single GPU) |
| 1B - 7B params | 16GB+ per GPU | AccelTrainer (data parallel) |
| 7B - 70B params | 24GB+ per GPU | AccelTrainer (data parallel) |
| > 70B params | Any | PipelineTrainer (pipeline parallel) |

## AccelTrainer: Data Parallelism

### Overview

AccelTrainer uses HuggingFace Accelerate for automatic multi-GPU data parallelism with minimal configuration changes.

**Key Features:**
- Automatic device placement and data distribution
- Gradient synchronization across processes
- Mixed precision training (FP16/BF16)
- Efficient communication with optimized collectives

### Basic Configuration

```yaml
-- extends 'trainers/accel_trainer.yaml'

-- block trainer_args
    == super()
    
    # Basic training settings
    output_dir: "./output_models/distributed_model"
    per_device_train_batch_size: 8  # Per GPU batch size
    per_device_eval_batch_size: 16
    
    # Training schedule
    num_train_epochs: 3
    logging_steps: 100
    eval_steps: 500
    save_steps: 1000
    
    # Accelerate-specific settings
    accelerator_args:
        device_placement: true
        split_batches: false
        dispatch_batches: true
-- endblock trainer_args
```

### Advanced Accelerate Configuration

```yaml
-- block trainer_args
    == super()
    
    # Distributed settings
    accelerator_args:
        # Device and placement
        device_placement: true
        cpu: false
        
        # Batch handling
        split_batches: false      # Don't split batches across devices
        dispatch_batches: true    # Efficiently dispatch batches
        
        # Mixed precision
        mixed_precision: "fp16"   # "no" | "fp16" | "bf16"
        
        # Gradient handling
        gradient_accumulation_plugin:
            num_steps: 4
            adjust_scheduler: true
        
        # Data loading
        dataloader_config:
            split_batches: false
            even_batches: true
            use_seedable_sampler: true
-- endblock trainer_args
```

### Execution with Accelerate

#### Using accelerate launch

```bash
# Configure accelerate (one-time setup)
accelerate config

# Launch distributed training
accelerate launch scripts/train_script.py my_config.yaml
```

#### Manual configuration

```bash
# 4 GPUs on single node
accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port 29500 \
    scripts/train_script.py my_config.yaml

# Multi-node setup (8 GPUs across 2 nodes)
# Node 0:
accelerate launch \
    --num_processes 8 \
    --num_machines 2 \
    --machine_rank 0 \
    --main_process_ip 192.168.1.10 \
    --main_process_port 29500 \
    scripts/train_script.py my_config.yaml

# Node 1:
accelerate launch \
    --num_processes 8 \
    --num_machines 2 \
    --machine_rank 1 \
    --main_process_ip 192.168.1.10 \
    --main_process_port 29500 \
    scripts/train_script.py my_config.yaml
```

### Performance Optimization

#### Batch Size Tuning

```yaml
-- block trainer_args
    # Effective batch size = per_device_batch_size * num_gpus * gradient_accumulation_steps
    per_device_train_batch_size: 16   # Per GPU
    gradient_accumulation_steps: 4    # If needed for memory
    
    # Total effective batch size with 4 GPUs = 16 * 4 * 4 = 256
-- endblock trainer_args
```

#### Communication Optimization

```yaml
accelerator_args:
    # Use optimized communication backend
    backend: "nccl"              # NCCL for NVIDIA GPUs
    
    # Optimize data loading
    dataloader_config:
        num_workers: 4           # Per process
        pin_memory: true
        persistent_workers: true
```

## PipelineTrainer: Pipeline Parallelism

### Overview

PipelineTrainer enables training of very large models that don't fit on a single GPU by partitioning the model across multiple devices.

**Key Features:**
- Automatic model partitioning
- Multiple pipeline scheduling strategies
- Supports both inter-node and intra-node parallelism
- Memory-efficient gradient computation

### Basic Configuration

```yaml
-- extends 'trainers/pipeline_trainer.yaml'

-- block trainer_args
    == super()
    
    # Output and logging
    output_dir: "./output_models/pipeline_model"
    logging_steps: 10
    save_steps: 500
    
    # Pipeline configuration
    split_spec:
        "layer.4": "end"         # End of stage 0 (layers 0-4)
        "layer.8": "end"         # End of stage 1 (layers 5-8)
        "layer.12": "end"        # End of stage 2 (layers 9-12)
        # Remaining layers go to stage 3
    
    pipeline_chunks: 4           # Number of micro-batches
    stages_per_rank: 1           # One stage per GPU
    
    # Training settings
    per_device_train_batch_size: 4
    max_steps: 2000
-- endblock trainer_args

-- block distributed_env
    !singleton:forgather.ml.distributed:DistributedEnvironment@distributed_env
-- endblock distributed_env

-- block loss_fn
    !singleton:torch.nn:CrossEntropyLoss@loss_fn
-- endblock loss_fn
```

### Advanced Pipeline Configuration

```yaml
-- block trainer_args
    == super()
    
    # Detailed pipeline setup
    split_spec:
        # Transformer layers
        "transformer.layers.6": "end"    # Stage 0: layers 0-6
        "transformer.layers.12": "end"   # Stage 1: layers 7-12
        "transformer.layers.18": "end"   # Stage 2: layers 13-18
        # Stage 3: layers 19+ and output layers
    
    # Pipeline scheduling
    schedule_class: "Schedule1F1B"       # 1Forward1Backward scheduling
    pipeline_chunks: 8                   # More micro-batches for efficiency
    
    # Memory optimization
    checkpoint_sequential: true          # Sequential gradient checkpointing
    
    # Performance tuning
    stages_per_rank: 2                   # Multiple stages per GPU
    pipeline_parallel_size: 4            # Total pipeline stages
    
    # Batch configuration
    per_device_train_batch_size: 2       # Smaller batches for pipeline
    gradient_accumulation_steps: 8       # Maintain effective batch size
-- endblock trainer_args
```

### Pipeline Scheduling Strategies

#### GPipe (Default)

```yaml
schedule_class: "ScheduleGPipe"
```
- **Characteristics**: Simple, bubble-minimal
- **Memory usage**: Higher (all activations stored)
- **Best for**: Smaller models, debugging

#### 1F1B (1Forward1Backward)

```yaml
schedule_class: "Schedule1F1B"
```
- **Characteristics**: Memory efficient, overlapped computation
- **Memory usage**: Lower (bounded activation storage)
- **Best for**: Large models, production training

#### Interleaved 1F1B

```yaml
schedule_class: "ScheduleInterleaved1F1B"
stages_per_rank: 2  # Multiple stages per device
```
- **Characteristics**: Best throughput, complex scheduling
- **Memory usage**: Moderate
- **Best for**: Maximum efficiency with sufficient GPUs

### Execution with Pipeline Parallelism

#### Single Node, Multiple GPUs

```bash
# 4 GPUs, 4 pipeline stages
torchrun \
    --nproc_per_node 4 \
    --nnodes 1 \
    scripts/train_script.py pipeline_config.yaml
```

#### Multi-Node Pipeline

```bash
# 2 nodes, 8 GPUs total, 8 pipeline stages
# Node 0:
torchrun \
    --nproc_per_node 4 \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr 192.168.1.10 \
    --master_port 29500 \
    scripts/train_script.py pipeline_config.yaml

# Node 1:
torchrun \
    --nproc_per_node 4 \
    --nnodes 2 \
    --node_rank 1 \
    --master_addr 192.168.1.10 \
    --master_port 29500 \
    scripts/train_script.py pipeline_config.yaml
```

### Memory and Performance Optimization

#### Model Partitioning Strategy

```python
# Manual partitioning for custom models
def get_split_spec(model, num_stages):
    """Generate split spec based on model architecture."""
    total_layers = len(model.transformer.layers)
    layers_per_stage = total_layers // num_stages
    
    split_spec = {}
    for i in range(num_stages - 1):
        layer_idx = (i + 1) * layers_per_stage - 1
        split_spec[f"transformer.layers.{layer_idx}"] = "end"
    
    return split_spec
```

#### Micro-batch Size Tuning

```yaml
# Balance between throughput and memory
pipeline_chunks: !calc "max(1, total_batch_size // per_device_batch_size)"

# Example calculations:
# - per_device_batch_size: 4
# - total_desired_batch_size: 64
# - num_pipeline_stages: 4
# - pipeline_chunks: 64 // 4 = 16
```

## Hybrid Parallelism

### Data + Pipeline Parallelism

For extremely large models, combine both strategies:

```yaml
# Configuration for hybrid parallelism
-- block trainer_args
    # Pipeline parallelism within each data parallel group
    split_spec:
        "layer.8": "end"
        "layer.16": "end"
    pipeline_chunks: 4
    
    # Data parallelism across pipeline groups
    data_parallel_size: 2        # 2 data parallel groups
    pipeline_parallel_size: 4    # 4 pipeline stages each
    
    # Total GPUs = data_parallel_size * pipeline_parallel_size = 8
-- endblock trainer_args
```

### Execution

```bash
# 8 GPUs: 2 data parallel groups × 4 pipeline stages
torchrun \
    --nproc_per_node 8 \
    scripts/train_script.py hybrid_config.yaml
```

## Distributed Environment Setup

### DistributedEnvironment Configuration

```yaml
distributed_env: &distributed_env !singleton:forgather.ml.distributed:DistributedEnvironment@distributed_env
    # Automatically detects distributed setup from environment variables
    backend: "nccl"              # "nccl" | "gloo" | "mpi"
    init_method: "env://"        # Use environment variables
    timeout_seconds: 1800        # 30 minutes
```

### Environment Variables

Key environment variables for distributed training:

```bash
# Set by torchrun automatically
export RANK=0                   # Global rank of this process
export WORLD_SIZE=4             # Total number of processes
export LOCAL_RANK=0             # Local rank on this node
export LOCAL_WORLD_SIZE=4       # Number of processes on this node
export MASTER_ADDR=localhost    # Master node address
export MASTER_PORT=29500        # Master node port

# Optional custom settings
export NCCL_DEBUG=INFO          # NCCL debugging
export CUDA_VISIBLE_DEVICES=0,1,2,3  # GPU visibility
```

## Monitoring and Debugging

### Distributed Logging

```yaml
trainer_callbacks: &trainer_callbacks !list:@trainer_callbacks
    # Only log from rank 0 to avoid duplication
    - !singleton:forgather.ml.tb_logger:TBLogger
        args: [*summary_writer]
        
    - !singleton:forgather.ml.json_logger:JsonLogger
        name: "Distributed Training"
        rank_filter: 0           # Only log from main process
```

### Performance Monitoring

```python
class DistributedMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            # Memory usage
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            
            # Communication stats (if available)
            if hasattr(torch.distributed, 'get_comm_stats'):
                stats = torch.distributed.get_comm_stats()
                print(f"Communication overhead: {stats}")
            
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
```

### Common Issues and Solutions

#### 1. NCCL Timeout

```bash
# Increase timeout for slow networks
export NCCL_TIMEOUT_SECONDS=3600
export NCCL_BLOCKING_WAIT=1
```

#### 2. Memory Imbalance

```yaml
# Adjust pipeline partitioning
split_spec:
    # Move split points to balance memory usage
    "layer.6": "end"    # Instead of "layer.4"
    "layer.12": "end"   # Instead of "layer.8"
```

#### 3. Communication Bottlenecks

```yaml
# Optimize communication
accelerator_args:
    # Reduce communication frequency
    gradient_accumulation_steps: 8
    
    # Use efficient data loading
    dataloader_config:
        num_workers: 2  # Reduce per-process workers
        prefetch_factor: 2
```

## Best Practices

### 1. Hardware Configuration

#### Network Requirements

- **InfiniBand**: Best for large-scale training
- **Ethernet**: 25Gbps+ recommended for multi-node
- **NVLink**: Optimal for single-node multi-GPU

#### Memory Considerations

```yaml
# For GPU memory efficiency
per_device_train_batch_size: 4    # Start small
gradient_accumulation_steps: 8    # Maintain effective batch size
torch_compile: true               # Memory optimization
```

### 2. Configuration Strategy

#### Start Simple

```yaml
# Begin with data parallelism
-- extends 'trainers/accel_trainer.yaml'
# Scale to pipeline parallelism only if needed
```

#### Gradual Scaling

1. **Single GPU**: Validate training logic
2. **Multi-GPU (single node)**: Test data parallelism
3. **Multi-node**: Scale horizontally
4. **Pipeline parallelism**: For memory constraints

### 3. Debugging Approach

#### Test Scaling

```python
# Test with smaller model first
def test_distributed_setup():
    # Use tiny model for validation
    small_config = {
        "hidden_size": 128,
        "num_layers": 2,
        "num_attention_heads": 2,
    }
    
    # Verify distributed communication
    trainer = AccelTrainer(model_config=small_config, ...)
    trainer.train()
```

#### Profiling

```python
# Profile distributed performance
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    trainer.train()
```

### 4. Production Deployment

#### Resource Planning

```python
# Calculate resource requirements
def calculate_resources(model_params, sequence_length, batch_size):
    # Rough memory estimation (GB)
    model_memory = model_params * 4 / 1e9  # FP32 weights
    activation_memory = batch_size * sequence_length * model_params * 4 / 1e9
    gradient_memory = model_memory  # Same as model for gradients
    
    total_memory = model_memory + activation_memory + gradient_memory
    return total_memory * 1.5  # Safety margin
```

#### Fault Tolerance

```yaml
# Enable checkpointing for long runs
save_strategy: "steps"
save_steps: 500
save_total_limit: 5

# Resume capability
resume_from_checkpoint: true
```

#### Monitoring

```yaml
trainer_callbacks: &trainer_callbacks !list:@trainer_callbacks
    - !singleton:forgather.ml.tb_logger:TBLogger
        args: [*summary_writer]
    - !singleton:custom_callbacks:ResourceMonitorCallback
        gpu_memory_threshold: 0.9
        alert_on_imbalance: true
```