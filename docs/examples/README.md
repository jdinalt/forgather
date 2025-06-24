# Examples

Working examples demonstrating Forgather concepts and common use cases.

## Quick Examples

### Basic Training
- **[Simple Training](basic-training/single-gpu.md)** - Train a small model on single GPU
- **[Multi-GPU Training](basic-training/multi-gpu.md)** - Scale to multiple GPUs with Accelerate
- **[Checkpoint Resume](basic-training/checkpointing.md)** - Save and resume training

### Distributed Training
- **[Data Parallelism](distributed-training/data-parallel.md)** - Multi-GPU data parallelism
- **[Pipeline Parallelism](distributed-training/pipeline-parallel.md)** - Large model pipeline training
- **[Multi-Node Training](distributed-training/multi-node.md)** - Scale across multiple machines

### Custom Models
- **[Custom Architecture](custom-models/architecture.md)** - Define new model architectures
- **[Model Templates](custom-models/templates.md)** - Create reusable model templates
- **[Component Library](custom-models/components.md)** - Build custom model components

### Advanced Workflows
- **[Hyperparameter Sweeps](advanced-workflows/hyperparameter-sweeps.md)** - Systematic hyperparameter exploration
- **[Multi-Stage Training](advanced-workflows/multi-stage.md)** - Complex training pipelines
- **[Custom Callbacks](advanced-workflows/custom-callbacks.md)** - Advanced monitoring and control

## Example Projects

Located in the `examples/` directory of the repository:

### Tiny Experiments
```
examples/tiny_experiments/
├── checkpointing/          # Checkpoint functionality demo
├── simple_training/        # Basic training example
└── custom_model/           # Custom model example
```

### Production Examples
```
examples/trainers/
├── simple/                 # Single GPU training
├── distributed/            # Multi-GPU with Accelerate  
├── pipeline/               # Pipeline parallel training
└── custom/                 # Custom trainer implementations
```

### Model Library
```
examples/models/
├── transformers/           # Transformer variants
├── diffusion/              # Diffusion models
└── custom/                 # Custom architectures
```

## Using Examples

### 1. Clone and Explore
```bash
git clone https://github.com/anthropics/forgather.git
cd forgather/examples
```

### 2. Run Examples
```bash
# Simple training example
cd tiny_experiments/checkpointing
python ../../../scripts/train_script.py train.yaml

# Multi-GPU training
cd ../../../examples/trainers/distributed
accelerate launch ../../../scripts/train_script.py config.yaml
```

### 3. Modify and Experiment
```bash
# Copy example as starting point
cp -r examples/tiny_experiments/checkpointing my_experiment
cd my_experiment

# Modify configuration
vim templates/configs/train.yaml

# Run your version
python ../../../scripts/train_script.py train.yaml
```

## Example Categories

### By Complexity
- **Beginner**: Single GPU, small models, basic configuration
- **Intermediate**: Multi-GPU, custom components, advanced configuration
- **Advanced**: Distributed training, custom trainers, complex workflows

### By Use Case
- **Research**: Systematic experimentation, ablation studies
- **Production**: Scalable training, monitoring, deployment
- **Education**: Learning ML concepts, understanding implementations

### By Framework Integration
- **HuggingFace**: Migration examples, compatibility patterns
- **PyTorch**: Native PyTorch integration, custom components
- **External**: Integration with other tools and frameworks

## Contributing Examples

See [Contributing Guidelines](../contributing/) for:
- Example structure and conventions
- Documentation requirements
- Testing and validation
- Submission process

## Example Index

### Getting Started
| Example | Description | Complexity | GPU Requirements |
|---------|-------------|------------|------------------|
| [Tiny Training](../../../examples/tiny_experiments/checkpointing/) | Basic training with checkpoints | Beginner | 1 GPU or CPU |
| [Simple Distributed](basic-training/multi-gpu.md) | Multi-GPU data parallelism | Intermediate | 2+ GPUs |

### Model Types
| Example | Model Type | Size | Training Time |
|---------|------------|------|---------------|
| Tiny Causal | Transformer | 4M params | 5 minutes |
| Small Language Model | GPT-style | 125M params | 30 minutes |
| Custom Architecture | User-defined | Variable | Variable |

### Training Strategies
| Example | Strategy | Use Case | Requirements |
|---------|----------|----------|--------------|
| Single GPU | Basic training | Development, small models | 1 GPU |
| Data Parallel | Multi-GPU scaling | Medium models | 2-8 GPUs |
| Pipeline Parallel | Memory scaling | Large models | 4+ GPUs |
| Hybrid Parallel | Maximum scaling | Very large models | 8+ GPUs |

---

*All examples are tested and maintained. Report issues or suggest improvements in [GitHub issues](https://github.com/anthropics/forgather/issues).*