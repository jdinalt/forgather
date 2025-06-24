# Guides

Practical guides for common tasks, best practices, and advanced topics.

## Essential Guides

- **[Migration from HuggingFace](migration-from-hf.md)** - Moving from transformers.Trainer
- **[Performance Optimization](performance-optimization.md)** - Getting the best performance
- **[Debugging](debugging.md)** - Troubleshooting common issues
- **[Best Practices](best-practices.md)** - Recommended patterns and approaches

## Advanced Topics

- **[Production Deployment](production-deployment.md)** - Running Forgather in production
- **[Custom Extensions](custom-extensions.md)** - Extending the framework
- **[Integration Patterns](integration-patterns.md)** - Integrating with other tools

## Quick Solutions

### Common Tasks

#### Migrating from HuggingFace
```python
# Old HuggingFace code
from transformers import Trainer, TrainingArguments

# New Forgather code (minimal changes)
from forgather.ml import Trainer, TrainingArguments
```

#### Debugging Configuration
```bash
# See preprocessed configuration
fgcli.py -t config.yaml pp

# Validate configuration
fgcli.py -t config.yaml validate
```

#### Performance Tuning
```yaml
# Enable optimizations
torch_compile: true
dataloader_num_workers: 4
per_device_train_batch_size: 32  # Tune for your GPU
```

#### Multi-GPU Setup
```bash
# Use accelerate for easy multi-GPU
accelerate config
accelerate launch scripts/train_script.py config.yaml
```

### Troubleshooting

#### Template Inheritance Issues
```yaml
# Problem: Configuration not appearing
-- extends 'base.yaml'
-- block trainer_args  # Gets overridden
    save_steps: 100
-- endblock

# Solution: Use specific inheritance
-- block trainer_definition
    -- include 'my_trainer_config'  
-- endblock
```

#### Memory Issues
```yaml
# Reduce memory usage
per_device_train_batch_size: 8    # Smaller batches
gradient_accumulation_steps: 4    # Maintain effective batch size
torch_compile: true               # Memory optimization
```

#### Checkpoint Problems
```yaml
# Robust checkpointing
save_strategy: "steps"
save_steps: 500
save_total_limit: 3
resume_from_checkpoint: true      # Auto-discover
```

## Guide Categories

### By Experience Level

#### Beginner Guides
- Basic configuration patterns
- Common setup tasks
- Simple troubleshooting

#### Intermediate Guides  
- Performance optimization
- Multi-GPU training
- Custom components

#### Advanced Guides
- Framework extension
- Production deployment
- Complex distributed setups

### By Topic

#### Configuration
- Template inheritance patterns
- YAML syntax and tags
- Debugging configuration issues

#### Training
- Trainer selection and setup
- Distributed training strategies
- Checkpoint management

#### Performance
- Memory optimization
- Throughput maximization
- Profiling and monitoring

#### Integration
- HuggingFace migration
- External tool integration
- CI/CD pipelines

## Best Practices Summary

### 1. Configuration Management
- Use template inheritance for reusability
- Keep configurations version controlled
- Document custom templates clearly

### 2. Training Workflows
- Start simple, scale gradually
- Enable checkpointing for long runs
- Monitor resource usage

### 3. Development Process
- Test with small models first
- Use consistent naming conventions
- Validate configurations before long runs

### 4. Production Deployment
- Automate testing and validation
- Monitor training progress
- Plan for fault tolerance

## Getting Help

### Self-Service
1. **Check this guides section** for common issues
2. **Search existing issues** in GitHub
3. **Review examples** for similar use cases

### Community Support
1. **GitHub Discussions** for questions and ideas
2. **GitHub Issues** for bugs and feature requests
3. **Documentation feedback** for improvements

### Professional Support
Contact the Forgather team for:
- Enterprise deployment assistance
- Custom feature development
- Training and consulting

---

*Guides are regularly updated based on community feedback and common issues. Suggest improvements in [GitHub issues](https://github.com/anthropics/forgather/issues).*