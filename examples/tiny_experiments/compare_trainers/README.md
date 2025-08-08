# Trainer Comparison

This project compares different trainer implementations available in Forgather to help understand their performance characteristics and use cases.

## Configurations

- **trainer.yaml** - Default Forgather trainer (forgather.ml.trainer.Trainer)
- **accel_trainer.yaml** - Accelerate-based trainer (forgather.ml.accel_trainer.AccelTrainer)
- **hf_trainer.yaml** - HuggingFace Transformers trainer (transformers.Trainer)

## Trainers Compared

- **Default Trainer** - Basic Forgather trainer implementation
- **Accelerate Trainer** - Multi-GPU trainer using Accelerate framework
- **HuggingFace Trainer** - Integration with HuggingFace Transformers trainer

## Usage

```bash
# List available configurations
forgather ls

# View preprocessed configuration
forgather -t trainer.yaml pp

# Run training comparison
forgather -t trainer.yaml train
forgather -t accel_trainer.yaml train
forgather -t hf_trainer.yaml train
```

## Purpose

This experiment helps determine which trainer implementation is most suitable for different hardware setups and model sizes, comparing Forgather's native trainers with HuggingFace integration.