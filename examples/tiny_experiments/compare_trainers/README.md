# Trainer Comparison

This project compares different trainer implementations available in Forgather to help understand their performance characteristics and use cases.

## Configurations

- **trainer.yaml** - Default Forgather trainer (forgather.ml.trainer.Trainer)
- **accel_trainer.yaml** - Accelerate-based trainer (forgather.ml.accel_trainer.AccelTrainer) 1 GPU
- **accel_trainer_ddp.yaml** - Accelerate-based trainer N GPUs via DDP
- **hf_trainer.yaml** - HuggingFace Transformers trainer (transformers.Trainer) 1 GPU
- **hf_trainer_ddp.yaml** - HuggingFace Transformers trainer (transformers.Trainer) N GPUs via DDP

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
forgather -t accel_trainer_ddp.yaml train
forgather -t hf_trainer.yaml train
forgather -t hf_trainer_ddp.yaml train
```

For the DDP variants, if you have more than 2 GPUs and wish to limit training to a subset of those GPUs, you can use the '-d' argument to specify which to use:

```bash
# Only train on GPUs 0 and 1
forgather -t accel_trainer_ddp.yaml train -d 0,1
```

## Purpose

This experiment helps determine which trainer implementation is most suitable for different hardware setups and model sizes, comparing Forgather's native trainers with HuggingFace integration.

This also serves as an integration test for the trainer implementations.