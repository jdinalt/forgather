# Torch Titan

Forgather integration with Torch Titan

https://github.com/pytorch/torchtitan/

## Basic usage

```
# List configurations
forgather ls

# Show details of preprocessed configuration
forgather [-t CONFIG_NAME] pp

# Train
# This will lauch using "torchrun" with the correct settings
forgather [-t CONFIG_NAME] train

# Start Tensorboard (on another terminal)
# Use "--bind_all," if you need to run on all interfaces, not just localhost
forgather tb [-- --bind_all]
```

## Native Titan

Just uses Forgather for setting up the configuraiton. No custom torchtian code

### Llaam3

Basic demonstration of using Forgather to manage Torch Titan configurations.

This has been tested against mainline, using the nightly PyTorch build. As the support for bfloat16 was added 2 days ago, this will probably not work with the "stable" release.

I haved reproduced two of the Llama3 base configurations, having only modified it them slighlty to use Forgather model output directory and meta-data conventions.

From these base configurations, I have added a few derived configurations to demonstrate how this makes working with Torch Titan configurations easier.

### Forgather Titan

Uses a customized titan trainer sub-class, allowing for dependency injection of trainer assets. e.g. (dataset, tokenizer, collate function, optimizer, lr-scheduler, etc.)

See [README.md](./tiny_titan/README.md)

### Tiny Titan

Reproduces the configuration in "examples/tutorials/tiny_llama"

This includes my original rough-draft configuration, which was used to scope out how to make this work.

There is also a FSDP configuraiton for a 117M parameter Llama3 model.

### Test Parallelisms

Demonstrates how to use various parallelisms and is setup to compare the results of what *should* be eqivalent outcomes.

