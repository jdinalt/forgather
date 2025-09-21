# Torch Titan

https://github.com/pytorch/torchtitan/


## Llama3

Basic demonstration of using Forgather to manage Torch Titan configurations.

This has been tested against mainline, using the nightly PyTorch build. As the support for bfloat16 was added 2 days ago, this will probably not work with the "stable" release.

I haved reproduced two of the Llama3 base configurations, having only modified it them slighlty to use Forgather model output directory and meta-data conventions.

From these base configurations, I have added a few derived configurations to demonstrate how this makes working with Torch Titan configurations easier.

## Tiny Titan

A new Forgather, Torch Titan based, trainer. I have written an initial test configuration, which reproduces the Tiny Llama project with Torch Titan, using Forgather assets (dataset, tokenizer, collate function, optimizer, lr-scheduler, etc.)

See [README.md](./tiny_titan/README.md)