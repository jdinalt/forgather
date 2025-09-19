# Torch Titan

https://github.com/pytorch/torchtitan/

Basic demonstration of using Forgather to manage Torch Titan configurations.

This has been tested against mainline, using the nightly PyTorch build. As the support for bfloat16 was added 2 days ago, this will probably not work with the "stable" release.

I haved reproduced two of the Llama3 base configurations, having only modified it them slighlty to use Forgather model output directory and meta-data conventions.

From these base configurations, I have added a few derived configurations to demonstrate how this makes working with Torch Titan configurations easier.

When I have a chance, I'll see about creating a custom trainer, which can use Forgather's models, datasets, optimizers, etc. This would be super-useful!