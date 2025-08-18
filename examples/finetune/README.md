# Finetuning Examples

This is a work-in-progress, demonstrating how to use Forgather for finetuning applications.

At present, I have only added a single dataset, Samantha, which is being used to test various use cases to see what
works, what does not (and needs to be fixed), and how to actually do it.

A few interesting datapoints:
- Using various memory saving techniques, it actually possible to perform full finetuning on a 7B Llama model, with a
  context length of 2048 tokens, on a single 24 GB GPU. This makes use of a custom Adafactor optimizer, gradient
checkpointing, and fused optimizer/backward steps. The performance is even reasonably good.
- With two RTX 4090s, using Pipeline Parallel, the same model can be trained without the above memory saving techniques.
- With 4 RTX 4090s, I was able to train a 30B Llama model -- with memory saving.
- With 6 RTX 4090, a 30B model can be trained with a pretty long context length.

Without NVLink, pipeline parallel is so much faster than FDSP!

There's still quite a bit of work to do, but I figured that this at least provides some guidance on how to make this
work.

Some of the examples are broken. I have added notes to the config, where applicable.
