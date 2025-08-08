## Pipeline Parallel Llama

Test project for finetuning a Llama 7B model using pipeline parallelism with 4 GPUs

This has been tested with "meta-llama--Llama-2-7b-hf"

The RoPE positional encoder used by Llama models greatly complicates using PyTorch's APIs for automatically splitting a model for Pipeline Parallelism. The HF LLama model implementation passes a shared set of cached RoPE positional embeddings to each layer through the the layer's "forward" method and these shared buffers are used by each layer's attention module to encode relative positions. Splitting the model causes a number of issues.

1. The HF positional encoder implementation uses a mixed-precision context manager, where the computations are always performed with 32-bit precision. This context manager is not supported by Torch Export, which is used for model tracing.
2. The positional encoder uses a "no_grad" context manager in the positional encoder. In theory, this is supported, but it appears to expose a bug in the model splitting code, for which I have filed a bug on.
3. Assuming the other two issues did not exist, the shared cache buffer would exist in the first layer and would be transmitted to each pipeline stage requiring the buffer -- usually, all of them. Aside from the inefficiency this creates, trying to do so exposed yet another bug. A two stage pipeline configured like this works, but if there are more than two stages, an assert is tripped. The issue is that pipeline code assumes that any tensor sent via "forward" requires a matching gradient in the backward pass. More specifically, there is a comment which states that it is the receiver's responsibility to discard the unused gradient, if it is not needed. This works just like that with two pipeline stages, but with three or more, the code assumes that the sender needs to accumulate gradients from multiple sources. This is not supported and everything comes to a grinding halt at an assert. Very annoying.

Torch Titan, which is the PyTorch reference implementation for Pipeline Parallelism addresses these issues by using manual model splitting. Essentially, each stage starts with an identical, complete model, then deletes the layers that are not required for that stage. The model code is written to conditionally skip modules which are not present. All copies retain the shared RoPE buffers and pass their individual copies of those buffers to each layer on the forward pass. As the buffers are not being sent from one stage to another, the backward pass issue is not an issue.

This approach works, but it requires very close collaboration between the implementation of the model and the trainer with respect to the conventions of how data is passed through the model stages. Everything is a special-case of a special-case for this exact implementation, which makes everything relatively brittle and unlikely to work with anything else, without a fair amount of modification.

My approach to the problem is to define a Llama compatible model which does not trigger any of the issues described above. Such a model can be automatically split and trained, but requires that the model weights be exported to the new model implementation before training and imported back to HF compatible weights, if you want to use the HF implementation again.

To address the RoPE buffer issue, we create a single instance of the RoPE encoder module which is shared by all layers. That is, superficially, it looks like there are as many copies of this module as there are layers, but all layers on the same pipeline stage have tied weights, so there is only a single copy of the buffers on each device. This avoids the need to send the buffers as an argument to the forward method and, in theory, should have the same performance as the original approach. We obviously also avoid using the troublesome context managers.

Problem solved! If only it were that simple... It would seem that while the Torch pipeline model splitting code tries to handle shared buffers correctly, there are bugs.

1. Under some undetermined circumstances (non-deterministic), some of the shared buffers become plain attributes of their parent modules, where they are not in the list of registered buffers (_buffers). We first construct the model on the meta device, then convert the meta tensors to real tensors on the target device via .to_empty(). The buffers which have lost their status as being registered buffers remain on the meta device and hilarity ensues on the first forward pass. The root cause is still TBD.
2. When splitting the model, a bug in the splitting code creates "vestigial" buffers. For example, if the model has 8 layers and is split it two, both split sub-modules will have a FQN like "model.layer.7.attention.rope.sin_cached" The vestigial buffer is not referenced by anything in the first half, yet the duplicate FQN creates a conflict when saving and loading the model's state dictionary. They both think that they own it clobber each others data when saving.

For now, the pipeline trainer explicitly handles both of these issues. It fixes the disowned buffers, adding them back as buffers, and removes the vestigial duplicate buffers. And it works.

```bash
# Convert pre-trained Llama model to Forgather Dynamic Llama model
# from "scripts" directory
# Usage: usage: convert_llama.py [-h] [--reverse] [--dtype DTYPE] [--device DEVICE] [-g] [--prompt PROMPT] [--debug-params] src_model_path dst_model_path
python3 convert_llama.py --device 'cuda:0' --dtype bfloat16 -g ~/models/meta-llama--Llama-2-7b-hf/ ~/models/llama-2-7b-fg/
```

Note that when converted to bfloat16, the logits don't quite match that of the HF model -- this is the reason they added the 32-bit mixed precision code to the implementation. Addressing this is on my "todo" list.

Remember to change the path in the project config to point to where your converted model is located!

Train on 4 GPUs

```bash
forgather -t gpipe_4gpu.yaml train
```

Convert model back to HF model

```bash
python3 convert_llama.py --reverse --device 'cuda:0' --dtype bfloat16 -g ~/models/llama-2-7b-fg/ ~/models/finetuned-Llama-2-7b-hf/
```

At present, this works as a proof-of-concept, but still needs considerable refinement. This is ongoing.

### Issues

The appears to be a persistant CPU memory leak. The cause is TBD.

It would appear that activation checkpointing does not work with DPP at present. Hopefully this is something we can get working!

The fixed batch introduces a few difficulties. In theory, this can be partially mitigated with Flex Attention, which is on the TODO list.

vbvz_4gpu.yaml: Presently does not support a separate inference model, which makes evalution slow. Otherwise, works very well.
zero_bubble_4gpu.yaml: This works reasonably well, except for a non-deterministic jump in CUDA memory when using the separate inference model. As a work around, it's using the train model for inference, which is slow.
gpipe_4gpu.yaml: Fairly reliable, but slow and uses more memory than most others. The batch size has been reduced to prevent OOM issues.
looped_bfs_4gpu: It works, but it's not very fast and, like GPipe, had to have the batch size reduced to avoid OOM issues.
1f1b.yaml: Not the fastest, but has the lowest memory utilization. It's also pretty reliable.
interleaved_1f1b_4gpu.yaml: Faster than 1f1b, but slighlty higher memory usage. It's pretty stable, so it's the default implementation at present.

### Testing Hardware Config

These configurations were tested with a single node using 4 RTX 4090 cards, 256 GB of DRAM, and an AMD Ryzen Threadripper PRO 5955WX 16-Cores.

I don't have a setup to easily test a multi-node configuration. If anyone has such a setup to test on, I would welcome feedback. This would likely entail a bit of work for setting up communications.