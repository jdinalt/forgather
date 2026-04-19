# Finetuning Examples

A collection of supervised fine-tuning (SFT) examples that demonstrate how to adapt pretrained language models to instruction-following and conversational tasks using Forgather's `finetune_v2` project template. The examples cover single-GPU and multi-GPU training (DDP, pipeline parallelism), packed-sequence datasets, WSD learning-rate schedules with automatic annealing, and the full model-conversion pipeline from HuggingFace checkpoints into Forgather format.

Each project extends `templatelib/examples/projects/finetune_v2.yaml` and accepts a target model path via the `-M` flag, so the same configuration files work against any compatible base model.

## Projects

- **[Samantha](./samantha/README.md)** - Full tutorial: fine-tune a 7B parameter model on the Samantha conversational dataset. Covers model conversion, pipeline parallelism, packed sequences, multi-node training over Gigabit Ethernet, checkpointing, and inference serving.
- **[Open-Orca](./open-orca/README.md)** - Fine-tune on the Open-Orca reasoning dataset using best-fit sequence packing and an automatically triggered WSD annealing schedule. Natural complement to Samantha: teaches chain-of-thought reasoning rather than conversational persona.
- **[OpenAssistant](./openassistant/README.md)** - Minimal fine-tuning example on the OpenAssistant dataset. Raw configs suitable as a starting point; refer to the Samantha tutorial for detailed setup instructions.
