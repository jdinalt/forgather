# Base LM Project

A test harness for the base LM project template.

Note that the default dataset is fairly large and can take a considerable amount of time to download on the first run.0

```bash
forgather [-t CONFIG] train --help
```

- [Main Documentation](../../docs/project-templates/lm-training-projects.md)
- [lm_training_project.yaml](../../templatelib/examples/projects/lm_training_project.yaml)
- [finetune_v2.yaml](../../templatelib/examples/projects/finetune_v2.yaml)
- [lm_callbacks.yaml](../../templatelib/examples/callbacks/lm_callbacks.yaml)

## Test LM Training Project Template

```bash
# With defaults  -- full float32 precision
forgather train

# For a quick test, without saving checkpoints.
forgather train --max-steps 200 --save-strategy no

# Test with the addition of standard trainer callbacks
forgather -t lm_callbacks.yaml train

# Train in mixed-precision (bf16), with compile, and Ampere or later GPU
# Slower startup, but _much_ faster!
forgather train --compile true --mixed-precision bf16 --float32-matmul-precision high

# Test with Distributed Data Parallel (DDP), 2 GPUs
forgather -t ddp.yaml train

# FSDP2 (fully_shard) -- parameters, gradients and optimizer state are sharded
# across the data-parallel mesh. 2 GPUs by default.
forgather -t fsdp2.yaml train

# Pipeline Parallel -- model is split accorss GPUs, 2 by default, where activations/gradients flow between stages.

forgather -t pp.yaml train

# Train with zero-bubble-v scheduler.
forgather -t pp.yaml train --pipeline-schedule ScheduleZBVZeroBubble

# Test the finetune template with model from pretrain/small-llm project using DDP
# The default dataset is Samantha, so this will produce a micro-Samantha.

# First produce a clean fine-tuning handoff directory: graft on the ChatML
# tokens, install the chat template, and synthesize a generation_config.json
# whose eos_token_id list contains both the original EOS and <|im_end|>.
forgather finalize ../pretrain/small-llm/output_models/sllm ../pretrain/small-llm/output_models/sllm_chat \
    --add-tokens ../../add_tokens_config/chatml.yaml \
    -t ../../chat_templates/chatml.jinja

# Then train the model for 3 epochs on Samantha
forgather -t finetune_v2.yaml --model-id-or-path ../pretrain/small-llm/output_models/sllm_chat --epochs 3 --trainer-type ddp
```
