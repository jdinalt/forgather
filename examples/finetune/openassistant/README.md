# OpenAssistant

Finetune a model with OpenAssistant dataset

See: [OpenAssistant Dataset](../../datasets/OpenAssistant/README.md)

For more detailed instructions on finetuning, refer to [Samantha](../samantha/README.md)

The configurations are still pretty raw and could use more tuning, but should work.

## Usage

```bash
# Basic Usage for 7B model on 1 GPU
forgather -t 7b_1gpu.yaml train -M /path/to/model

# Note: The "pipeline" configs will only work with Forgather
# pipeline compatible models. See Samantha for instructions on model conversion.

# 7B model on 2 GPU pipeline
forgather -t 7b_2gpu.yaml train -M /path/to/model

# 7B model on 4 GPU pipeline
forgather -t 7b_4gpu.yaml train -M /path/to/model
```

## Test Trained Model Performance

From examples/evaluate/eval

```bash
# One GPU
forgather train -M /path/to/model --dataset-proj ../../datasets/OpenAssistant/ \
--dataset-config openassistant_packed.yaml --batch-size 4 --dtype bfloat16 \
--attn-implementation flex_attention --max-length 4096 --max-steps 24

# Two GPUs
forgather -t pp_2gpu.yaml train -M /path/to/model --dataset-proj ../../datasets/OpenAssistant/ \
--dataset-config openassistant_packed.yaml --batch-size 8 --dtype bfloat16 \
--attn-implementation flex_attention --max-length 4096 --max-steps 12

# Four GPUs
forgather -t pp_2gpu.yaml train -M /path/to/model --dataset-proj ../../datasets/OpenAssistant/ \
--dataset-config openassistant_packed.yaml --batch-size 16 --dtype bfloat16 \
--attn-implementation flex_attention --max-length 4096 --max-steps 6
```
