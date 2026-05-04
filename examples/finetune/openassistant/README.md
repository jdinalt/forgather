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

Measure loss/perplexity on the OpenAssistant test split with `forgather eval`:

```bash
forgather eval test openassistant -M /path/to/model --dtype bfloat16
```

See [docs/guides/evaluating-models.md](../../../guides/evaluating-models.md)
for the full workflow (trainer choices, multi-GPU, pipeline parallelism, ...)
and `forgather eval list` for the other shipped configs.
