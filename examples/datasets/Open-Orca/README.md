# OpenOrca

https://huggingface.co/datasets/Open-Orca/OpenOrca

> OpenOrca is a collection of augmented FLAN data containing approximately 1M GPT-4 completions and 3.2M GPT-3.5 completions, designed to align with the distributions outlined in the Orca paper. The data augments FLAN Collection questions with detailed step-by-step reasoning traces from GPT-3.5 and GPT-4.

## Configurations

- [openorca.yaml](./templatelib/configs/openorca.yaml) Open Orca, one conversation per example
- [openorca-packed.yaml](./templatelib/configs/openorca-packed.yaml) Open Orca with sequence packing for higher GPU utilization
