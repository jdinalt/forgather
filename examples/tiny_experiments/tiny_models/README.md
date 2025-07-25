## Tiny Models

A collection of tiny models to train on the Tiny Stories dataset with the tiny_stories_2k tokenizer.

This allows for direct comparison of model archetectures.

### Featuring
- Tiny Causal Transformer -- an example custom transformer model
- Tiny Llama
- Tiny GPT2

### Common Configuration
- Tokenizer: tokenizers/tiny_2k_bpe.yaml
    - Vocabulary Size: 2000
    - Maximum Model Sequence: 2048
- Dataset: datasets/tiny/tiny_stories_abridged.yaml
    - Dataset ID: roneneldan/TinyStories
    - Reference: https://arxiv.org/abs/2305.07759
    - Train Select Range: 10% 
- Model:
    - Model Dimension: 256
    - MLP Dimension: 1024
    - Layers: 4
    - Heads: 2
    - All Dropout Probabilities: 0.0
- Trainer:
    - Class: aiws.trainer.Trainer
    - Epochs: 1
    - Learning Rate: 1.0e-3
    - Batch Size: 16