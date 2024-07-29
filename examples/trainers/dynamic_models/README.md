## Dynamic Models

This is a demonstraction of how to perform model archetecture experiments by using the configuration system to dynamically change module types.

As most of the examples, we use "Tiny Causal" as a baseline, then make various changes for comparison.

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
    - Initial Learning Rate: 1.0e-3
    - Train Batch Size: 32
    - LR Sheduler: Cosine