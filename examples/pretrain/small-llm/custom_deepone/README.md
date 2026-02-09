# Custom DeepOne

A customized model project, based on the DeepOne model

We have replaced ALiBi positional encoding with RoPE, primarily because it is better optimized for speed.

We have also added Qwen3-style QK-Norm, expanded the vocabulary size to 32K, and made the model deeper and narrower.