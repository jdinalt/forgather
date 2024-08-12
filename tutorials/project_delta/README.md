# Tiny Generative Language Model

This project demonstrates how to use the templates library to construct a tiny causal transformer.

Most of the remaining examples make use of the Tiny Stories dataset, as it allows one to quickly train a relatively small transformer model (< 10M parameters), which can generate relatively coherent speech.

- Dataset: datasets/tiny/tiny_stories_abridged.yaml
    - Dataset ID: roneneldan/TinyStories
    - Reference: https://arxiv.org/abs/2305.07759

Unlike the previous examples, the project meta-config now makes use of the templates library, thus many more templates are now available to the project.

The project configuration itself is now even derived from a common "Tiny Experiments" project template, which defines defaults for a number of projects with similar setups. See "projects/tiny.yaml."

---