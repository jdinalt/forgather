# Tiny Experiments

A collection of example ML projects built around the Forgather ML library

This started as a template project built around the [Tiny Stories](https://arxiv.org/abs/2305.07759) dataset, as I have found it to be great for testing the performance of very small transformer models.

Given a model with about 4M parameters and about 2 minutes of training time on a single RTX 4090 GPU, the trained model models can already generate mostly coherent output.

This has subsequenlty evolved into a family of projects which serve as both examples of how to use Forgather and as tests for the library itself.

## Project Index

### activation_checkpoint

This demonstrates how to use activation checkpointing to trade compute for memory. The notebook includes example code for measuring memory utilization.

### compare_tokenizers

This project demonstrates how to swap tokenizers in a project configuration.

### compare_trainers

In this project, we demonstrate how we change "trainer" implementations with very little effort.

### deepnet

Demonstrates [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)

### flash_attention

Test Flash Attention 2 from both the official library and the PyTorch implementation.

### optimizers

Compares the performance of various optimizers.

### pipeline_parallel

Demonstrates how to use Forgather's Pipeline Parallel Trainer implementation.

### template_project

A starting point for other "Tiney Experiments" projects.

### tiny_models

Compares tiny: gpt2, llama, and forgather-dynamic-transformer models.

## Other

### tiny_templates

Contains the common templates used by this workspace.

### forgather_workspace

This is a special directory which is contains the common meta-configuration for all of the subprojects.