#!/bin/bash
CUDA_VISIBLE_DEVICES='0,1' torchrun --standalone --nproc-per-node 'gpu' '../../scripts/train_script.py' -p '.' -s '../../src' "base_config.yaml"
