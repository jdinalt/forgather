#!/bin/bash
torchrun --standalone --nproc-per-node gpu '../scripts/train_script.py' -w 'forgather_demo/whitelist.yaml' -I 'forgather_demo' -I '../templates' -I '../model_zoo'  -s '..' "${@}"
