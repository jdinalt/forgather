#!/bin/bash

# The number of concurrent processes to use: 'gpu' or integer
NPROCS="1"

# What is the path to the package root?
PACKAGE_ROOT="../.."

# Path to training script
TRAIN_SCRIPT="${PACKAGE_ROOT}/scripts/train_script.py"

# Project templates
PROJECT_INCLUDES="templates/"

# Model templates
MODEL_INCLUDES="${PACKAGE_ROOT}/model_zoo/"

# Common templates.
COMMON_INCLUDES="${PACKAGE_ROOT}/templates/"

# Project whitelist; controls which Python objects types can be created by the config. script.
WHITELIST="templates/project/whitelist.yaml"

torchrun --standalone --nproc-per-node "${NPROCS}" "${TRAIN_SCRIPT}" -w "${WHITELIST}" \
    -I "${PROJECT_INCLUDES}" -I "${MODEL_INCLUDES}" -s "${PACKAGE_ROOT}" \
    -I "${COMMON_INCLUDES}" "$@"