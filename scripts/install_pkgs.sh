#!/bin/bash

# Only run in remote environments
if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
  exit 0
fi

set -e

VENV_DIR="$CLAUDE_PROJECT_DIR/.venv"

# Create venv using Python 3.12 if it doesn't already exist
if [ ! -d "$VENV_DIR" ]; then
    /usr/bin/python3.12 -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"

# Install CPU-only torch first to avoid downloading CUDA wheels
if ! "$PIP" show torch > /dev/null 2>&1; then
    "$PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install forgather and remaining dependencies
if ! "$PIP" show forgather > /dev/null 2>&1; then
    "$PIP" install -e "$CLAUDE_PROJECT_DIR"
fi

# Persist venv on PATH for subsequent Bash commands in this session
if [ -n "$CLAUDE_ENV_FILE" ]; then
    echo "PATH=$VENV_DIR/bin:$PATH" >> "$CLAUDE_ENV_FILE"
    echo "VIRTUAL_ENV=$VENV_DIR" >> "$CLAUDE_ENV_FILE"
fi

exit 0
