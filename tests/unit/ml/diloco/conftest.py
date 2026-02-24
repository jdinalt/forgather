"""Shared fixtures and helpers for DiLoCo unit tests."""

import os

import torch

from forgather.ml.sharded_checkpoint import save_checkpoint as _save_model_checkpoint


def make_initial_checkpoint(state_dict, base_dir, name="initial_checkpoint"):
    """Create a minimal checkpoint (model weights only) at base_dir/name.

    Returns the absolute path to the checkpoint directory.
    """
    ckpt = os.path.join(str(base_dir), name)
    _save_model_checkpoint(ckpt, state_dict, safetensors=True)
    return ckpt
