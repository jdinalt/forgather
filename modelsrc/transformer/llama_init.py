import torch
from torch import nn, Tensor

"""
    Llama weight init based upon:
        https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/model/model.py

    I was unable to find a primary source for what was actually used. Given than the source is Meta, 
    it seems plausible that this may be accurate.

    We also provide an implementation which matches that used by the HF Llama model.
"""


def init_output_layer(weight: Tensor, d_model: int) -> None:
    """
    Output layer initialization
    """
    std = d_model**-0.5
    cutoff_factor = 3

    nn.init.trunc_normal_(
        weight,
        mean=0.0,
        std=std,
        a=-cutoff_factor * std,
        b=cutoff_factor * std,
    )


def trunc_normal_magic(weight: Tensor) -> None:
    """
    Init with truncated normal with magic std of 0.02

    How was this number arrived at?
    """
    nn.init.trunc_normal_(weight, mean=0.0, std=0.02)


def trunc_normal(weight: Tensor, std: float) -> None:
    """
    Init with explicit std
    See llama_std() and llama_std_depth()
    """
    nn.init.trunc_normal_(weight, mean=0.0, std=std)


def llama_std(n_layers: int) -> float:
    """
    Standard deviation, where not using magic number
    Used in attention and feedforward
    """
    return 0.02 / (2 * n_layers) ** 0.5


def llama_std_depth(layer_id: int) -> float:
    """
    Alternative std, which varies with layer depth
    """
    return 0.02 / (2 * (layer_id + 1)) ** 0.5


def init_embeddings(weight: Tensor) -> None:
    """
    Init embeddings
    Note that this is not scaled by 1/sqrt(d_model)
    """
    nn.init.normal_(weight)


def hf_llama_weight_init(
    model: nn.Module,
    std: float = 0.02,
) -> None:
    """
    Use same initialization as HF Llama model.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
