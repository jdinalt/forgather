from typing import List, Tuple, Callable, Dict
import re
import math
from torch import nn, Tensor
import torch

@torch.no_grad()
def simple_weight_init(model: nn.Module) -> None:
    """
        Simple and reasonable init for LLMs
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            init_torch_linear_default(module.weight)
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            init_embeddings(module.weight, module.padding_idx)

def init_torch_linear_default(weight: Tensor, gain: float=1.0):
    """
        This is the torch default for nn.Linear()
        This appears to be a He init variant, which always uses
        fan-in and does not try to compensate for the activation.

        You can use the "gain" parameter for compensating for the 
        activation.

        Note: torch has a rather unusual implementation. See:
        https://github.com/pytorch/pytorch/issues/57109
        https://soumith.ch/files/20141213_gplus_nninit_discussion.htm
    """
    in_features = weight.shape[1]
    uniform_range = gain / math.sqrt(in_features)
    torch.nn.init.uniform_(weight, -uniform_range, uniform_range)

@torch.no_grad()
def init_embeddings(
    weight: Tensor,
    padding_index: int=None,
    std: float=1.0,
    scale_rsqrt_d_model: bool=True,
):
    """
        Simple embedding init.
        
        Zeros out the pad embedding.
        If scale_rsqrt_d_model == True, then std = 1 / sqrt(d_model), as per
            per Attention is All you Need.
        Note: When scale_rsqrt_d_model == True, the input encoder should scale the embedding outputs
            by sqrt(d_model).
    """
    if scale_rsqrt_d_model:
        std = 1. / math.sqrt(weight.shape[1])
    torch.nn.init.normal_(weight, std=std)

    # If pad index, zero that embedding.
    if padding_index is not None:
        weight[padding_index].zero_()
