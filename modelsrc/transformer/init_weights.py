from typing import List, Tuple, Callable, Dict
import re
import math
from torch import nn, Tensor
import torch


# https://arxiv.org/pdf/2407.17465
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


def init_torch_linear_default(weight: Tensor, gain: float = 1.0):
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
    padding_index: int = None,
    std: float = 1.0,
    scale_rsqrt_d_model: bool = True,
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
        std = 1.0 / math.sqrt(weight.shape[1])
    torch.nn.init.normal_(weight, std=std)

    # If pad index, zero that embedding.
    if padding_index is not None:
        weight[padding_index].zero_()


def init_pass(weight):
    pass


@torch.no_grad()
def init_weights_by_regex(
    module: torch.nn.Module,
    regex_list: List[Tuple[str, str]],
    init_f_map: Dict[str, Callable],
    debug: bool = False,
) -> None:
    """
    Initialize model weights, where a regular expression is used to select the initialization

    regex_list: A list of tuples of (regular_expression, init_function_name)
        The first element is a regular expression to match with the parameter name
        The second is a name in "init_f_map," which maps to a Callable

        The expressions are processed in order, stopping at the first match.

    init_f_map: A map of name -> Callable. This is used to lookup the initialization function
        name in "group_map" and then call it on the parameter.

    Example:
        from functools import partial

        regex_list = [
            ( r"bias", "zeros" ),
            ( r"embedding\.weight", "embedding" ),
            ( r"feedforward|attention|output_decoder", "linear" ),
        ]

        init_f_map = {
            "zeros": torch.nn.init.zeros_,
            "embedding": partial(
                init_embeddings,
                padding_index=0,
            ),
            "linear": partial(torch.nn.init.xavier_normal_, gain=alpha)
        }

        init_weights_by_regex(model, regex_list, init_f_map)

    As shown in the example, using partial functions makes it fairly easy to specify
    additional arguments to the initialization functions.

    """
    # Make sure all keys are defined. Makes debugging easier.
    for regex, key in regex_list:
        if key not in init_f_map:
            raise Exception(
                f"Undefined key {key} found in regex list. Add definition to init_f_map"
            )

    for param_name, param_value in module.named_parameters():
        for regex, key in regex_list:
            m = re.search(regex, param_name)
            if m is not None:
                if debug:
                    print(f"calling {key} on {param_name}")
                init_f_map[key](param_value)
                break
