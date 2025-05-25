from typing import List, Tuple, Callable, Dict
import re
import math
import torch

def init_pass(weight):
    pass

@torch.no_grad()
def init_weights_by_regex(
    module: torch.nn.Module,
    regex_list: List[Tuple[str, str]],
    init_f_map: Dict[str, Callable],
    debug: bool = False
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
            raise Exception(f"Undefined key {key} found in regex list. Add definition to init_f_map")
    
    for param_name, param_value in module.named_parameters():
        for regex, key in regex_list:
            m = re.search(regex, param_name)
            if m is not None:
                if debug:
                    print(f"calling {key} on {param_name}")
                init_f_map[key](param_value)
                break