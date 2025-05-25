import math
from typing import Callable, Iterable, Tuple
import re

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Optimizer

def make_re_multiopt(named_parameters, optimizer_map, factories):
    groups = { group_name: [] for regex, group_name in optimizer_map }
        
    for param_name, param_value in named_parameters:
        for regex, group_name in optimizer_map:
            m = re.search(regex, param_name)
            if m is not None:
                groups[group_name].append((param_name, param_value))
                break
    optimizers = [ factories[group_name](params) for group_name, params in groups.items() ]
    
    return Multiopt(optimizers)

class Multiopt:
    def __init__(
        self,
        optimizers: list,
    ):
        self.optimizers = optimizers
        
    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for opt in self.optimizers:
            opt.step()

        return loss

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()