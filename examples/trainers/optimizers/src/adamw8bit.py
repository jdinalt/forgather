import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version
from .subspace_proj import SubspaceProjector

from bitsandbytes.optim.optimizer import Optimizer2State

import torch

class AdamW8bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, optim_bits=32,args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        super().__init__( "adam", params, lr, betas, eps, weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        overflows = []

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        #if self.is_paged: self.page_mng.prefetch_all()
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                
                # GaLore Projection
                 # Projection
                if "proj_args" in group:
                    if "projector" not in state:
                        state["projector"] = Projector(
                            p.grad,
                            **group["proj_args"],
                        )
                    else:
                        state["projector"].update(p.grad)

                    if 'weight_decay' in group and group['weight_decay'] > 0:
                        # ensure that the weight decay is not applied to the norm grad
                        group['weight_decay_saved'] = group['weight_decay']
                        group['weight_decay'] = 0
                    
                    grad = state["projector"].down(p.grad)
                    
                    # suboptimal implementation
                    p.saved_data = p.data.clone()
                    p.data = grad.clone().to(p.data.dtype).to(p.data.device)
                    p.data.zero_()
                    p.grad = grad

                if 'state1' not in state:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                torch.cuda.synchronize()
                
               # Project up
                if "proj_args" in group:
                    p.data = p.saved_data.add_(state["projector"].up(p.data))  
                    
                    # apply weight decay
                    if 'weight_decay_saved' in group:
                        p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay_saved'])
                        group['weight_decay'] = group['weight_decay_saved']
                        del group['weight_decay_saved']
                
        if self.is_paged:
            # all paged operation are asynchronous, we need
            # to sync to make sure all tensors are in the right state
            torch.cuda.synchronize()


        return loss
