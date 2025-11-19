from torch.nn.functional import cross_entropy
from torch import Tensor, FloatTensor, LongTensor
import torch

def _causal_loss_fn(logits: FloatTensor, labels: LongTensor) -> FloatTensor:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            # labels with this value are ignored when computing loss
            ignore_index=-100,
            reduction="mean",
        )

        return loss

class CausalLoss:
    def __init__(self, compile=False):
        super().__init__()
        if compile:
            self.loss_fn = torch.compile(_causal_loss_fn)
        else:
            self.loss_fn = _causal_loss_fn
        self.compile = compile
    
    def __repr__(self):
        return f"{type(self).__name__}(compile={self.compile})"

    def __call__(self, logits: FloatTensor, labels: LongTensor) -> FloatTensor:
        return self.loss_fn(logits, labels)
    
