from torch.optim import AdamW
from aiws.dotdict import DotDict

# From https://huggingface.co/blog/codeparrot
def get_grouped_params(model, args, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay): params_without_wd.append(p)
        else: params_with_wd.append(p)
    return [{"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},]


# An AdamW optimizer factory
class OptiFactory:
    def __init__(self, weight_decay=0.01, **kwargs):
        kwargs['weight_decay'] = weight_decay
        self.adam_args = DotDict(kwargs)
    
    def __call__(self, model, training_args):
        return AdamW(get_grouped_params(model, self.adam_args), lr=training_args.learning_rate, **self.adam_args)