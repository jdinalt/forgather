from torch.optim.lr_scheduler import SequentialLR

def sequential_lr_factory(optimizer, schedulers, milestones, last_epoch=-1):
    """
        A helper for late construction of a torch SequentialLR, where
        "optimizer" is forwarded to a list of partial functions for 
        constructing LR schedulers.

        See: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html
    """
    return SequentialLR(
        optimizer=optimizer,
        schedulers=[f(optimizer) for f in schedulers],
        milestones=milestones,
        last_epoch=last_epoch,
    )