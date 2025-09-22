# Compare Parallelisms

The "control" configuration is a fairly simple configuration, with one GPU, and 10 gradient accumulation steps, over 500 steps.

The others use various parallel strateges, but the same effective batch size (320). In theory, this should produce reasonably comparable results.

---
I confirmed that torchtitan has the same bug that I found in my own pipeline parallel implementation.

```
commit 1c60d29d1955a7270d680291e40825017d2c63dd
Author: Jason dinAlt <joedkloss@gmail.com>
Date:   Sun Sep 14 03:53:13 2025 +0000

    Minimal working Pipeline grad accumulation
    
    Implemented gradient accumulation in pipeline trainer.
    
    It seems that the
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/stage.py#L567
    (scale_grads()) automatically scales grads by the number of microbatches
    by dividing the gradients by N. This is completely broken when using
    gradient accumulation, as this causes the total accumulated gradient to
    be rescaled by this factor at each step, resulting in the total
    gradient, at the end of gradient accumulation, being too small.
    
    There is a scheduler option, "scale_grads," which defaults to True,
    which controls this. As we already need to wrap the loss function to
    scale the gradients by the number of accumulation steps, we can resolve
    this by disabling "scale_grads" and including the number of microbatches
    in the total scale factor.
    
    When compared to the equivalent test run, without gradient accumulation,
    but not Pipeline Parallel, this prodcues results much closer to what is
    expected.
    
    Skimming over the Torch Titan code, it looks like they probably have
    this bug, where gradient accumulation is not computed correctly. I only
    noticed, as I have a reference run to compare against.
    
    I'll try to confirm if this bug is present and offer a PR, if found to
    be the case.
```

I have created a PR and am setting up for submitting it.

I have applied the required change to my sub-class of Titan's Trainer class. As I only override
the __init__ method, it require the rest of the changes to torchtitan to actually work.

For now, I have added a temporary "disable_pp_fix" argument to my trainer constructor, which defaults to,
true, keeping the original problem as-is. This is important for reproducing the issue using my version.

When testing the fix, using Forgather's trainer, set this flag to "False,"

Note: The bug can be reproduced with just Torch Titan, and the fix also works fine without my code.

I'm just setting this up, as it's a convinient way to reprocude the issue and test the fix.

As I have already used the same approach in fixing my own PP trainer, I can confirm that it works fine
with other schedulers as well (not just 1F1B).

## Reproducing the Issue

Obviously, if reproducing the issue with Forgather, you will need to install it first...

From this project directory, run:

```
forgather -t control.yaml train
forgather -t pp.yaml train
```

These will log the results to Tensorboard, which can be started like this:

```
forgather tb [-- <TB options>]
```

## Testing the fix

Apply the changes from my PR to torchtitan, then modify the "pp.yaml" to enable the fix.

```
...
[trainer]
    == super()
    disable_pp_fix: False
```

Then run training again.

```
forgather -t pp.yaml train
```

The control run and the pp run with the fix are idential. Without the fix, the loss is way off and the gradient-norms are way too small.