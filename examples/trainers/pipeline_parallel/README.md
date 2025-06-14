## Pipeline Paraallel Training

This is presenlty a test harness for the Pipleline Parallel Trainer class.

### References

- https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420  
- https://docs.pytorch.org/docs/stable/distributed.pipelining.html  
- https://arxiv.org/abs/2410.06511  
- https://github.com/pytorch/PiPPy

### Description

The pipeline trainer makes use of the new PyTorch Pipeline Parallelism support added to torch, which was originally in [PiPPy](https://github.com/pytorch/PiPPy).
