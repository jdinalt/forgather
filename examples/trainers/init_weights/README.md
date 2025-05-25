## Test Various Weight Initialization Methods

### Control

This uses the standard PyTorch initializaiton methods for Linear and Embedding layers.

Torch uses code equivalent to the followning for initializing linear layers:

```python
stdv = 1. / math.sqrt(self.weight.size(1))
self.weight.data.uniform_(-stdv, stdv)
```

See interesting discussions about this method:

https://github.com/pytorch/pytorch/issues/57109
https://soumith.ch/files/20141213_gplus_nninit_discussion.htm

### Regex

This is the same initialization as "Control," but it uses regular-expressions to control how the parameters are initialized.

This is more complex, but far more flexible.

### Xavier Uniform

Here. we use regex again, but use it to replace the torch default init with Xavier Unifrom initializaiton.

This performs relatively poorly.

### Xavier Uniform No Feedforward

This is the same as Xavier Unifrom, except for the feedforward layers, which are initialized with the "torch" method.

This demonstrates that the primary issue is with using fan-out to compute the scaling-factor, where fan-out is 4x fan-in.

Note that both methods are effectively the same for symetric matrices, like those used by the attention layers.

The only difference in this case is with the initialization of the output layers.

### Deepnet

DeepNet: Scaling Transformers to 1,000 Layers  
https://arxiv.org/pdf/2203.00555

Here we try using the method described in the above paper. Among the changes, this rescales both the feedforward initialization and that of the 
attention value and output layers by "beta," which is computed from the number of transformer layers and scales the residuals by "alpha," 
also dervived from the number of layers.

Even though this is using Xavier Uniform initializaiton, this performs on-par with the control, thus not showing the issue identified when 
testing with a simple Xavier Unifrom initialization.

### Deepnet Init

This uses the deepnet initialization method, but not the residual scaling factor. Performance is close to the other good methods.

### Deepnet Torch

This is the same as Deepnet, but we replace Xavier Uniform with the "Torch" method. Again, similar performance.

