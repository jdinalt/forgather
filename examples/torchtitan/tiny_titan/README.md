# Tiny Titan

This my first shot at creating a native Torch Titain trainer, which uses dependency injection for most the majority of the training assets. This is opposed to the torchtitan approach of giving the passing a configuration into the trainer and having it construct all of the assets internally from the configuration.

## Why?

The dependency injection approach has a number of advantages. The biggest is that the coupling between components is much looser, in that nothing requires explicitly defined support, as long as the interfaces are compatible. For example, I can inject any optimizer I want, without ever having to add explcit support for it in torchtitan. There's no longer any need for pre-registering components to be able to use them. You just construct the object and inject it into the trainer. Being able to do this through configuration files was one of the major driving forces for writing Forgather in the first place.

This is also an advantage, even outside of the context of Forgather. If I am constructing objects for my trainer in native Python, why should I not be able to just pass in a constructor for a model or an optimizer? Why can't I just build the dataloader myself and pass it into the trainer? I don't need the trainer to do these things for me. It just gets in the way and make the task more difficult. Doubly so, when you want to construct an object which does not have support. Say for example, constructing an Adafactor optimizer.

## Details

I have created a sub-class of torchtitain.train.Trainer and have completley overrided the constructor to take a collection of injected assets. The JobConfig is still needed, even though we ignore large parts of it, because the JobConfig class is so entagled with the titan ecosystem.

As to test it, I have written a very raw config file, no templates, which reproduces something resembling the configuration of the Tiny Llama tutorial training project. This is both a proof-of-concept and will serve for reference, when refactoring the configuration into reusable templates. I'll probably keep this raw configuration around after refactoring, as this process serves as an example of how to write Forgather configurtions for something completely new.


### Approach 
- Keep it simple
- Don't try to use fancy templating features while you are still trying to understand the required structure. It's only when you find that you are duplicaitng code that you should consider refactoring.
- No template blocks are defined.
- Minimize the use of pre-processing features.
- We use YAML anchors and aliases to define variables that we use more than once and for complex data structures.
- We import base directories to get the location of the Forgather directory, as we need that to find our assets.

### Distributed Enviornment

The original Titan Trainer initializes torch distributed direclty in the class constructor. It does not both to check if it is already initialized, so if someone has already done so, an exception is raised. We may need to distributed enviornment to be initialized *before* constructing the trainer, so we use the same solution Forgather normally uses to solve this problem.

We define a singleton instance of DistributedEnvironment in the config and make anythig which depends upon torch distributed being initialized depend upon this object, which ensures that it is constructed (and torch distributed initialized) before it is needed.

DistributedEnvironment() also checks if torch distributed has already been initialized and just reads the config, if alreayd set up.

The code in my implementation is very similar to what Titan does. It's missing some features, like supporting non-cuda acceelerators, it's a Protocol. You can replace it with one that does whatever your environment requires -- another benifit of our weak coupling approach.

### Tokenizer

We import the Tiny Stories 2K BPE tokenizer sub-project to construct the tokenizer. If the tokenizer does not already exist, the infrastructure will download the required dataset and build the tokenizer first.

If you look at the configuration, you will notice that we actually use the tokenizer parameters for defining the model configuration, pulling details like the vocabulary size, max model lenth, and EOS token direclty from the tokenizer instance. This ensures that they are consistent with the model definition.

### Dataset

We import the predefined Tiny Stories dataset, using the "iterable" version, as torchtitan does not appear to support mapped datasets. You should be able to use any other Forgathe dataset definition as well, as long as it is an IterableDataset.

### Dataloader

We directly construct an instance of Titan's ParallelAwareDataloader and pass this to the trainer. This is complicated by the fact that it expects the arguments "dp_rank" and "dp_degree." The official Titan Trainer implementaiton computes these values in the constructor from both the internally constructed ParallelDims and FTManager instances. Thus, to construct this from outside of the Trainer, we need to compute these values, which depend upon both of these objects.

To make this somewhat less ugly, I sub-classed both ParallelDims and FTManager, with additional properties, to compute the required values. Then we define instances of our derived classes and get the computed values, which are then passed as arugments -- this also makes ExtendedFTManager depnend upon ExtendedParallelDims, as the former requires computed values from the latter.

I think this aspect of Titan needs some work (it's a mess). I'll see if I can help, when I have a few cycles.

### Data Collator

I initially tried using Titan's default collator, but it barfs when passed sequences of varrying lengths. Our dataset preprocessing functions expect the data collator to pad examples when this happens, but I guess Titan must be doing this at the preprocessing stage.

We use the default Forgather data-collator. As Titan assumes somewhat different convention, regarding what is yielded by the dataiterator, I added a couple of new options to the class to help. You can explicilty set the name of the primary input ("input" in Titan vs. "input_ids") and the name of the labels, which can be None. In the case of setting this to None, the collator returns a:

```
Tuple[Dict[str, Tensor], Tensor]
```

This is the format that Titan expects.

### Loss Function

We use the standard Forgather causal loss function, rather than building one using Titan's "build_loss_function()," as their assumptions differ from ours, which use Huggingface conventions, where the labels returned by the datacollator are not shifted and the loss function must do this.

It should be possible to 

### Model

The model is defined directly in the configuration. It's still a Torch Titan Llama3 model, but we don't use the regular infrastructure for specifying a model variant. Instead, we directly construct the model args and define a partial function for constructing the model instance.

I tried to keep the configuration consistent with the Tiny Llama definition we use in our tutorial.

As noted above, we import some of the model arguments from the tokenizer instance.

Regarding using Forgather models. I will be looking into this shortly, but I'll probably have to make some changes to the model lib to support the expected arguments / return values and make the changes so that modules can be deleted for pipeline splitting. Our own pipeline splitter uses torch.export(), so does not depend upon any particual model structure to work.

### parallelize_fn and pipeline_fn

We just directly specify the functions and pass them into the Trainer constructor.

### Optimizer

You should now be able to use any optimizer that you want, without having to sub-class anything or special-case anything. We use the same approach we use in other Forgather projects, where the optmizer is defined as a partial function, which takes the model's parameters as an argument. This provides a uniform abstraction to the trainer, while still allowing one to set optimizer specific arguments.

Previously, when Titan has multiple "model_parts," it would create an optimizer instance, of the same class, for each instance and wrap these in an optimizer container, which exports an "optimizer" protocol and hides the fact that it's composed of multiple sub-optimizers.

I just used the same approach that I did in Pipeline Trainer, which is to create a unified dictionary of all sub-modules parameters, and pass this to the optmizer partial function.

This *should* work, as long as all the FQNs are unique; something which I have yet to verify, so hopefully this is the case.

### LR Scheduler

We use the same approach as the optimizer. It's a partial function, which is passed an optimizer instance.

I did find one case where the Titan LR scheduler container is leaking. When retrieving the last lr, for logging, the abstraction is side-stepped and the code directly accesses the internal list of schedulers, where it retrieves the first instance to get the last-lr.

I addressed this with a hack, which is to add a "schedulers" member to the LR scheduler instance, which is a list with a reference to the schedulr in it. This seems to work, but is a little ugly.

### Validation

Similar to Trainer, a sub-class of the basic Validator has been added, which takes a Dataloader as an argument, rather than a tokenizer, and constructing the dataloader internally. The dataloader is passed throught Trainer.

A factory Protocol has also been defined for constructing a validator, to check the signature against.

## Results

The Tiny Llama trainer appears to run and produce a usable model. Further testing required.

## Next Steps

- Implement validation
- Refactor into templates for reuese
- Write more example configs and test